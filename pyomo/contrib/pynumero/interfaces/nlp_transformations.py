from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import COOSymMatrix

import numpy as np

__all__ = ['AdmmNLP']


class AdmmNLP(NLP):

    def __init__(self,
                 nlp,
                 complicating_vars,
                 rho=1.0,
                 w_estimates=None,
                 z_estimates=None):

        """

        Parameters
        ----------
        nlp: NLP
        complicating_vars: with complicated variable indices
        rho: penalty parameter

        """

        super(AdmmNLP, self).__init__(nlp.model)

        # initialize components
        self._initialize_nlp_components(nlp,
                                        complicating_vars,
                                        rho=rho,
                                        w_estimates=w_estimates,
                                        z_estimates=z_estimates)

    def _initialize_nlp_components(self, *args, **kwargs):

        nlp = args[0]
        complicating_vars = args[1]
        rho = kwargs.pop('rho', 1.0)
        w_estimates = kwargs.pop('w_estimates', None)
        z_estimates = kwargs.pop('z_estimates', None)

        self._base_nlp = nlp

        self._nx = self._base_nlp.nx
        self._ng = self._base_nlp.ng
        self._nc = self._base_nlp.nc
        self._nd = self._base_nlp.nd
        self._nnz_jac_g = self._base_nlp.nnz_jacobian_g
        self._nnz_jac_c = self._base_nlp.nnz_jacobian_c
        self._nnz_jac_d = self._base_nlp.nnz_jacobian_d

        # this gets updated later to account for regularization (rhoAtA)
        self._nnz_hess_lag = self._base_nlp.nnz_hessian_lag

        # initial point
        self._init_x = self._base_nlp.x_init()
        self._init_y = self._base_nlp.y_init()

        # bounds
        self._upper_x = self._base_nlp.xu()
        self._lower_x = self._base_nlp.xl()
        self._upper_g = self._base_nlp.gu()
        self._lower_g = self._base_nlp.gl()

        # jacobian structure # ToDo: think of removing this?
        x = self.x_init()
        jac = self._base_nlp.jacobian_g(x)
        self._irows_jac_g = jac.row
        self._jcols_jac_g = jac.col
        jac = self._base_nlp.jacobian_c(x)
        self._irows_jac_c = jac.row
        self._jcols_jac_c = jac.col
        jac = self._base_nlp.jacobian_d(x)
        self._irows_jac_d = jac.row
        self._jcols_jac_d = jac.col

        var_indices = list()
        for val in complicating_vars:
            if val > self._base_nlp.nx:
                raise RuntimeError("Variable index cannot be greater than number of vars in NLP")
            var_indices.append(val)

        # number of complicated variables
        self._nz = len(var_indices)
        self._nw = len(var_indices)

        # Populate container for complicated variables estimates
        if z_estimates is not None:
            if len(z_estimates) != self._nz:
                err_msg = "Dimension of vector does not match. \nInitial guess z "
                err_msg += "is of dimension {} but must "
                err_msg += "be of dimension {}".format(len(z_estimates), self.nz)
                raise RuntimeError(err_msg)
            self._z_values = np.asarray(z_estimates, dtype=np.double)
        else:
            self._z_values = np.zeros(self.nz)

        # container for multiplier estimates
        if w_estimates is not None:
            if len(w_estimates) != self._nw:
                err_msg = "Dimension of vector does not match. \nInitial guess w "
                err_msg += "is of dimension {} but must "
                err_msg += "be of dimension {}".format(len(w_estimates), self.nw)
                raise RuntimeError(err_msg)
            self._w_values = np.asarray(w_estimates, dtype=np.double)
        else:
            self._w_values = np.zeros(self.nw)

        # indices of complicated variables
        self._zid_to_xid = var_indices

        # complicated variable indices to zid
        self._xid_to_zid = dict()
        for zid, xid in enumerate(self._zid_to_xid):
            self._xid_to_zid[xid] = zid

        # penalty parameter
        self._rho = rho

        # update hessian structure (this includes regularization term)
        x = self.x_init()
        y = self.y_init()
        _tmp_hess = self.hessian_lag(x, y)
        self._nnz_hess_lag = _tmp_hess.nnz
        self._irows_hess = _tmp_hess.row
        self._jcols_hess = _tmp_hess.col

        # keeps masks and maps
        self._lower_x_mask = self._base_nlp._lower_x_mask
        self._upper_x_mask = self._base_nlp._upper_x_mask
        self._lower_g_mask = self._base_nlp._lower_g_mask
        self._upper_g_mask = self._base_nlp._upper_g_mask
        self._lower_d_mask = self._base_nlp._lower_d_mask
        self._upper_d_mask = self._base_nlp._upper_d_mask
        self._c_mask = self._base_nlp._c_mask
        self._d_mask = self._base_nlp._d_mask

        self._lower_x_map = self._base_nlp._lower_x_map
        self._upper_x_map = self._base_nlp._upper_x_map
        self._lower_g_map = self._base_nlp._lower_g_map
        self._upper_g_map = self._base_nlp._upper_g_map
        self._lower_d_map = self._base_nlp._lower_d_map
        self._upper_d_map = self._base_nlp._upper_d_map
        self._c_map = self._base_nlp._c_map
        self._d_map = self._base_nlp._d_map

    @property
    def nz(self):
        return self._nz

    @property
    def nw(self):
        return self._nw

    def create_vector_z(self):
        return np.zeros(self.nz, dtype=np.double)

    def create_vector_w(self):
        return np.zeros(self.nz, dtype=np.double)

    # ToDo: change these properties to methods later
    @property
    def w_estimates(self):
        """
        Return multiplier estimates in a 1d-array.
        """
        return np.copy(self._w_values)

    @w_estimates.setter
    def w_estimates(self, other):
        """
        Change multiplier estimates
        """
        if len(other) != self.nw:
            raise RuntimeError('Dimension of vector does not match')
        self._w_values = np.copy(other)

    @property
    def z_estimates(self):
        """
        Return complicated variable estimates in a 1d-array.
        """
        return np.copy(self._z_values)

    @z_estimates.setter
    def z_estimates(self, other):
        """
        Change value for complicated variable estimates
        """
        if len(other) != self.nz:
            raise RuntimeError('Dimension of vector does not match')
        self._z_values = np.copy(other)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        assert value >= 0, "Penalty parameter must be a positive number"
        self._rho = value

    def objective(self, x, **kwargs):

        z = self._z_values
        w = self._w_values

        obj = self._base_nlp.objective(x)
        # slice only the x that are "complicated"
        xs = x[self._zid_to_xid]
        difference = xs - z
        multiplier = w.dot(difference)
        penalty = 0.5 * self.rho * np.linalg.norm(difference) ** 2
        return obj + multiplier + penalty

    def grad_objective(self, x, out=None, **kwargs):

        z = self._z_values
        w = self._w_values

        df = self._base_nlp.grad_objective(x, out=out)

        # add augmentation
        # ToDo: can this be done using only numpy?
        for zid, xid in enumerate(self._zid_to_xid):
            multiplier = w[zid]
            penalty = self.rho * (x[xid] - z[zid])
            df[xid] += multiplier + penalty
        return df

    def rhs_addition_terms(self):
        a = self.create_vector_x()
        z = self._z_values
        w = self._w_values
        for zid, xid in enumerate(self._zid_to_xid):
            a[xid] = w[zid] - self.rho * z[zid]
        return a

    def atw(self):
        a = self.create_vector_x()
        for zid, xid in enumerate(self._zid_to_xid):
            a[xid] = self._w_values[zid]
        return a

    def ratbz(self):
        a = self.create_vector_x()
        for zid, xid in enumerate(self._zid_to_xid):
            a[xid] = -self.rho * self._z_values[zid]
        return a

    def evaluate_g(self, x, out=None, **kwargs):
        """Return the constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            array to store the constraint evaluations. Size ng (optional)

        Returns
        -------
        The evaluation of the constraints in a 1d-array

        """
        return self._base_nlp.evaluate_g(x, out=out)

    def evaluate_c(self, x, out=None, **kwargs):

        """Return the equality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            array to store the constraint evaluations. Size nc (optional)

        Returns
        -------
        The evaluation of the equality constraints in a 1d-array

        """
        return self._base_nlp.evaluate_c(x, out=out, **kwargs)

    def evaluate_d(self, x, out=None, **kwargs):

        """Return the inequality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            array to store the constraint evaluations. Size nd (optional)

        Returns
        -------
        The evaluation of the inequality constraints in a 1d-array

        """
        return self._base_nlp.evaluate_d(x, out=out, **kwargs)

    def jacobian_g(self, x, out=None, **kwargs):
        """Return the jacobian of constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            COOMatrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the contraints in a COOMatrix format

        """
        return self._base_nlp.jacobian_g(x, out=out)

    def jacobian_c(self, x, out=None, **kwargs):
        """Return the jacobian of equality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            COOMatrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the equality contraints in a COOMatrix format

        """
        return self._base_nlp.jacobian_c(x, out=out, **kwargs)

    def jacobian_d(self, x, out=None, **kwargs):

        """Return the jacobian of inequality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            COOMatrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the inequality contraints in a COOMatrix format

        """
        return self._base_nlp.jacobian_d(x, out=out, **kwargs)

    def hessian_lag(self, x, y, out=None, **kwargs):

        """Return the hessian of the lagrangian function evaluated at x and y

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        y : 1d-array
            array with values of dual variables. Size ng
        out : 1d-array
            SymCOOMatrix with the structure of the hessian already defined. Optional

        Returns
        -------
        SYMCOOMatrix

        """
        # ToDo: find a way to include the out because of nnz not matching
        hess = self._base_nlp.hessian_lag(x, y, **kwargs)

        append_row = np.array([xid for xid in self._zid_to_xid], dtype=hess.row.dtype)
        append_col = np.array([xid for xid in self._zid_to_xid], dtype=hess.col.dtype)
        append_data = np.ones(self.nz, dtype=np.double) * self.rho

        # this will add rho to the diagonal
        hess.row = np.concatenate([hess.row, append_row])
        hess.col = np.concatenate([hess.col, append_col])
        hess.data = np.concatenate([hess.data, append_data])

        hess.sum_duplicates()
        if out is not None:
            assert isinstance(out, COOSymMatrix), "hessian must be a COOSymMatrix"
            assert out.shape[0] == self.nx, "hessian has {} rows".format(self.nx)
            assert out.shape[1] == self.nx, "hessian has {} columns".format(self.nx)
            assert out.nnz == self.nnz_hessian_lag, "hessian has {} nnz".format(self.nnz_hessian_lag)
            out.data = hess.data
        return hess

    # ToDo: variable order?













