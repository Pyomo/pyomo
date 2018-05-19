from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP, NLP
from pyomo.core.base import ComponentMap
import numpy as np


class PyomoADMMNLP(NLP):

    def __init__(self, model, complicated_vars, rho=0.0):

        super(PyomoADMMNLP, self).__init__(model)
        self._base_nlp = PyomoNLP(model)

        self._nx = self._base_nlp.nx
        self._ng = self._base_nlp.ng
        self._nc = self._base_nlp.nc
        self._nd = self._base_nlp.nd
        self._nnz_jac_g = self._base_nlp.nnz_jacobian_g
        self._nnz_jac_c = self._base_nlp.nnz_jacobian_c
        self._nnz_jac_d = self._base_nlp.nnz_jacobian_d
        # this gets updated later in the constructor
        self._nnz_hess_lag = self._base_nlp.nnz_hessian_lag

        # initial point
        self._init_x = self._base_nlp.x_init
        self._init_y = self._base_nlp.y_init

        # bounds
        self._upper_x = self._base_nlp.xu
        self._lower_x = self._base_nlp.xl
        self._upper_g = self._base_nlp.gu
        self._lower_g = self._base_nlp.gl

        # jacobian structure
        self._irows_jac_g = self._base_nlp._irows_jac_g
        self._jcols_jac_g = self._base_nlp._jcols_jac_g
        self._irows_jac_c = self._base_nlp._irows_jac_c
        self._jcols_jac_c = self._base_nlp._jcols_jac_c
        self._irows_jac_d = self._base_nlp._irows_jac_d
        self._jcols_jac_d = self._base_nlp._jcols_jac_d


        var_indices = []
        z_vars = []
        for v in complicated_vars:
            if v.is_indexed():
                for vd in v.values():
                    var_id = self._base_nlp._varToIndex[vd]
                    var_indices.append(var_id)
                    z_vars.append(vd)
            else:
                var_id = self._base_nlp._varToIndex[v]
                var_indices.append(var_id)
                z_vars.append(v)

        # number of complicated variables
        self._nz = len(var_indices)

        # container for complicated variables estimates
        self._z_values = np.zeros(self.nz)

        # container for multiplier estimates
        self._w_values = np.zeros(self.nw)

        # indices of complicated variables
        self._zid_to_xid = var_indices

        # complicated variable indices to zid
        self._xid_to_zid = dict()
        for zid, xid in enumerate(self._zid_to_xid):
            self._xid_to_zid[xid] = zid

        # complicated variables (pointers to pyomo model)
        self._zid_to_xvar = z_vars

        # complicated variable to zid
        self._xvar_to_zid = ComponentMap()
        for zid, xvar in enumerate(self._zid_to_xvar):
            self._xvar_to_zid[xvar] = zid

        # penalty parameter
        self._rho = rho

        # update hessian structure
        x = self.x_init
        y = self.y_init
        _tmp_hess = self.hessian_lag(x, y)
        self._nnz_hess_lag = _tmp_hess.nnz
        self._irows_hess = _tmp_hess.row
        self._jcols_hess = _tmp_hess.col

    @property
    def xl(self):
        """
        Return lower bounds of primal variables in a 1d-array.
        """
        return self._lower_x

    @property
    def xu(self):
        """
        Return upper bounds of primal variables in a 1d-array.
        """
        return self._upper_x

    @property
    def gl(self):
        """
        Return lower bounds of general inequality constraints.
        in a 1d-array
        """
        return self._lower_g

    @property
    def gu(self):
        """
        Return upper bounds of general inequality constraints.
        in a 1d-array
        """
        return self._upper_g

    @property
    def x_init(self):
        """
        Return initial guess of primal variables in a 1d-array.
        """
        return np.copy(self._init_x)

    @property
    def y_init(self):
        """
        Return initial guess of dual variables in a 1d-array.
        """
        return np.copy(self._init_y)

    @xl.setter
    def xl(self, other):
        """
        Prevent changing lower bounds of primal variables
        """
        raise RuntimeError('Changing bounds not supported for now')

    @xu.setter
    def xu(self, other):
        """
        Prevent changing upper bounds of primal variables
        """
        raise RuntimeError('Changing bounds not supported for now')

    @gl.setter
    def gl(self, other):
        """
        Prevent changing lower bounds of constraints
        """
        raise NotImplementedError('Abstract class method')

    @gu.setter
    def gu(self, other):
        """
        Prevent changing upper bounds of constraints
        """
        raise NotImplementedError('Abstract class method')

    @x_init.setter
    def x_init(self, other):
        """
        Change initial guesses of primal variables
        """
        if len(other) != self.nx:
            raise RuntimeError('Dimension of vector does not match')
        self._init_x = np.copy(other)

    @y_init.setter
    def y_init(self, other):
        """
        Change initial guesses of dual variables
        """
        if len(other) != self.ng:
            raise RuntimeError('Dimension of vector does not match')
        self._init_y = np.copy(other)

    def create_vector_x(self, subset=None):
        """Return 1d-array of vector of primal variables

        Parameters
        ----------
        subset : str
            type of vector. xl returns a vector of
            variables with lower bounds. xu returns a
            vector of variables with upper bounds (optional)

        Returns
        -------
        1d-array

        """
        if subset is None:
            return np.zeros(self.nx, dtype=np.double)
        elif subset == 'l':
            nx_l = len(self._lower_x_map)
            return np.zeros(nx_l, dtype=np.double)
        elif subset == 'u':
            nx_u = len(self._upper_x_map)
            return np.zeros(nx_u, dtype=np.double)
        else:
            raise RuntimeError('Subset not recognized')

    def create_vector_y(self, subset=None):
        """Return 1d-array of vector of constraints

        Parameters
        ----------
        subset : str
            type of vector. yd returns a vector of
            inequality constriants. yc returns a
            vector of equality constraints (optional)

        Returns
        -------
        1d-array

        """
        if subset is None:
            return np.zeros(self.ng, dtype=np.double)
        elif subset == 'c':
            return np.zeros(self.nc, dtype=np.double)
        elif subset == 'd':
            return np.zeros(self.nd, dtype=np.double)
        else:
            raise RuntimeError('Subset not recognized')

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        raise NotImplementedError('Action not supported')

    @property
    def nw(self):
        return self._nz

    @nw.setter
    def nw(self, value):
        raise NotImplementedError('Action not supported')

    def create_vector_z(self):
        return np.zeros(self.nz, dtype=np.double)

    def create_vector_w(self):
        return np.zeros(self.nz, dtype=np.double)

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

    def complicated_variable_order(self):
        var_order = [None] * self.nx
        for zid, v in enumerate(self._zid_to_xvar):
            var_order[zid] = v.name
        return var_order

    def objective(self, x):

        z = self._z_values
        w = self._w_values

        obj = self._base_nlp.objective(x)

        # slice only the x that are "complicated"
        xs = x[self._zid_to_xid]
        difference = xs - z
        multiplier = w.dot(difference)
        penalty = 0.5 * self.rho * np.linalg.norm(difference) ** 2

        return obj + multiplier + penalty

    def grad_objective(self, x, out=None):

        z = self._z_values
        w = self._w_values

        df = self._base_nlp.grad_objective(x, out=out)

        # add augmentation
        for zid, xid in enumerate(self._zid_to_xid):
            multiplier = w[zid]
            penalty = self.rho * (x[xid] - z[zid])
            df[xid] += multiplier + penalty
        return df

    def evaluate_g(self, x, out=None):
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

    def jacobian_g(self, x, out=None):
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
        The hessian of the Lagrangian function in a SYMCOOMatrix format

        """

        hess = self._base_nlp.hessian_lag(x, y, out=out, **kwargs)

        append_row = np.array([xid for xid in self._zid_to_xid])
        append_col = np.array([xid for xid in self._zid_to_xid])
        append_data = np.ones(self.nz, dtype=np.double) * self.rho

        # this will add rho to the diagonal
        hess.row = np.concatenate([hess.row, append_row])
        hess.col = np.concatenate([hess.col, append_col])
        hess.data = np.concatenate([hess.data, append_data])

        hess.sum_duplicates()
        return hess

    def finalize_solution(self, status, x, y):
        """
        Write .sol file

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        y : 1d-array
            array with values of dual variables. Size ng

        Returns
        -------
        None
        """
        self._base_nlp.finalize_solution(status, x, y)



















