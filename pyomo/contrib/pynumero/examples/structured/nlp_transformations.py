#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import empty_matrix

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import pyomo.environ as aml
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
        self._upper_d = self._base_nlp.du()
        self._lower_d = self._base_nlp.dl()

        x = self.x_init()

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
        self._zid_to_xid = np.array(var_indices)

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

        # expansion matrix z to x
        col = self._zid_to_xid
        row = np.arange(self.nz, dtype=np.int)
        data = np.ones(self.nz)
        self._compress_x_to_z = csr_matrix((data, (row, col)), shape=(self.nz, nlp.nx))
        self._expand_z_to_x = self._compress_x_to_z.transpose()

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

    def w_estimates(self):
        """
        Return multiplier estimates in a 1d-array.
        """
        return np.copy(self._w_values)

    def set_w_estimates(self, other):
        """
        Change multiplier estimates
        """
        if len(other) != self.nw:
            raise RuntimeError('Dimension of vector does not match')
        self._w_values = np.copy(other)

    def z_estimates(self):
        """
        Return complicated variable estimates in a 1d-array.
        """
        return np.copy(self._z_values)

    def set_z_estimates(self, other):
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
        xs = x[self._zid_to_xid]
        penalty = self.rho * (xs-z)
        df += self._expand_z_to_x.dot(penalty+w)
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
        return self._base_nlp.evaluate_g(x, out=out, **kwargs)

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
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the contraints in a coo_matrix format

        """
        return self._base_nlp.jacobian_g(x, out=out, **kwargs)

    def jacobian_c(self, x, out=None, **kwargs):
        """Return the jacobian of equality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the equality contraints in a coo_matrix format

        """
        return self._base_nlp.jacobian_c(x, out=out, **kwargs)

    def jacobian_d(self, x, out=None, **kwargs):

        """Return the jacobian of inequality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the inequality contraints in a coo_matrix format

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
            coo_matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        coo_matrix

        """
        hess = self._base_nlp.hessian_lag(x, y, **kwargs)

        append_row = self._zid_to_xid
        append_col = self._zid_to_xid
        append_data = np.ones(self.nz, dtype=np.double)
        append_data.fill(self.rho)

        # this will add rho to the diagonal
        hess.row = np.concatenate([hess.row, append_row])
        hess.col = np.concatenate([hess.col, append_col])
        hess.data = np.concatenate([hess.data, append_data])

        hess.sum_duplicates()

        if out is not None:
            assert isinstance(out, coo_matrix), "hessian must be a coo_matrix"
            assert out.shape[0] == self.nx, "hessian has {} rows".format(self.nx)
            assert out.shape[1] == self.nx, "hessian has {} columns".format(self.nx)
            assert out.nnz == self.nnz_hessian_lag, "hessian has {} nnz".format(self.nnz_hessian_lag)
            out.data = hess.data
        return hess

    def variable_order(self):
        return self._base_nlp.variable_order()

    def constraint_order(self):
        return self._base_nlp.constraint_order()


class AugmentedLagrangianNLP(NLP):

    def __init__(self,
                 nlp,
                 rho=1.0,
                 w_estimates=None):
        """

        Parameters
        ----------
        nlp: PyomoNLP
        subset_constraints: list of pyomo constraints
        """

        super(AugmentedLagrangianNLP, self).__init__(nlp.model)
        self._initialize_nlp_components(nlp, rho=rho, w_estimates=w_estimates)

    def _initialize_nlp_components(self, *args, **kwargs):

        nlp = args[0]
        rho = kwargs.pop('rho', 1.0)
        w_estimates = kwargs.pop('w_estimates', None)

        self._base_nlp = nlp
        self._rho = rho

        self._nx = self._base_nlp.nx
        self._ng = self._base_nlp.nd
        self._nc = 0
        self._nd = self._base_nlp.nd
        self._nw = self._base_nlp.nc

        self._nnz_jac_g = self._base_nlp.nnz_jacobian_d
        self._nnz_jac_c = 0
        self._nnz_jac_d = self._base_nlp.nnz_jacobian_d

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

        self._lower_x_mask = self._base_nlp._lower_x_mask
        self._upper_x_mask = self._base_nlp._upper_x_mask

        self.Pc = self._base_nlp.expansion_matrix_c()
        self.Pd = self._base_nlp.expansion_matrix_d()

        x = self._base_nlp.x_init()
        y = self._base_nlp.y_init()
        self._init_x = x
        self._init_y = self.Pd.transpose().dot(y)

        # update hessian structure (this includes regularization term)
        y = self._init_y
        _tmp_hess = self.hessian_lag(x, y)
        self._nnz_hess_lag = _tmp_hess.nnz

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        assert value >= 0, "Penalty parameter must be a positive number"
        self._rho = value

    @property
    def nw(self):
        return self._nw

    def w_estimates(self):
        """
        Return multiplier estimates in a 1d-array.
        """
        return np.copy(self._w_values)

    def set_w_estimates(self, other):
        """
        Change multiplier estimates
        """
        if len(other) != self.nw:
            raise RuntimeError('Dimension of vector does not match')
        self._w_values = np.copy(other)

    def create_vector_w(self):
        return np.zeros(self.nz, dtype=np.double)

    def objective(self, x, **kwargs):

        y = self._w_values
        obj = self._base_nlp.objective(x)
        h = self._base_nlp.evaluate_c(x)
        return obj + y.dot(h) + 0.5 * self.rho * h.dot(h)

    def grad_objective(self, x, out=None, **kwargs):

        y = self._w_values
        y_tilde = y + self.rho * self._base_nlp.evaluate_c(x)
        jac = self._base_nlp.jacobian_c(x)

        if out is None:
            df = self._base_nlp.grad_objective(x, out=out)
            return df + jac.transpose().dot(y_tilde)
        else:
            self._base_nlp.grad_objective(x, out=out)
            out += + jac.transpose().dot(y_tilde)
            return out

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
        return np.zeros(0)

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

    def evaluate_g(self, x, out=None, **kwargs):

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
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the contraints in a coo_matrix format

        """
        return self._base_nlp.jacobian_d(x, out=out)

    def jacobian_c(self, x, out=None, **kwargs):
        """Return the jacobian of equality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the equality contraints in a coo_matrix format

        """
        return empty_matrix(0, self.nx)

    def jacobian_d(self, x, out=None, **kwargs):

        """Return the jacobian of inequality constraints evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            coo_matrix with the structure of the jacobian already defined. Optional

        Returns
        -------
        The jacobian of the inequality contraints in a coo_matrix format

        """
        return self._base_nlp.jacobian_d(x, out=out, **kwargs)

    def hessian_lag(self, x, y, out=None, **kwargs):
        # y should only include yd
        assert y.size == self.nd
        Pc = self.Pc
        Pd = self.Pd
        yd = Pd.dot(y)
        y_tilde = self._w_values + self.rho * self._base_nlp.evaluate_c(x)
        yc = Pc.dot(y_tilde)
        yall = yd + yc
        hess = self._base_nlp.hessian_lag(x, yall, **kwargs)
        jac = self._base_nlp.jacobian_c(x)
        rjac = self.rho * jac.transpose().dot(jac)
        res = hess + rjac
        return res.tocoo()

    def expansion_matrix_c(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = np.zeros(0)
        nnz = 0
        col = np.arange(nnz, dtype=np.int)
        data = np.zeros(nnz)
        return csr_matrix((data, (row, col)), shape=(self.ng, nnz))

    def expansion_matrix_d(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = self._base_nlp._d_map
        nnz = len(self._base_nlp._d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.ng, nnz))

    def expansion_matrix_du(self):
        """
        Returns expansion matrix upper bounds on inequality constraints
        """
        row = self._base_nlp._upper_d_map
        nnz = len(self._base_nlp._upper_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nd, nnz))

    def expansion_matrix_dl(self):
        """
        Returns expansion matrix lower bounds on inequality constraints
        """

        row = self._base_nlp._lower_d_map
        nnz = len(self._base_nlp._lower_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nd, nnz))

    def expansion_matrix_xu(self):
        """
        Returns expansion matrix for upper bounds on primal variables
        """
        return self._base_nlp.expansion_matrix_xu()

    def expansion_matrix_xl(self):
        """
        Returns expansion matrix for lower bounds on primal variables
        """
        return self._base_nlp.expansion_matrix_xl()


def compose_two_stage_stochastic_model(models, complicating_vars):

    if not isinstance(models, dict):
        raise RuntimeError("Model must be a dictionary")
    if not isinstance(complicating_vars, dict):
        raise RuntimeError("complicating_vars must be a dictionary")
    if len(complicating_vars) != len(models):
        raise RuntimeError("Each scenario must have a list of complicated variables")

    counter = 0
    nz = -1
    for k, v in complicating_vars.items():
        if counter == 0:
            nz = len(v)
        else:
            assert len(v) == nz, 'all models must have same number of complicating variables'
        counter += 1

    model = aml.ConcreteModel()
    model.z = aml.Var(range(nz))
    model.scenario_names = sorted([name for name in models.keys()])

    obj = 0.0

    for i, j in enumerate(model.scenario_names):
        instance = models[j].clone()
        model.add_component("{}_linking".format(j), aml.ConstraintList())
        model.add_component("{}".format(j), instance)
        linking = getattr(model, "{}_linking".format(j))
        x = complicating_vars[j]

        for k, var in enumerate(x):
            if var.is_indexed():
                raise RuntimeError('indexed complicating variables not supported')
            vid = aml.ComponentUID(var)
            vv = vid.find_component_on(instance)
            linking.add(vv == model.z[k])

        # gets objective
        objectives = instance.component_map(aml.Objective, active=True)
        if len(objectives) > 1:
            raise RuntimeError('Multiple objectives not supported')
        instance_obj = list(objectives.values())[0]
        obj += instance_obj.expr
        instance_obj.deactivate()
    model.obj = aml.Objective(expr=obj)
    return model