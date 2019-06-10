#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyomo
import pyomo.environ as aml

try:
    import pyomo.contrib.pynumero.extensions.asl as _asl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing asl while running nlp interface. '
                      'Make sure libpynumero_ASL is installed and added to path.')

from pyomo.contrib.pynumero.sparse import empty_matrix
from scipy.sparse import coo_matrix, csr_matrix
from .nlp import NLP
import numpy as np


class EqualityQuadraticModel(object):

    def __init__(self, Q, c, d, A, b):

        nx = Q.shape[1]
        assert Q.shape[1] == Q.shape[0]
        assert A.shape[1] == nx
        assert A.shape[0] == b.size
        assert c.size == nx
        assert isinstance(Q, coo_matrix)
        assert isinstance(A, coo_matrix)

        self.hessian = Q
        self.jacobian = A
        self.linear_obj_term = c
        self.scalar_obj_term = d
        self.rhs = b


class EqualityQP(NLP):

    def __init__(self, model, **kwargs):

        assert isinstance(model, EqualityQuadraticModel)
        super(EqualityQP, self).__init__(model)

        # initialize components
        self._initialize_nlp_components()

        # make pointer unmutable from outside world
        self._make_unmutable_caches()

    def _initialize_nlp_components(self, *args, **kwargs):

        model = self._model

        # dimensions
        self._nx = model.hessian.shape[0]
        self._ng = model.jacobian.shape[0]
        self._nnz_jac_g = model.jacobian.nnz
        self._nnz_jac_c = model.jacobian.nnz
        self._nnz_jac_d = 0
        self._nnz_hess_lag = model.hessian.nnz

        # initial point
        self._init_x = np.zeros(self.nx)
        self._init_y = np.zeros(self.ng)

        # bounds on x
        self._upper_x = np.full((self.nx, 1), np.inf)
        self._lower_x = np.full((self.nx, 1), -np.inf)
        self._upper_x = self._upper_x.flatten()
        self._lower_x = self._lower_x.flatten()

        # bounds on g
        self._upper_g = np.copy(model.rhs)
        self._lower_g = np.copy(model.rhs)

        self._lower_x_mask = np.zeros(self.nx, dtype=bool)
        self._lower_x_map = np.zeros(0)
        self._upper_x_mask = np.zeros(self.nx, dtype=bool)
        self._upper_x_map = np.zeros(0)

        self._c_mask = np.ones(self.ng, dtype=bool)
        self._c_map = np.arange(self.ng)
        self._d_mask = np.zeros(self.ng, dtype=bool)
        self._d_map = np.zeros(0)

        self._lower_g_mask = np.ones(self.ng, dtype=bool)
        self._lower_g_map = np.arange(self.ng)
        self._upper_g_mask = np.ones(self.ng, dtype=bool)
        self._upper_g_map = np.arange(self.ng)

        self._lower_d_mask = np.zeros(0, dtype=bool)
        self._lower_d_map = np.zeros(0)
        self._upper_d_mask = np.zeros(0, dtype=bool)
        self._upper_d_map = np.zeros(0)

        # define bounds on d extracted from g
        self._lower_d = np.compress(self._d_mask, self._lower_g)
        self._upper_d = np.compress(self._d_mask, self._upper_g)

        # internal pointer for evaluation of g
        self._g_rhs = self._upper_g.copy()
        self._g_rhs[~self._c_mask] = 0.0
        self._lower_g[self._c_mask] = 0.0
        self._upper_g[self._c_mask] = 0.0

        # set number of equatity and inequality constraints from maps
        self._nc = len(self._c_map)
        self._nd = len(self._d_map)

    def _make_unmutable_caches(self):
        """
        Sets writable flag of internal arrays (cached) to false
        """
        self._lower_x.flags.writeable = False
        self._upper_x.flags.writeable = False
        self._lower_g.flags.writeable = False
        self._upper_g.flags.writeable = False
        self._init_x.flags.writeable = False
        self._init_y.flags.writeable = False

        # make maps and masks not rewritable
        self._c_mask.flags.writeable = False
        self._c_map.flags.writeable = False
        self._d_mask.flags.writeable = False
        self._d_map.flags.writeable = False

        self._lower_x_mask.flags.writeable = False
        self._upper_x_mask.flags.writeable = False
        self._lower_g_mask.flags.writeable = False
        self._upper_g_mask.flags.writeable = False
        self._lower_d_mask.flags.writeable = False
        self._upper_d_mask.flags.writeable = False

        self._lower_x_map.flags.writeable = False
        self._upper_x_map.flags.writeable = False
        self._lower_g_map.flags.writeable = False
        self._upper_g_map.flags.writeable = False
        self._lower_d_map.flags.writeable = False
        self._upper_d_map.flags.writeable = False

    @property
    def model(self):
        """
        Return optimization model
        """
        return self._model

    def objective(self, x, **kwargs):
        """Returns value of objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.

        Returns
        -------
        float

        """
        model = self._model
        Q = model.hessian
        c = model.linear_obj_term
        d = model.scalar_obj_term
        f = Q.dot(x)
        return 0.5 * x.dot(f) + c.dot(x) + d

    def grad_objective(self, x, out=None, **kwargs):
        """Returns gradient of the objective function evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        model = self._model
        Q = model.hessian
        c = model.linear_obj_term
        if out is None:
            df = Q.dot(x) + c
        else:
            msg = "grad_objective takes a ndarray of size {}".format(self.nx)
            assert isinstance(out, np.ndarray) and out.size == self.nx, msg
            np.copyto(out, Q.dot(x) + c)
            df = out

        return df

    def evaluate_g(self, x, out=None, **kwargs):
        """Return general inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        model = self._model
        A = model.jacobian
        b = model.rhs
        if out is None:
            res = A.dot(x) - b
        else:
            msg = "evaluate_g takes a ndarray of size {}".format(self.ng)
            assert isinstance(out, np.ndarray) and out.size == self.ng, msg
            np.copyto(out, A.dot(x) - b)
            res = out

        return res

    def evaluate_c(self, x, out=None, **kwargs):
        """Returns the equality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """
        evaluated_g = kwargs.pop('evaluated_g', None)
        model = self._model
        A = model.jacobian
        b = model.rhs

        if evaluated_g is None:
            res = A.dot(x) - b
            eval_g = res
        else:
            msg = "evaluate_c takes a ndarray of size {} for evaluated_g".format(self.ng)
            assert isinstance(evaluated_g, np.ndarray) and evaluated_g.size == self.ng, msg
            eval_g = evaluated_g

        if out is not None:
            msg = "evaluate_c takes a ndarray of size {}".format(self.nc)
            assert isinstance(out, np.ndarray) and out.size == self.nc, msg
            np.copyto(out, eval_g)

        return eval_g

    def evaluate_d(self, x, out=None, **kwargs):
        """Returns the inequality constraints evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : array_like
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        array_like

        """

        if out is not None:
            msg = "evaluate_d takes a ndarray of size {}".format(self.nd)
            assert isinstance(out, np.ndarray) and out.size == self.nd, msg
            out.fill(0.0)
        return np.zeros(self.nd)

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        coo_matrix

        """
        model = self._model
        A = model.jacobian
        if out is None:
            jac = A.copy()
        else:
            assert isinstance(out, coo_matrix), "jacobian_g must be a COOMatrix"
            assert out.shape[0] == self.ng, "jacobian_g has {} rows".format(self.ng)
            assert out.shape[1] == self.nx, "jacobian_g has {} columns".format(self.nx)
            assert out.nnz == self.nnz_jacobian_g, "jacobian_g has {} nnz".format(self.nnz_jacobian_g)

            out.data = A.data.copy()
            jac = out

        return jac

    def jacobian_c(self, x, out=None, **kwargs):
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        coo_matrix

        """
        evaluated_jac_g = kwargs.pop('evaluated_jac_g', None)

        model = self._model
        A = model.jacobian

        if out is not None:
            assert isinstance(out, coo_matrix), "jacobian_c must be a COOMatrix"
            assert out.shape[0] == self.nc, "jacobian_c has {} rows".format(self.nc)
            assert out.shape[1] == self.nx, "jacobian_c has {} columns".format(self.nx)
            out.data = A.data
            jac = out
        else:
            jac = A

        return jac

    def jacobian_d(self, x, out=None, **kwargs):
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : coo_matrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        coo_matrix

        """

        evaluated_jac_g = kwargs.pop('evaluated_jac_g', None)

        if out is not None:
            assert isinstance(out, coo_matrix), "jacobian_d must be a COOMatrix"
            assert out.shape[0] == self.nd, "jacobian_d has {} rows".format(self.nd)
            assert out.shape[1] == self.nx, "jacobian_d has {} columns".format(self.nx)
            out.data = np.zeros(0)
            jac = out
        else:
            jac = empty_matrix(0, self.nx)

        return jac

    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : coo_matrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        coo_matrix

        """

        model = self._model
        Q = model.hessian

        if out is None:
            hess = Q.copy()
        else:
            assert isinstance(out, coo_matrix), "hessian must be a COOSymMatrix"
            assert out.shape[0] == self.nx, "hessian has {} rows".format(self.nx)
            assert out.shape[1] == self.nx, "hessian has {} columns".format(self.nx)
            assert out.nnz == self.nnz_hessian_lag, "hessian has {} nnz".format(self.nnz_hessian_lag)

            out.data = Q.data.copy()
            hess = out

        return hess

    def create_vector(self, vector_type):

        if vector_type == 'x':
            return np.zeros(self.nx, dtype=np.double)
        elif vector_type == 'xl':
            nx_l = len(self._lower_x_map)
            return np.zeros(nx_l, dtype=np.double)
        elif vector_type == 'xu':
            nx_u = len(self._upper_x_map)
            return np.zeros(nx_u, dtype=np.double)
        elif vector_type == 'g' or vector_type == 'y':
            return np.zeros(self.ng, dtype=np.double)
        elif vector_type == 'c' or vector_type == 'yc':
            return np.zeros(self.nc, dtype=np.double)
        elif vector_type == 'd' or vector_type == 'yd' or vector_type == 's':
            return np.zeros(self.nd, dtype=np.double)
        elif vector_type == 'dl':
            ndl = len(self._lower_d_map)
            return np.zeros(ndl, dtype=np.double)
        elif vector_type == 'du':
            ndu = len(self._upper_d_map)
            return np.zeros(ndu, dtype=np.double)
        else:
            raise RuntimeError('vector_type not recognized')

    def projection_matrix_xl(self):
        """
        Returns expansion matrix for lower bounds on primal variables
        """
        row = self._lower_x_map
        nnz = len(self._lower_x_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz))

    def projection_matrix_xu(self):
        """
        Returns expansion matrix for upper bounds on primal variables
        """
        row = self._upper_x_map
        nnz = len(self._upper_x_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz))

    def projection_matrix_dl(self):
        """
        Returns expansion matrix lower bounds on inequality constraints
        """

        row = self._lower_d_map
        nnz = len(self._lower_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nd, nnz))

    def projection_matrix_du(self):
        """
        Returns expansion matrix upper bounds on inequality constraints
        """
        row = self._upper_d_map
        nnz = len(self._upper_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nd, nnz))

    def projection_matrix_d(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = self._d_map
        nnz = len(self._d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.ng, nnz))

    def projection_matrix_c(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = self._c_map
        nnz = len(self._c_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.ng, nnz))

    def report_solver_status(self, status_num, status_msg, x, y):
        raise NotImplementedError('EqualityQP does not support report_solver_status')
