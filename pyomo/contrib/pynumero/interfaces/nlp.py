"""
The pyomo.contrib.pynumero.interfaces.nlp module includes methods to query nonlinear
programming problems. The NLPs considered in this module have
the following form:

minimize             f(x)
subject to    g_L <= g(x) <= g_U
              x_L <=  x   <= x_U

where x \in R^{n_x} are the primal variables,
      x_L \in R^{n_x} are the lower bounds of the primal variables,
      x_U \in R^{n_x} are the uppper bounds of the primal variables,
      g: R^{nx} \rightarrow R^{n_g} are general inequality constraints
      bounded by g_L \in R^{n_g}, and g_U \in R^{n_g}
      y: \in R^{n_g} are the dual variables corresponding to the
      general inequality constraints

The general inequalitites can be further query to obtain equality and
strict inequality constraints. In such case the NLP problem considered
has the following form

minimize             f(x)
subject to         c(x) = 0
              d_L <= d(x) <= d_U
              x_L <=  x   <= x_U

where c: R^{n_x} \rightarrow R^{n_c} are the equality constraints
      d: R^{n_x} \rightarrow R^{n_d} are the inequality constraints
      note that nc + nd = ng
.. rubric:: Contents

"""

try:
    import pyomo.contrib.pynumero.extensions.asl as _asl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing asl while running nlp interface. '
                      'Make sure libpynumero_ASL is installed and added to path.')

from pyomo.contrib.pynumero.sparse import (COOMatrix,
                             COOSymMatrix,
                             CSCMatrix,
                             CSCSymMatrix,
                             CSRMatrix,
                             CSRSymMatrix)
from scipy.sparse import dok_matrix
import abc
import numpy as np
import tempfile
import os
import six
import shutil

__all__ = ['AmplNLP', 'PyomoNLP']


@six.add_metaclass(abc.ABCMeta)
class NLP(object):
    """
    Abstract class for nonlinear program

    Parameters
    -------------------
    model: object containing an optimization model. This is changes,
    for the different subclasses considered. PyomoNLP for instance
    takes a pyomo model, AmplNLP takes a standard nl file. For composite
    models the subclass takes a scenario tree for pysp.
    """

    def __init__(self, model, **kwargs):


        self._model = model
        self._nx = 0
        self._ng = 0
        self._nc = 0
        self._nd = 0
        self._nnz_jac_g = 0
        self._nnz_jac_c = 0
        self._nnz_jac_d = 0
        self._nnz_hess_lag = 0

        # initial point
        self._init_x = None
        self._init_y = None

        # bounds
        self._upper_x = None
        self._lower_x = None
        self._upper_g = None
        self._lower_g = None

        # jacobian structure
        self._irows_jac_g = None
        self._jcols_jac_g = None
        self._irows_jac_c = None
        self._jcols_jac_c = None
        self._irows_jac_d = None
        self._jcols_jac_d = None

        # hessian structure
        self._irows_hess = None
        self._jcols_hess = None

    @property
    def nx(self):
        """
        Return the number of variables.
        """
        return self._nx

    @property
    def ng(self):
        """
        Return the total number of constraints.
        """
        return self._ng

    @property
    def nc(self):
        """
        Return the number of equality constraints.
        """
        return self._nc

    @property
    def nd(self):
        """
        Return the number of inequality constraints.
        """
        return self._nd

    @property
    def nnz_jacobian_g(self):
        """
        Return the number of nonzero values in the
        jacobian of all constraints.
        """
        return self._nnz_jac_g

    @property
    def nnz_jacobian_c(self):
        """
        Return the number of nonzero values in the
        jacobian of equality constraints (subset of g).
        """
        return self._nnz_jac_c

    @property
    def nnz_jacobian_d(self):
        """
        Return the number of nonzero values in the
        jacobian of inequality constraints (subset of g).
        """
        return self._nnz_jac_d

    @property
    def nnz_hessian_lag(self):
        """
        Return the number of nonzero values in the
        hessian of the Lagrangian function.
        """
        return self._nnz_hess_lag

    @property
    def xl(self):
        """
        Return lower bounds of primal variables in a 1d-array.
        """
        raise NotImplementedError('Abstract class method')

    @property
    def xu(self):
        """
        Return upper bounds of primal variables in a 1d-array.
        """
        raise NotImplementedError('Abstract class method')

    @property
    def gl(self):
        """
        Return lower bounds of general inequality constraints.
        in a 1d-array
        """
        raise NotImplementedError('Abstract class method')

    @property
    def gu(self):
        """
        Return upper bounds of general inequality constraints.
        in a 1d-array
        """
        raise NotImplementedError('Abstract class method')

    @property
    def x_init(self):
        """
        Return initial guess of primal variables in a 1d-array.
        """
        raise NotImplementedError('Abstract class method')

    @property
    def y_init(self):
        """
        Return initial guess of dual variables in a 1d-array.
        """
        raise NotImplementedError('Abstract class method')

    @nx.setter
    def nx(self, value):
        """
        Prevent changing number of primal variables.
        """
        raise RuntimeError('Cannot change number of variables')

    @nc.setter
    def nc(self, value):
        """
        Prevent changing number of equality constraints.
        """
        raise RuntimeError('Cannot change number of constraints')

    @ng.setter
    def ng(self, value):
        """
        Prevent changing number of general inequality constraints.
        """
        raise RuntimeError('Cannot change number of constraints')

    @nd.setter
    def nd(self, value):
        """
        Prevent changing number of inequality constraints.
        """
        raise RuntimeError('Cannot change number of constraints')

    @nnz_jacobian_g.setter
    def nnz_jacobian_g(self, value):
        """
        Prevent changing number of nonzero values in jacobian
        of all constraints.
        """
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_jacobian_c.setter
    def nnz_jacobian_c(self, value):
        """
        Prevent changing number of nonzero values in jacobian
        of equality constraints.
        """
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_jacobian_d.setter
    def nnz_jacobian_d(self, value):
        """
        Prevent changing number of nonzero values in jacobian
        of inequality constraints.
        """
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_hessian_lag.setter
    def nnz_hessian_lag(self, value):
        """
        Prevent changing number of nonzero values in hessian
        of the Lagrangian Function.
        """
        raise RuntimeError('Cannot change number of nonzeros in hessian')

    @xl.setter
    def xl(self, other):
        """
        Change lower bounds of primal variables
        """
        raise NotImplementedError('Abstract class method')

    @xu.setter
    def xu(self, other):
        """
        Change upper bounds of primal variables
        """
        raise NotImplementedError('Abstract class method')

    @gl.setter
    def gl(self):
        """
        Change lower bounds of constraints
        """
        raise NotImplementedError('Abstract class method')

    @gu.setter
    def gu(self):
        """
        Change upper bounds of constraints
        """
        raise NotImplementedError('Abstract class method')

    @x_init.setter
    def x_init(self, other):
        """
        Change initial guesses of primal variables
        """
        raise NotImplementedError('Abstract class method')

    @y_init.setter
    def y_init(self, other):
        """
        Change initial guesses of dual variables
        """
        raise NotImplementedError('Abstract class method')

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
    def objective(self, x):
        """Return value of objective function evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx

        Returns
        -------
        The value of the objective function

        """
        return

    @abc.abstractmethod
    def grad_objective(self, x, out=None):
        """Return gradient of the objective function evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        out : 1d-array
            array to store the gradient of the objective function (optional)

        Returns
        -------
        The gradient of the objective function in a 1d-array

        """
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return


class StubNLP(NLP):
    """
    Nonlinear programm interface that relies on the Ampl solver library

    Parameters
    -------------------
    model: string
    filename of the NL-file containing the model
    row_filename: string (optional)
    filename of .row file with identity of constraints (optional)
    col_filename: string (optional)
    filename of .col file with identity of variables (optional)
    """

    # ToDo: add access to d_l d_u
    # ToDo: change on x_l and x_u, g_u, g_l?

    def __init__(self, model):

        # call parent class to set model
        super(StubNLP, self).__init__(model)

        # ampl interface
        self._asl = _asl.AmplInterface(self._model)

        # dimensions
        self._nx = self._asl.get_n_vars()
        self._ng = self._asl.get_n_constraints()

        self._nnz_jac_g = self._asl.get_nnz_jac_g()
        self._nnz_hess_lag = self._asl.get_nnz_hessian_lag()

        # initial point
        self._init_x = np.zeros(self.nx)
        self._init_y = np.zeros(self.ng)
        self._asl.get_starting_point(self._init_x, self._init_y)

        # bounds on x
        self._upper_x = np.zeros(self.nx, dtype=np.double)
        self._lower_x = np.zeros(self.nx, dtype=np.double)
        self._asl.get_x_lower_bounds(self._lower_x)
        self._asl.get_x_upper_bounds(self._upper_x)

        # build maps for x
        self._build_x_maps()

        # bounds on g
        self._upper_g = np.zeros(self.ng, dtype=np.double)
        self._lower_g = np.zeros(self.ng, dtype=np.double)
        self._asl.get_g_lower_bounds(self._lower_g)
        self._asl.get_g_upper_bounds(self._upper_g)

        # build maps for g
        self._build_g_maps()

        # internal pointer for evaluation of g
        self._g_rhs = self._upper_g.copy()
        self._g_rhs[~self._c_mask] = 0.0

        # set number of equatity and inequality constraints from maps
        self._nc = len(self._c_map)
        self._nd = len(self._d_map)

        # populate jacobian structure
        self._irows_jac_g = np.zeros(self.nnz_jacobian_g, dtype=np.intc)
        self._jcols_jac_g = np.zeros(self.nnz_jacobian_g, dtype=np.intc)
        self._asl.struct_jac_g(self._irows_jac_g, self._jcols_jac_g)

        # this is to index from zero the triplets
        self._irows_jac_g -= 1
        self._jcols_jac_g -= 1

        self._irows_c_mask = np.in1d(self._irows_jac_g, self._c_map)
        self._irows_d_mask = np.logical_not(self._irows_c_mask)
        self._irows_jac_c = np.compress(self._irows_c_mask, self._irows_jac_g)
        self._jcols_jac_c = np.compress(self._irows_c_mask, self._jcols_jac_g)
        self._irows_jac_d = np.compress(self._irows_d_mask, self._irows_jac_g)
        self._jcols_jac_d = np.compress(self._irows_d_mask, self._jcols_jac_g)

        # this is expensive but only done once TODO: to vectorize later
        mapa = {self._c_map[i]: i for i in range(self.nc)}
        for i, v in enumerate(self._irows_jac_c):
            self._irows_jac_c[i] = mapa[v]

        mapa = {self._d_map[i]: i for i in range(self.nd)}
        for i, v in enumerate(self._irows_jac_d):
            self._irows_jac_d[i] = mapa[v]

        # set nnz jacobian c and d
        self._nnz_jac_c = len(self._jcols_jac_c)
        self._nnz_jac_d = len(self._jcols_jac_d)

        # make pointers unmutable from the outside world
        self._irows_jac_c.flags.writeable = False
        self._jcols_jac_c.flags.writeable = False
        self._irows_jac_d.flags.writeable = False
        self._jcols_jac_d.flags.writeable = False
        self._irows_jac_g.flags.writeable = False
        self._jcols_jac_g.flags.writeable = False

        # populate hessian structure
        self._irows_hess = np.zeros(self.nnz_hessian_lag, dtype=np.intc)
        self._jcols_hess = np.zeros(self.nnz_hessian_lag, dtype=np.intc)
        self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
        self._irows_hess -= 1
        self._jcols_hess -= 1
        self._irows_hess.flags.writeable = False
        self._jcols_hess.flags.writeable = False

        # make lower, upper init arrays inmutable
        self._lower_x.flags.writeable = False
        self._upper_x.flags.writeable = False
        self._lower_g.flags.writeable = False
        self._upper_g.flags.writeable = False
        self._init_x.flags.writeable = False
        self._init_y.flags.writeable = False


    @property
    def nx(self):
        """
        Return the number of variables.
        """
        return self._nx

    @property
    def ng(self):
        """
        Return the total number of constraints.
        """
        return self._ng

    @property
    def nc(self):
        """
        Return the number of equality constraints.
        """
        return self._nc

    @property
    def nd(self):
        """
        Return the number of inequality constraints.
        """
        return self._nd

    @property
    def nnz_jacobian_g(self):
        """
        Return the number of nonzero values in the
        jacobian of all constraints.
        """
        return self._nnz_jac_g

    @property
    def nnz_jacobian_c(self):
        """
        Return the number of nonzero values in the
        jacobian of equality constraints (subset of g).
        """
        return self._nnz_jac_c

    @property
    def nnz_jacobian_d(self):
        """
        Return the number of nonzero values in the
        jacobian of inequality constraints (subset of g).
        """
        return self._nnz_jac_d

    @property
    def nnz_hessian_lag(self):
        """
        Return the number of nonzero values in the
        hessian of the Lagrangian function.
        """
        return self._nnz_hess_lag

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

    @nx.setter
    def nx(self, value):
        """
        Prevent changing number of primal variables.
        """
        raise RuntimeError('Cannot change number of variables')

    @nc.setter
    def nc(self, value):
        """
        Prevent changing number of primal variables.
        """
        raise RuntimeError('Cannot change number of constraints')

    @nnz_jacobian_g.setter
    def nnz_jacobian_g(self, value):
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_jacobian_c.setter
    def nnz_jacobian_c(self, value):
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_jacobian_d.setter
    def nnz_jacobian_d(self, value):
        raise RuntimeError('Cannot change number of nonzeros in jacobian')

    @nnz_hessian_lag.setter
    def nnz_hessian_lag(self, value):
        raise RuntimeError('Cannot change number of nonzeros in hessian')

    @xl.setter
    def xl(self, other):
        """
        Prevent changing lower bounds of primal variables
        """
        raise RuntimeError('Changing bounds not supported for now')
        #if len(other) != self.nx:
        #    raise RuntimeError('Dimension of vector does not match')
        #self._lower_x = np.copy(other)

    @xu.setter
    def xu(self, other):
        """
        Prevent changing upper bounds of primal variables
        """
        raise RuntimeError('Changing bounds not supported for now')
        #if len(other) != self.nx:
        #    raise RuntimeError('Dimension of vector does not match')
        #self._upper_x = np.copy(other)

    @gl.setter
    def gl(self):
        """
        Prevent changing lower bounds of constraints
        """
        raise NotImplementedError('Abstract class method')

    @gu.setter
    def gu(self):
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

    def _build_x_maps(self):

        # sanity check for unsupported bounds on x
        tolerance_fixed_bounds = 1e-6  # TODO: define this in another place
        bounds_difference = self._upper_x - self._lower_x
        abs_bounds_difference = np.absolute(bounds_difference)
        fixed_vars = np.any(abs_bounds_difference < tolerance_fixed_bounds)
        if fixed_vars:
            raise RuntimeError("Variables fixed from bounds not supported")
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            # TODO: improve error message
            raise RuntimeError("Inconsistent bounds on x")

        # build x lower and upper bound maps
        self._lower_x_mask = np.isfinite(self._lower_x)
        self._lower_x_map = self._lower_x_mask.nonzero()[0]
        self._upper_x_mask = np.isfinite(self._upper_x)
        self._upper_x_map = self._upper_x_mask.nonzero()[0]

    def _build_g_maps(self):

        # sanity check for unsupported bounds on g
        bounds_difference = self._upper_g - self._lower_g
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            # TODO: improve error message # TODO: define this in another place
            raise RuntimeError("Inconsistent bounds on g")

        # build x lower and upper bound maps
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._c_mask = abs_bounds_difference < tolerance_equalities
        self._c_map = self._c_mask.nonzero()[0]
        self._d_mask = abs_bounds_difference >= tolerance_equalities
        self._d_map = self._d_mask.nonzero()[0]

        self._lower_d_mask = np.isfinite(self._lower_g) * self._d_mask
        self._lower_d_map = self._lower_d_mask.nonzero()[0]
        self._upper_d_mask = np.isfinite(self._upper_g) * self._d_mask
        self._upper_d_map = self._upper_d_mask.nonzero()[0]

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

    def objective(self, x):
        """Return value of objective function evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx

        Returns
        -------
        The value of the objective function

        """
        return self._asl.eval_f(x)

    def grad_objective(self, x, out=None):
        """Return gradient of the objective function evaluated at x

        Parameters
        ----------
        x : 1d-array
            array with values of primal variables. Size nx
        other : 1d-array
            array to store the gradient of the objective function (optional)

        Returns
        -------
        The gradient of the objective function in a 1d-array

        """
        if out is None:
            df = self.create_vector_x()
        else:
            msg = "grad_objective takes a ndarray of size {}".format(self.nx)
            assert isinstance(out, np.ndarray) and out.size == self.nx, msg
            df = out

        self._asl.eval_deriv_f(x, df)
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
        if out is None:
            res = self.create_vector_y()
        else:
            msg = "evaluate_g takes a ndarray of size {}".format(self.ng)
            assert isinstance(out, np.ndarray) and out.size == self.ng, msg
            res = out

        self._asl.eval_g(x, res)
        #res -= self._g_rhs
        np.subtract(res, self._g_rhs, res)
        return res

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
        evaluated_g = kwargs.pop('evaluated_g', None)

        if evaluated_g is None:
            res = self.create_vector_y()
            self._asl.eval_g(x, res)
            eval_g = res - self._g_rhs
        else:
            msg = "evaluate_c takes a ndarray of size {} for evaluated_g".format(self.ng)
            assert isinstance(evaluated_g, np.ndarray) and evaluated_g.size == self.ng, msg
            eval_g = evaluated_g

        if out is not None:
            msg = "evaluate_c takes a ndarray of size {}".format(self.nc)
            assert isinstance(out, np.ndarray) and out.size == self.nc, msg

        return eval_g.compress(self._c_mask, out=out)

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
        evaluated_g = kwargs.pop('evaluated_g', None)

        if evaluated_g is None:
            res = self.create_vector_y()
            self._asl.eval_g(x, res)
            eval_g = res - self._g_rhs
        else:
            msg = "evaluate_d takes a ndarray of size {} for evaluated_g".format(self.ng)
            assert isinstance(evaluated_g, np.ndarray) and evaluated_g.size == self.ng, msg
            eval_g = evaluated_g

        if out is not None:
            msg = "evaluate_d takes a ndarray of size {}".format(self.nd)
            assert isinstance(out, np.ndarray) and out.size == self.nd, msg

        return eval_g.compress(self._d_mask, out=out)

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
        if out is None:
            data = np.zeros(self.nnz_jacobian_g, np.double)
            self._asl.eval_jac_g(x, data)
            jac = COOMatrix((data, (self._irows_jac_g, self._jcols_jac_g)),
                             shape=(self.ng, self.nx))
        else:
            assert isinstance(out, COOMatrix), "jacobian_g must be a COOMatrix"
            assert out.shape[0] == self.ng, "jacobian_g has {} rows".format(self.ng)
            assert out.shape[1] == self.nx, "jacobian_g has {} columns".format(self.nx)
            assert out.nnz == self.nnz_jacobian_g, "jacobian_g has {} nnz".format(self.nnz_jacobian_g)

            data = out.data
            self._asl.eval_jac_g(x, data)
            jac = out

        return jac

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
        evaluated_jac_g = kwargs.pop('evaluated_jac_g', None)

        if evaluated_jac_g is None:
            data = np.zeros(self.nnz_jacobian_g, np.double)
            self._asl.eval_jac_g(x, data)

            if out is not None:
                assert isinstance(out, COOMatrix), "jacobian_c must be a COOMatrix"
                assert out.shape[0] == self.nc, "jacobian_c has {} rows".format(self.nc)
                assert out.shape[1] == self.nx, "jacobian_c has {} columns".format(self.nx)
                data.compress(self._irows_c_mask, out=out.data)
                jac = out
            else:
                c_data = data.compress(self._irows_c_mask)
                jac = COOMatrix((c_data, (self._irows_jac_c, self._jcols_jac_c)),
                                shape=(self.nc, self.nx))
        else:
            assert isinstance(evaluated_jac_g, COOMatrix), "jacobian_g must be a COOMatrix"
            assert evaluated_jac_g.shape[0] == self.ng, "jacobian_g has {} rows".format(self.ng)
            assert evaluated_jac_g.shape[1] == self.nx, "jacobian_g has {} columns".format(self.nx)
            assert evaluated_jac_g.nnz == self.nnz_jacobian_g, "jacobian_g has {} nnz".format(self.nnz_jacobian_g)
            c_data = evaluated_jac_g.data.compress(self._irows_c_mask)
            jac = COOMatrix((c_data, (self._irows_jac_c, self._jcols_jac_c)),
                            shape=(self.nc, self.nx))

            if out is not None:
                assert isinstance(out, COOMatrix), "jacobian_c must be a COOMatrix"
                assert out.shape[0] == self.nc, "jacobian_c has {} rows".format(self.nc)
                assert out.shape[1] == self.nx, "jacobian_c has {} columns".format(self.nx)
                evaluated_jac_g.data.compress(self._irows_c_mask, out=out.data)
                jac = out
            else:
                c_data = evaluated_jac_g.data.compress(self._irows_c_mask)
                jac = COOMatrix((c_data, (self._irows_jac_c, self._jcols_jac_c)),
                                shape=(self.nc, self.nx))
        return jac

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

        evaluated_jac_g = kwargs.pop('evaluated_jac_g', None)

        if evaluated_jac_g is None:
            data = np.zeros(self.nnz_jacobian_g, np.double)
            self._asl.eval_jac_g(x, data)

            if out is not None:
                assert isinstance(out, COOMatrix), "jacobian_d must be a COOMatrix"
                assert out.shape[0] == self.nd, "jacobian_d has {} rows".format(self.nd)
                assert out.shape[1] == self.nx, "jacobian_d has {} columns".format(self.nx)
                data.compress(self._irows_d_mask, out=out.data)
                jac = out
            else:
                d_data = data.compress(self._irows_d_mask)
                jac = COOMatrix((d_data, (self._irows_jac_d, self._jcols_jac_d)),
                                shape=(self.nd, self.nx))
        else:

            assert isinstance(evaluated_jac_g, COOMatrix), "jacobian_g must be a COOMatrix"
            assert evaluated_jac_g.shape[0] == self.ng, "jacobian_g has {} rows".format(self.ng)
            assert evaluated_jac_g.shape[1] == self.nx, "jacobian_g has {} columns".format(self.nx)
            assert evaluated_jac_g.nnz == self.nnz_jacobian_g, "jacobian_g has {} nnz".format(self.nnz_jacobian_g)

            if out is not None:
                assert isinstance(out, COOMatrix), "jacobian_d must be a COOMatrix"
                assert out.shape[0] == self.nd, "jacobian_d has {} rows".format(self.nd)
                assert out.shape[1] == self.nx, "jacobian_d has {} columns".format(self.nx)
                evaluated_jac_g.data.compress(self._irows_d_mask, out=out.data)
                jac = out
            else:
                d_data = evaluated_jac_g.data.compress(self._irows_d_mask)
                jac = COOMatrix((d_data, (self._irows_jac_d, self._jcols_jac_d)),
                                shape=(self.nd, self.nx))
        return jac

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

        eval_f_c = kwargs.pop('eval_f_c', True)

        if eval_f_c:
            res = self.create_vector_y()
            self._asl.eval_g(x, res)
            self._asl.eval_f(x)
        if out is None:

            data = np.zeros(self.nnz_hessian_lag, np.double)
            self._asl.eval_hes_lag(x, y, data)

            hess = COOSymMatrix((data, (self._irows_hess, self._jcols_hess)),
                                shape=(self.nx, self.nx))
        else:
            assert isinstance(out, COOSymMatrix), "hessian must be a COOSymMatrix"
            assert out.shape[0] == self.nx, "hessian has {} rows".format(self.nx)
            assert out.shape[1] == self.nx, "hessian has {} columns".format(self.nx)
            assert out.nnz == self.nnz_hessian_lag, "hessian has {} nnz".format(self.nnz_hessian_lag)

            data = out.data
            self._asl.eval_hes_lag(x, y, data)
            hess = out

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
        self._asl.finalize_solution(status, x, y)

    def _build_component_names_list(self, filename):

        ordered_names = list()
        with open(filename, 'r') as f:
            for line in f:
                ordered_names.append(line.strip('\n'))
        return ordered_names


class AmplNLP(StubNLP):
    """
    Nonlinear programm interface that relies on the Ampl solver library

    Parameters
    -------------------
    model: string
    filename of the NL-file containing the model
    row_filename: string (optional)
    filename of .row file with identity of constraints (optional)
    col_filename: string (optional)
    filename of .col file with identity of variables (optional)
    """

    # ToDo: add access to d_l d_u
    # ToDo: change on x_l and x_u, g_u, g_l?

    def __init__(self, model, row_filename=None, col_filename=None):

        # call parent class to set model
        super(AmplNLP, self).__init__(model)

        self._rowfile = row_filename
        self._colfile = col_filename

        # create containers with names of variables
        self._vid_to_name = None
        self._name_to_vid = None
        if col_filename is not None:
            self._vid_to_name = self._build_component_names_list(col_filename)
            self._name_to_vid = {self._vid_to_name[vid]: vid for vid in range(self.nx)}

        # create containers with names of constraints and objective
        self._cid_to_name = None
        self._name_to_cid = None
        self._obj_name = None
        if row_filename is not None:
            all_names = self._build_component_names_list(row_filename)
            self._obj_name = all_names[-1]
            del all_names[-1]
            self._cid_to_name = all_names
            self._name_to_cid = {all_names[cid]: cid for cid in range(self.ng)}

    # Query methods
    def Grad_objective(self, x, var_names=None, var_indices=None):

        if var_indices is not None and var_names is not None:
            raise RuntimeError('pick indices or names but not both')

        df = self.create_vector_x()


        self._asl.eval_deriv_f(x, df)

        if var_indices is not None:
            return df[var_indices]

        if var_names is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify variable names')
            indices = [self._name_to_vid[n] for n in var_names]
            return df[indices]
        return df

    def Evaluate_g(self, x, constraint_indices=None, constraint_names=None):

        if constraint_indices is not None and constraint_names is not None:
            raise RuntimeError('pick indices or names but not both')

        res = self.evaluate_g(x)

        if constraint_indices is not None:
            return res[constraint_indices]

        if constraint_names is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify constraint names')
            indices = [self._name_to_cid[n] for n in constraint_names]
            return res[indices]
        return res

    def Jacobian_g(self, x, matrix_format='coo', var_names=None, var_indices=None,
                 constraint_indices=None, constraint_names=None):

        if var_indices is not None and var_names is not None:
            raise RuntimeError('pick indices or names but not both')

        if constraint_indices is not None and constraint_names is not None:
            raise RuntimeError('pick indices or names but not both')

        subset_vars = False
        if var_indices is not None:
            indices_vars = var_indices
            subset_vars = True
        if var_names is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify variable names')
            indices_vars = [self._name_to_vid[n] for n in var_names]
            subset_vars = True

        subset_constraints = False
        if constraint_indices is not None:
            indices_constraints = constraint_indices
            subset_constraints = True
        if constraint_names is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify constraint names')
            indices_constraints = [self._name_to_cid[n] for n in constraint_names]
            subset_constraints = True

        if subset_vars:
            jcols_bool = np.isin(self._jcols_jac_g, indices_vars)
            ncols = len(indices_vars)
        else:
            jcols_bool = np.ones(self.nnz_jacobian_g, dtype=bool)
            ncols = self.nx

        if subset_constraints:
            irows_bool = np.isin(self._irows_jac_g, indices_constraints)
            nrows = len(indices_constraints)
        else:
            irows_bool = np.ones(self.nnz_jacobian_g, dtype=bool)
            nrows = self.ng

        vals_bool = irows_bool * jcols_bool
        vals_indices = np.where(vals_bool)

        vals_jac = np.zeros(self.nnz_jacobian_g, np.double)
        self._asl.eval_jac_g(x, vals_jac)
        data = vals_jac[vals_indices]

        # map indices to new indices
        new_col_indices = self._jcols_jac_g[vals_indices]
        if subset_vars:
            old_col_indices = self._jcols_jac_g[vals_indices]
            vid_to_nvid = {vid: idx for idx, vid in enumerate(indices_vars)}
            new_col_indices = np.array([vid_to_nvid[vid] for vid in old_col_indices])

        new_row_indices = self._irows_jac_g[vals_indices]
        if subset_constraints:
            old_const_indices = self._irows_jac_g[vals_indices]
            cid_to_ncid = {cid: idx for idx, cid in enumerate(indices_constraints)}
            new_row_indices = np.array([cid_to_ncid[cid] for cid in old_const_indices])

        if matrix_format == 'coo':
            return COOMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'csr':
            return CSRMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'csc':
            return CSCMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'dok':
            dok = dok_matrix((nrows, ncols), dtype=np.float64)
            for i, v in enumerate(data):
                dok[new_row_indices[i], new_col_indices[i]] = v
            return dok

        raise RuntimeError('Matrix format not recognized')

    def Hessian_lag(self,
                    x,
                    lam,
                    matrix_format='coo',
                    var_names_rows=None,
                    var_indices_rows=None,
                    var_names_cols=None,
                    var_indices_cols=None,
                    eval_f_c=True):

        """
        Note: this follows the same order given in the subset of vars. It does not retain
        the symmetric property.
        Parameters
        ----------
        x
        lam
        matrix_format
        var_names_rows
        var_indices_rows
        var_names_cols
        var_indices_cols
        eval_f_c

        Returns
        -------

        """

        if var_indices_cols is not None and var_names_cols is not None:
            raise RuntimeError('pick indices or names but not both')

        if var_indices_rows is not None and var_names_rows is not None:
            raise RuntimeError('pick indices or names but not both')

        subset_cols = False
        if var_indices_cols is not None:
            indices_cols = var_indices_cols
            subset_cols = True
        if var_names_cols is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify variable names')
            indices_cols = [self._name_to_vid[n] for n in var_names_cols]
            subset_cols = True

        subset_rows = False
        if var_indices_rows is not None:
            indices_rows = var_indices_rows
            subset_rows = True
        if var_names_rows is not None:
            if self._vid_to_name is None:
                raise RuntimeError('specify variable names')
            indices_rows = [self._name_to_vid[n] for n in var_names_rows]
            subset_rows = True

        if eval_f_c:
            res = self.create_vector_y()
            self._asl.eval_g(x, res)
            self._asl.eval_f(x)

        if subset_cols:
            jcols_bool = np.isin(self._jcols_hess, indices_cols)
            ncols = len(indices_cols)
        else:
            jcols_bool = np.ones(self.nnz_hessian_lag, dtype=bool)
            ncols = self.nx

        if subset_rows:
            irows_bool = np.isin(self._irows_hess, indices_rows)
            nrows = len(indices_rows)
        else:
            irows_bool = np.ones(self.nnz_hessian_lag, dtype=bool)
            nrows = self.nx

        vals_bool = irows_bool * jcols_bool
        vals_indices = np.where(vals_bool)

        vals_hess = np.zeros(self.nnz_hessian_lag, 'd')
        self._asl.eval_hes_lag(x, lam, vals_hess)
        data = vals_hess[vals_indices]

        # map indices to new indices
        new_col_indices = self._jcols_hess[vals_indices]
        if subset_cols:
            old_col_indices = self._jcols_hess[vals_indices]
            vid_to_nvid = {vid: idx for idx, vid in enumerate(indices_cols)}
            new_col_indices = np.array([vid_to_nvid[vid] for vid in old_col_indices])

        new_row_indices = self._irows_hess[vals_indices]
        if subset_rows:
            old_row_indices = self._irows_hess[vals_indices]
            cid_to_ncid = {cid: idx for idx, cid in enumerate(indices_rows)}
            new_row_indices = np.array([cid_to_ncid[cid] for cid in old_row_indices])

        if matrix_format == 'coo':
            return COOMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'csr':
            return CSRMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'csc':
            return CSCMatrix((data, (new_row_indices, new_col_indices)),
                             shape=(nrows, ncols))
        if matrix_format == 'dok':
            dok = dok_matrix((nrows, ncols), dtype=np.float64)
            for i, v in enumerate(data):
                dok[new_row_indices[i], new_col_indices[i]] = v
            return dok

        raise RuntimeError('Matrix format not recognized')


class PyomoNLP(AmplNLP):

    def __init__(self, model):

        # TODO: create maps id to name component directly with pyomo symbolic_labels
        temporal_dir = tempfile.mkdtemp()
        try:
            filename = os.path.join(temporal_dir, "pynumero_pyomo")
            model.write(filename+'.nl', 'nl', io_options={"symbolic_solver_labels": True})
            import pyomo
            fname, symbolMap = pyomo.opt.WriterFactory('nl')(model, filename, lambda x:True, {})
            varToIndex = pyomo.core.kernel.ComponentMap()
            conToIndex = pyomo.core.kernel.ComponentMap()
            for name, obj in six.iteritems(symbolMap.bySymbol):
                if name[0] == 'v':
                    varToIndex[obj()] = int(name[1:])
                elif name[1] == 'c':
                    conToIndex[obj()] = int(name[1:])
            nl_file = filename+".nl"
            row_file = filename+".row"
            col_file = filename + ".col"
            super(PyomoNLP, self).__init__(nl_file, row_filename=row_file, col_filename=col_file)
            self._model = model
        finally:
            shutil.rmtree(temporal_dir)


