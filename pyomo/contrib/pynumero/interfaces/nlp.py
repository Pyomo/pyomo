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
import pyomo
import pyomo.environ as aml

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
    Base class for nonlinear programs.

    Note
    ----
    Any subclass of NLP must overwrite _build_x_maps and _build_gcd_maps.
    Any subclass must call super at the beginning of the __init__() method

    Attributes
    ----------
    _model : optimization model
        object containing an optimization model. Type of model depends
        on subclass. PyomoNLP takes a pyomo ConcreteModel

    _nx : int
        Number of primal variables

    _ng : int
        Number of general inequalities g_l <= g(x) <= g_u

    _nc : int
        Number of equalities c(x) = 0

    _ng : int
        Number of inequalities d_l <= d(x) <= d_u

    _nnz_jac_g : int
        Number of nonzeros in jacobian of general inequalities g(x)

    _nnz_jac_c : int
        Number of nonzeros in jacobian of equalities c(x)

    _nnz_jac_d : int
        Number of nonzeros in jacobian of inequalities d(x)

    _nnz_hess_lag : int
        Number of nonzeros in jacobian of inequalities d(x)

    _init_x : array_like
        1D array containing primal variables initial guesses

    _init_y : array_like
        1D array containing dual variable initial guesses

    _lower_x : array_like
        1D array containing primal variables lower bounds.
        Variables without lower bound have -numpy.inf entries

    _upper_x : array_like
        1D array containing primal variables upper bounds.
        Variables without upper bound have numpy.inf entries

    _lower_g : array_like
        1D array containing general inequalities lower bounds.
        General inequalities without lower bound have -numpy.inf entries

    _upper_g : array_like
        1D array containing general inequalities upper bounds.
        General inequalities without upper bound have numpy.inf entries

    _lower_d : array_like
        1D array containing inequalities lower bounds.
        Inequalities without lower bound have -numpy.inf entries

    _upper_d : array_like
        1D array containing inequalities upper bounds.
        Inequalities without upper bound have numpy.inf entries

    _lower_x_mask : array_like
        1D array containing booleans that indicate if primal variable
        has lower bound or not.

    _upper_x_mask : array_like
        1D array containing booleans that indicate if primal variable
        has upper bound or not.

    _lower_g_mask : array_like
        1D array containing booleans that indicate if general inequality
        has lower bound or not.

    _upper_g_mask : array_like
        1D array containing booleans that indicate if general inequality
        has upper bound or not.

    _lower_d_mask : array_like
        1D array containing booleans that indicate if inequality
        has lower bound or not.

    _upper_d_mask : array_like
        1D array containing booleans that indicate if inequality
        has upper bound or not.

    _c_mask : array_like
        1D array containing booleans that indicate if general inequality g
        is an equality c(x)

    _lower_x_map : array_like
        1D array containing indices of _lower_x that have lower bound
        different than -numpy.inf

    _upper_x_map : array_like
        1D array containing indices of _upper_x that have upper bound
        different than numpy.inf

    _lower_g_map : array_like
        1D array containing indices of _lower_g that have lower bound
        different than -numpy.inf

    _upper_g_map : array_like
        1D array containing indices of _upper_g that have upper bound
        different than numpy.inf

    _lower_d_map : array_like
        1D array containing indices of _lower_d that have lower bound
        different than -numpy.inf

    _upper_d_map : array_like
        1D array containing indices of _upper_d that have upper bound
        different than numpy.inf

    _c_map : array_like
        1D array containing indices of g(x) that are equalities c(x)

    _d_map : array_like
        1D array containing indices of g(x) that are inequalities d(x)
    """

    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model : optimization model
            Type of model depends on subclass. PyomoNLP takes a
            pyomo ConcreteModel
        kwargs
            Arbitrary keyword arguments
        """

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
        self._upper_d = None
        self._lower_d = None

        # masks
        self._lower_x_mask = None
        self._upper_x_mask = None
        self._lower_g_mask = None
        self._upper_g_mask = None
        self._lower_d_mask = None
        self._upper_d_mask = None
        self._c_mask = None
        self._d_mask = None  # ToDo: think if this is needed or simply use negate _c_mask

        # maps
        self._lower_x_map = None
        self._upper_x_map = None
        self._lower_g_map = None
        self._upper_g_map = None
        self._lower_d_map = None
        self._upper_d_map = None
        self._c_map = None
        self._d_map = None

    @abc.abstractmethod
    def _initialize_nlp_components(self, *args, **kwargs):
        """
        Initializes all attributes of the nlp

        Note
        ----
        This method must be called always in the subclasses constructor
        The attributes that must be initialized are:
            _nx
            _init_x
            _lower_x
            _upper_x
            _lower_x_mask
            _upper_x_mask
            _lower_x_map
            _upper_x_map

            _ng
            _nc
            _nd
            _nnz_jac_g
            _nnz_jac_c
            _nnz_jac_d
            _init_y
            _lower_g
            _upper_g
            _lower_g_mask
            _upper_g_mask
            _lower_g_map
            _upper_g_map

            _d_mask
            _d_map
            _lower_d
            _upper_d
            _lower_d_mask
            _upper_d_mask
            _lower_d_map
            _upper_d_map

            _c_mask
            _c_map
        Subclass attributes may be intialized here as well
        """
        return

    def _make_unmutable_caches(self):
        """
        Sets writable flag of internal arrays (cached) to false
        """
        # make lower, upper init arrays inmutable
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

    @property
    def nx(self):
        """
        Returns number of primal variables
        """
        return self._nx

    @property
    def ng(self):
        """
        Returns number of general inequality constraints g(x)
        """
        return self._ng

    @property
    def nc(self):
        """
        Returns number of equality constraints c(x)
        """
        return self._nc

    @property
    def nd(self):
        """
        Returns number of inequality constraints d(x)
        """
        return self._nd

    @property
    def nnz_jacobian_g(self):
        """
        Returns number of nonzero values in jacobian of general inequalities g(x)
        """
        return self._nnz_jac_g

    @property
    def nnz_jacobian_c(self):
        """
        Returns number of nonzero values in jacobian of equalities c(x)
        """
        return self._nnz_jac_c

    @property
    def nnz_jacobian_d(self):
        """
        Returns number of nonzero values in jacobian of inequalities d(x)
        """
        return self._nnz_jac_d

    @property
    def nnz_hessian_lag(self):
        """
        Returns number of nonzero values in hessian of the lagrangian function
        """
        return self._nnz_hess_lag

    def xl(self, condensed=False):
        """
        Returns array of primal variable lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._lower_x_mask, self._lower_x)
        return self._lower_x

    def xu(self, condensed=False):
        """
        Returns array of primal variable upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._upper_x_mask, self._upper_x)
        return self._upper_x

    def gl(self, condensed=False):
        """
        Returns array of general inequalities lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._lower_g_mask, self._lower_g)
        return self._lower_g

    def gu(self, condensed=False):
        """
        Returns array of general inequalities upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._upper_g_mask, self._upper_g)
        return self._upper_g

    def dl(self, condensed=False):
        """
        Returns array of inequalities lower bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes -numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._lower_d_mask, self._lower_d)
        return self._lower_d

    def du(self, condensed=False):
        """
        Returns array of inequalities upper bounds

        Parameters
        ----------
        condensed :  bool, optional
            Boolean flag to indicate if array excludes numpy.inf values (default False)

        Returns
        -------
        ndarray

        """
        if condensed:
            return np.compress(self._upper_d_mask, self._upper_d)
        return self._upper_d

    def x_init(self):
        """
        Returns initial guess of primal variables in a 1d-array.
        """
        return self._init_x.copy()

    def y_init(self):
        """
        Returns initial guess of dual variables in a 1d-array.
        """
        return self._init_y.copy()

    def expansion_matrix_xl(self):
        """
        Returns expansion matrix for lower bounds on primal variables
        """
        row = self._lower_x_map
        nnz = len(self._lower_x_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.nx, nnz))

    def expansion_matrix_xu(self):
        """
        Returns expansion matrix for upper bounds on primal variables
        """
        row = self._upper_x_map
        nnz = len(self._upper_x_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.nx, nnz))

    def expansion_matrix_dl(self):
        """
        Returns expansion matrix lower bounds on inequality constraints
        """

        row = self._lower_d_map
        nnz = len(self._lower_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.nd, nnz))

    def expansion_matrix_du(self):
        """
        Returns expansion matrix upper bounds on inequality constraints
        """
        row = self._upper_d_map
        nnz = len(self._upper_d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.nd, nnz))

    def expansion_matrix_d(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = self._d_map
        nnz = len(self._d_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.ng, nnz))

    def expansion_matrix_c(self):
        """
        Returns expansion matrix inequality constraints
        """
        row = self._c_map
        nnz = len(self._c_map)
        col = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return CSRMatrix((data, (row, col)), shape=(self.ng, nnz))

    def create_vector_x(self, subset=None):
        """Returns ndarray of primal variables

        Parameters
        ----------
        subset : str, optional
            determines size of vector.
            `l`: only primal variables with lower bounds
            `u`: only primal variables with upper bounds

        Returns
        -------
        ndarray

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

    def create_vector_s(self, subset=None):
        """Returns ndarray of slack variables

        Parameters
        ----------
        subset : str, optional
            determines size of vector.
            `l`: only slack variables with lower bounds
            `u`: only slack variables with upper bounds

        Returns
        -------
        ndarray

        """
        if subset is None:
            return self.create_vector_y(subset='d')
        elif subset == 'l':
            return self.create_vector_y(subset='dl')
        elif subset == 'u':
            return self.create_vector_y(subset='du')
        else:
            raise RuntimeError('Subset not recognized')

    def create_vector_y(self, subset=None):
        """Return ndarray of vector of constraints

        Parameters
        ----------
        subset : str, optional
            determines size of vector.
            `c`: only equality constraints
            `d`: only inequality constraints
            `dl`: only inequality constraints with lower bound
            `du`: only inequality constraints with upper bound

        Returns
        -------
        ndarray

        """
        if subset is None:
            return np.zeros(self.ng, dtype=np.double)
        elif subset == 'c':
            return np.zeros(self.nc, dtype=np.double)
        elif subset == 'd':
            return np.zeros(self.nd, dtype=np.double)
        elif subset == 'dl':
            ndl = len(self._lower_d_map)
            return np.zeros(ndl, dtype=np.double)
        elif subset == 'du':
            ndu = len(self._upper_d_map)
            return np.zeros(ndu, dtype=np.double)
        else:
            raise RuntimeError('Subset not recognized')

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
    def evaluate_g(self, x, out=None, **kwargs):
        """Returns general inequality constraints evaluated at x

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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

        """
        return

    @abc.abstractmethod
    def jacobian_c(self, x, out=None, **kwargs):
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

        """
        return

    @abc.abstractmethod
    def jacobian_d(self, x, out=None, **kwargs):
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

        """
        return

    @abc.abstractmethod
    def hessian_lag(self, x, y, out=None, **kwargs):
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : SymCOOMatrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        SYMCOOMatrix

        """
        return


class AslNLP(NLP):
    """
    ASL nonlinear program interface

    Attributes
    ----------
    _asl : AmplInterface
        ASL interface that links with c-functions
    _irows_jac_g: ndarray
        Array with row indices of nonzero elements in Jacobian of g(x)
    _irows_jac_c: ndarray
        Array with row indices of nonzero elements in Jacobian of c(x)
    _irows_jac_d: ndarray
        Array with row indices of nonzero elements in Jacobian of d(x)
    _jcols_jac_g: ndarray
        Array with column indices of nonzero elements in Jacobian of g(x)
    _jcols_jac_c: ndarray
        Array with column indices of nonzero elements in Jacobian of c(x)
    _jcols_jac_d: ndarray
        Array with column indices of nonzero elements in Jacobian of d(x)
    """

    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model : string
            filename of the NL-file containing the model
        kwargs
            Arbitrary keyword arguments
        """

        # call parent class to set model
        super(AslNLP, self).__init__(model)

        # ampl interface
        self._asl = _asl.AmplInterface(self._model)

        # initialize components
        self._initialize_nlp_components()

        # make pointer unmutable from outside world
        self._make_unmutable_caches()

    def _initialize_nlp_components(self, *args, **kwargs):

        # dimensions
        self._nx = self._asl.get_n_vars()
        self._ng = self._asl.get_n_constraints()
        self._nnz_jac_g = self._asl.get_nnz_jac_g()
        self._nnz_hess_lag = self._asl.get_nnz_hessian_lag()

        # initial point
        self._init_x = np.zeros(self.nx)
        self._init_y = np.zeros(self.ng)
        self._asl.get_init_x(self._init_x)
        self._asl.get_init_multipliers(self._init_y)

        # bounds on x
        self._upper_x = np.zeros(self.nx, dtype=np.double)
        self._lower_x = np.zeros(self.nx, dtype=np.double)
        self._asl.get_x_lower_bounds(self._lower_x)
        self._asl.get_x_upper_bounds(self._upper_x)

        # bounds on g
        self._upper_g = np.zeros(self.ng, dtype=np.double)
        self._lower_g = np.zeros(self.ng, dtype=np.double)
        self._asl.get_g_lower_bounds(self._lower_g)
        self._asl.get_g_upper_bounds(self._upper_g)

        # build maps for compressing vectors (only lower, upper bounds, and equalities)
        self._build_x_maps()
        self._build_gcd_maps()

        # define bounds on d extracted from g
        self._lower_d = np.compress(self._d_mask, self._lower_g)
        self._upper_d = np.compress(self._d_mask, self._upper_g)

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
        self._irows_jac_g -= 1
        self._jcols_jac_g -= 1

        self._irows_c_mask = np.in1d(self._irows_jac_g, self._c_map)
        self._irows_d_mask = np.logical_not(self._irows_c_mask)
        self._irows_jac_c = np.compress(self._irows_c_mask, self._irows_jac_g)
        self._jcols_jac_c = np.compress(self._irows_c_mask, self._jcols_jac_g)
        self._irows_jac_d = np.compress(self._irows_d_mask, self._irows_jac_g)
        self._jcols_jac_d = np.compress(self._irows_d_mask, self._jcols_jac_g)

        # this is expensive but only done once.
        # Could be vectorized or done from the c-side
        mapa = {self._c_map[i]: i for i in range(self.nc)}
        for i, v in enumerate(self._irows_jac_c):
            self._irows_jac_c[i] = mapa[v]

        mapa = {self._d_map[i]: i for i in range(self.nd)}
        for i, v in enumerate(self._irows_jac_d):
            self._irows_jac_d[i] = mapa[v]

        # set nnz jacobian c and d
        self._nnz_jac_c = len(self._jcols_jac_c)
        self._nnz_jac_d = len(self._jcols_jac_d)

        # populate hessian structure
        self._irows_hess = np.zeros(self.nnz_hessian_lag, dtype=np.intc)
        self._jcols_hess = np.zeros(self.nnz_hessian_lag, dtype=np.intc)
        self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
        self._irows_hess -= 1
        self._jcols_hess -= 1

    def _build_x_maps(self):

        # sanity check for unsupported bounds on x
        tolerance_fixed_bounds = 1e-6
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

    def _build_gcd_maps(self):

        # sanity check for unsupported bounds on g
        bounds_difference = self._upper_g - self._lower_g
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            # TODO: improve error message
            raise RuntimeError("Inconsistent bounds on g")

        # build x lower and upper bound maps
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._c_mask = abs_bounds_difference < tolerance_equalities
        self._c_map = self._c_mask.nonzero()[0]
        self._d_mask = abs_bounds_difference >= tolerance_equalities
        self._d_map = self._d_mask.nonzero()[0]

        self._lower_g_mask = np.isfinite(self._lower_g) * self._d_mask
        self._lower_g_map = self._lower_g_mask.nonzero()[0]
        self._upper_g_mask = np.isfinite(self._upper_g) * self._d_mask
        self._upper_g_map = self._upper_g_mask.nonzero()[0]

        self._lower_d_mask = np.isin(self._d_map, self._lower_g_map)
        self._lower_d_map = np.where(self._lower_d_mask)[0]
        self._upper_d_mask = np.isin(self._d_map, self._upper_g_map)
        self._upper_d_map = np.where(self._upper_d_mask)[0]

    def _make_unmutable_caches(self):
        """
        Sets writable flag of internal arrays (cached) to false
        """
        super(AslNLP, self)._make_unmutable_caches()

        self._irows_jac_c.flags.writeable = False
        self._jcols_jac_c.flags.writeable = False
        self._irows_jac_d.flags.writeable = False
        self._jcols_jac_d.flags.writeable = False
        self._irows_jac_g.flags.writeable = False
        self._jcols_jac_g.flags.writeable = False

        self._irows_hess.flags.writeable = False
        self._jcols_hess.flags.writeable = False

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
        return self._asl.eval_f(x)

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
        if out is None:
            df = self.create_vector_x()
        else:
            msg = "grad_objective takes a ndarray of size {}".format(self.nx)
            assert isinstance(out, np.ndarray) and out.size == self.nx, msg
            df = out

        self._asl.eval_deriv_f(x, df)
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

    def jacobian_g(self, x, out=None, **kwargs):
        """Returns the Jacobian of the general inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

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
        """Returns the Jacobian of the equalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

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
        """Returns the Jacobian of the inequalities evaluated at x

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        out : COOMatrix, optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        COOMatrix

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
        """Return the Hessian of the Lagrangian function evaluated at x and y

        Parameters
        ----------
        x : array_like
            Array with values of primal variables.
        y : array_like
            Array with values of dual variables.
        out : SymCOOMatrix
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        SYMCOOMatrix

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

    def finalize_solution(self, status_num, status_msg, x, y):
        """
        Write .sol file

        Parameters
        ----------
        status_num : int
            exit status (as integer code)
        status_msg : str
            exit status message
        x : ndarray
            1D array with values of primal variables. Size nx
        y : ndarray
            1D array with values of dual variables. Size ng

        Returns
        -------
        None
        """
        self._asl.finalize_solution(status_num, status_msg, x, y)


class AmplNLP(AslNLP):
    """
    AMPL nonlinear program interface


    Attributes
    ----------
    _rowfile: str
        Filename with names of constraints
    _colfile: str
        Filename with names of variables
    _vid_to_name: list
        Map from variable idx to variable name
    _name_to_vid: dict
        Map from variable name to variable idx
    _cid_to_name: list
        Map from constraint idx to constraint name
    _name_to_cid: dict
        Map from constraint name to constraint idx
    _obj_name: str
        Name of the objective function
    """

    def __init__(self, model, row_filename=None, col_filename=None):
        """

        Parameters
        ----------
        model : string
            filename of the NL-file containing the model
            filename of the NL-file containing the model
        row_filename: str, optional
            filename of .row file with identity of constraints
        col_filename: str, optional
            filename of .col file with identity of variables
        """

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

    def variable_order(self):
        return [name for name in self._vid_to_name]

    def constraint_order(self):
        return [name for name in self._cid_to_name]

    def variable_idx(self, var_name):
        return self._name_to_vid[var_name]

    def constraint_idx(self, con_name):
        return self._name_to_cid[con_name]

    @staticmethod
    def _build_component_names_list(filename):

        ordered_names = list()
        with open(filename, 'r') as f:
            for line in f:
                ordered_names.append(line.strip('\n'))
        return ordered_names


class PyomoNLP(AslNLP):
    """
    Pyomo nonlinear program interface


    Attributes
    ----------
    _varToIndex: pyomo.core.kernel.ComponentMap
        Map from variable name to variable idx
    _conToIndex: pyomo.core.kernel.ComponentMap
        Map from constraint name to constraint idx
    """

    def __init__(self, model):
        """

        Parameters
        ----------
        model : ConcreteModel
            Pyomo concrete model
        """
        temporal_dir = tempfile.mkdtemp()
        try:
            filename = os.path.join(temporal_dir, "pynumero_pyomo")
            objectives = model.component_map(aml.Objective, active=True)
            if len(objectives) == 0:
                model._dummy_obj = aml.Objective(expr=0.0)

            model.write(filename+'.nl', 'nl', io_options={"symbolic_solver_labels": True})

            fname, symbolMap = pyomo.opt.WriterFactory('nl')(model, filename, lambda x:True, {})
            varToIndex = pyomo.core.kernel.component_map.ComponentMap()
            conToIndex = pyomo.core.kernel.component_map.ComponentMap()
            for name, obj in six.iteritems(symbolMap.bySymbol):
                if name[0] == 'v':
                    varToIndex[obj()] = int(name[1:])
                elif name[0] == 'c':
                    conToIndex[obj()] = int(name[1:])

            self._varToIndex = varToIndex
            self._conToIndex = conToIndex

            nl_file = filename+".nl"

            super(PyomoNLP, self).__init__(nl_file)
            self._model = model

        finally:
            shutil.rmtree(temporal_dir)

    # Query methods
    def Grad_objective(self, x, variables=None):

        df = self.create_vector_x()

        self._asl.eval_deriv_f(x, df)

        if variables is not None:
            var_indices = []
            for v in variables:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices.append(var_id)
            return df[var_indices]

        return df

    def Evaluate_g(self, x, constraints=None):

        res = self.evaluate_g(x)

        if constraints is not None:
            con_indices = []
            for c in constraints:
                if c.is_indexed():
                    for cd in c.values():
                        con_id = self._conToIndex[cd]
                        con_indices.append(con_id)
                else:
                    con_id = self._conToIndex[c]
                    con_indices.append(con_id)
            return res[con_indices]

        return res

    def Jacobian_g(self, x, variables=None, constraints=None):

        subset_vars = False
        if variables is not None:
            var_indices = []
            for v in variables:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices.append(var_id)
            indices_vars = var_indices
            subset_vars = True

        subset_constraints = False
        if constraints is not None:
            con_indices = []
            for c in constraints:
                if c.is_indexed():
                    for cd in c.values():
                        con_id = self._conToIndex[cd]
                        con_indices.append(con_id)
                else:
                    con_id = self._conToIndex[c]
                    con_indices.append(con_id)
            indices_constraints = con_indices
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

        return COOMatrix((data, (new_row_indices, new_col_indices)),
                         shape=(nrows, ncols))

    def Hessian_lag(self,
                    x,
                    lam,
                    variables_rows=None,
                    variables_cols=None,
                    eval_f_c=True):

        subset_cols = False
        if variables_cols is not None:
            var_indices_cols = []
            for v in variables_cols:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices_cols.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices_cols.append(var_id)

            indices_cols = var_indices_cols
            subset_cols = True

        subset_rows = False
        if variables_rows is not None:
            var_indices_rows = []
            for v in variables_rows:
                if v.is_indexed():
                    for vd in v.values():
                        var_id = self._varToIndex[vd]
                        var_indices_rows.append(var_id)
                else:
                    var_id = self._varToIndex[v]
                    var_indices_rows.append(var_id)
            indices_rows = var_indices_rows
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

        return COOMatrix((data, (new_row_indices, new_col_indices)),
                         shape=(nrows, ncols))

    def variable_order(self):

        var_order = [None] * self.nx
        for v, idx in self._varToIndex.items():
            var_order[idx] = v.name
        return var_order

    def constraint_order(self):

        con_order = [None] * self.ng
        for c, idx in self._conToIndex.items():
            con_order[idx] = c.name
        return con_order

    def variable_idx(self, var):
        if var.is_indexed():
            raise RuntimeError("Var must be not indexed")
        return self._varToIndex[var]

    def constraint_idx(self, constraint):
        if constraint.is_indexed():
            raise RuntimeError("Constraint must be not indexed")
        return self._conToIndex[constraint]

