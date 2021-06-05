"""
UncertaintySet class: defines generic methods and attributes
of an uncertainty set in the context of robust optimization. UncertaintySet objects only
contain data which describes the set, and does not contain any Pyomo object information.

Supports the following classes of uncertainty sets:

- UncertaintySet (user defined/implemented)
- Ellipsoidal
- AxesAlignedEllipsoidal
- Polyhedral
- Box
- BudgetSet
- Cardinality/Gamma
- Discrete
- FactorModel

Every uncertainty set must have:
function set_as_constraint(): a method returning a Constraint representation of the uncertainty set.
function point_in_set(): a method which takes a point and determines if it is in the uncertainty set.

"""

import six
import abc
import functools
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.param import Param, _ParamData, IndexedParam
from pyomo.core.expr import value
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters

def uncertainty_sets(obj):
    if not isinstance(obj, UncertaintySet):
        raise ValueError("Expected an UncertaintySet object, instead recieved %s" % (obj,))
    return obj

def column(matrix, i):
    # Get column i of a given multi-dimensional list
    return [row[i] for row in matrix]

@six.add_metaclass(abc.ABCMeta)
class UncertaintySet:
    '''
    Abstract class for custom user-defined uncertainty sets.
    '''

    def __init__(self, **kwargs):
        """
        Constructor for UncertaintySet base class

        Args:
             kwargs: use the kwargs for specifying data for the UncertaintySet object
        """
        return

    @property
    def dim(self):
        """
        UncertaintySet dimension, e.g. dimension of uncertain parameters
        """
        raise NotImplementedError

    @property
    def geometry(self):
        """
        UncertaintySet geometry:
        1 is linear,
        2 is convex nonlinear,
        3 is general nonlinear,
        4 is discrete.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_as_constraint(self, **kwargs):
        """
        An uncertainty set *must* have a set_as_constraint method. UncertaintySets are instantiated with "q" as
        the list of uncertain param objects. Returns a Pyomo Constraint object (could
        be indexed) representing the uncertainty set for use in the separation problem

        Args:
            **kwargs: may be used to pass any additional information needed to generate the constraint(s)
            representing the UncertaintySet
        """
        pass

    def point_in_set(self, uncertain_params, point, **kwargs):
        """
        This function takes a n-dimensional point, where n is the dimension of the vector of uncertain
        parameters q, and determines if it is in the set defined by set_as_constraint(). Returns True of False.
        Given the uncertainty set constraint, verify if the point "point" is in the set.

        Args:
            uncertain_params: list of uncertain param objects, to construct the uncertainty set constraint
            point: the point being checked for existence in the set.
        """

        # === Ensure point is of correct dimensionality as the uncertain parameters
        if len(point) != len(uncertain_params):
            raise AttributeError("Point must have same dimensions as uncertain parameters.")

        # === Check if uncertain_params are Params or Vars, if Params set value, if vars fix value
        if type(uncertain_params) is IndexedVar or type(uncertain_params) is IndexedParam:
            the_params = list(uncertain_params.values())
        else:
            the_params = uncertain_params
        for idx, p in enumerate(the_params):
            if isinstance(p, Param) or isinstance(p, _ParamData):
                p.value = point[idx]
            else:
                p.fix(point[idx])

        original_values = list(p.value for p in the_params)

        # === Generate constraint for set
        set_constraint = self.set_as_constraint(uncertain_params=the_params)

        # === value() returns True if the constraint is satisfied, False else.
        is_in_set = all(value(con.expr) for con in list(set_constraint.values()))

        # === Revert uncertain_params to their original value and unfix them
        for idx, p in enumerate(the_params):
            p.value = original_values[idx]
            if type(p) is Var or type(p) is _VarData:
                p.unfix()

        return is_in_set


class BoxSet(UncertaintySet):
    """
    Box uncertainty set
    """

    def __init__(self, bounds):
        """
        BoxSet constructor

        Args:
            bounds: a list of tuples or a list of lists giving (lb, ub) for each uncertain parameter, in the same order as the 'uncertain_params' required input that is to be supplied to the PyROS solve statement.
        """
        # === non-empty bounds
        if len(bounds) == 0:
            raise AttributeError("Vector of bounds must be non-empty")
        # === Real number valued bounds
        if not all(isinstance(bound, (int, float)) for tup in bounds for bound in tup):
            raise AttributeError("Bounds must be real numbers.")
        # === Ensure no bound is None e.g. all are bounded
        if any(bound is None for tup in bounds for bound in tup):
            raise AttributeError("All bounds for uncertain parameters must be real numbers, not None.")
        # === Ensure each tuple has a lower and upper bound
        if not all(len(b) == 2 for b in bounds):
            raise AttributeError("Vector of bounds must include a finite lb and ub for each uncertain parameter")
        # === Ensure each lb <= ub
        if not all(bound[0] <= bound[1] for bound in bounds):
               raise AttributeError("Lower bounds must be less than or equal to upper bounds")

        self.bounds = bounds
        self.type = "box"

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def geometry(self):
        return 1

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the BoxSet uncertainty set.

        Args:
            uncertain_params: uncertain parameter objects for writing constraint objects
        """

        conlist = ConstraintList()
        conlist.construct()

        set_i = list(range(len(uncertain_params)))

        for i in set_i:
            conlist.add(uncertain_params[i] >= self.bounds[i][0])
            conlist.add(uncertain_params[i] <= self.bounds[i][1])

        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        bounds = set.bounds
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            lb = bounds[i][0]
            ub = bounds[i][1]
            p.setlb(lb)
            p.setub(ub)
        return


class CardinalitySet(UncertaintySet):
    """
    Cardinality uncertainty set
    """

    def __init__(self, origin, positive_deviation, gamma):
        """
        CardinalitySet constructor

        Args:
            origin: the origin of the set (e.g., the nominal point)
            positive_deviation: vector (list) of max deviations of each parameter
            gamma: scalar to bound the total number of params that can maximally deviate from 'origin'. gamma = 0 reduces the set to the 'origin' point. gamma = number-of-params makes the hyper-rectangle [origin, origin+positive_deviation]
        """
        # === Real number valued data
        if not all(isinstance(elem, (int, float)) for elem in origin):
            raise AttributeError("Elements of origin vector must be numeric.")
        if not all(isinstance(elem, (int, float)) for elem in positive_deviation):
            raise AttributeError("Elements of positive_deviation vector must be numeric")
        # === Dimension of positive_deviations and origin must be same
        if len(origin) != len(positive_deviation):
            raise AttributeError("Vectors for origin and positive_deviation must have same dimensions.")
        # === Gamma between 0,1
        if gamma < 0 or gamma > len(origin):
            raise AttributeError("Gamma parameter must be in [0, n].")
        # === positive_deviations must all be >= 0
        if any(elem < 0 for elem in positive_deviation):
            raise AttributeError("Elements of positive_deviations vector must be non-negative.")
        # === Non-emptiness is implied

        self.origin = origin
        self.positive_deviation = positive_deviation
        self.gamma = gamma
        self.type = "cardinality"

    @property
    def dim(self):
        return len(self.origin)

    @property
    def geometry(self):
        return 1

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the CardinalitySet uncertainty set.

        Args:
            uncertain_params: uncertain parameter objects for writing constraint objects
        """

        # === Ensure dimensions
        if len(uncertain_params) != len(self.origin):
               raise AttributeError("Dimensions of origin and uncertain_param lists must be equal.")

        model = kwargs['model']
        set_i = []

        for i in range(len(uncertain_params)):
            set_i.append(i)
        model.util.cassi = Var(set_i, initialize=0, bounds=(0, 1))

        # Make n equality constraints
        conlist = ConstraintList()
        conlist.construct()
        for i in set_i:
            conlist.add(self.origin[i] + self.positive_deviation[i] * model.util.cassi[i] == uncertain_params[i])

        conlist.add(sum(model.util.cassi[i] for i in set_i) <= self.gamma)

        return conlist

    def point_in_set(self, uncertain_params, point, **kwargs):
        """
        CardinalitySet class-specific method for identifying if a point is in the set

        Args:
             uncertain_params: uncertain parameter objects
             point: the point to query if it is in the set or not
        """

        cassis = []
        for i in range(len(uncertain_params)):
            if self.positive_deviation[i] > 0:
                cassis.append((point[i] - self.origin[i])/self.positive_deviation[i])

        if sum(cassi for cassi in cassis) <= self.gamma and \
            all(cassi >= 0 and cassi <= 1 for cassi in cassis):
            return True
        else:
            return False

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        nom_val = config.nominal_uncertain_param_vals
        deviation = set.positive_deviation
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            lb = nom_val[i]
            ub = nom_val[i] + min(config.uncertainty_set.gamma, 1) * deviation[i]
            p.setlb(lb)
            p.setub(ub)
        return


class PolyhedralSet(UncertaintySet):
    """
    Polyhedral uncertainty set
    """

    def __init__(self, lhs_coefficients_mat, rhs_vec):
        """
        PolyhedralSet constructor

        Args:
            lhs_coefficients_mat: matrix of coefficients for inequality constraints defining the polyhedral set
            rhs_vec: right-hand side vector for the inequality constraints
        """

        # === Real valued data
        mat = np.asarray(lhs_coefficients_mat)
        if not all(isinstance(elem, (int, float)) for row in lhs_coefficients_mat for elem in row):
            raise AttributeError("Matrix lhs_coefficients_mat must be real-valued and numeric.")
        if not all(isinstance(elem, (int, float)) for elem in rhs_vec):
            raise AttributeError("Vector rhs_vec must be real-valued and numeric.")
        # === Check columns of A must be same length as rhs
        if mat.shape[0] != len(rhs_vec):
            raise AttributeError("Rows of lhs_coefficients_mat matrix must equal length of rhs_vec list.")
        # === Columns are non-zero
        if mat.shape[1] == 0:
            raise AttributeError("Columns of lhs_coefficients_mat must be non-zero.")
        # === Matrix is not all zeros
        if all(np.isclose(elem, 0) for row in lhs_coefficients_mat for elem in row):
            raise AttributeError("Matrix lhs_coefficients_mat cannot be all zeroes.")
        # === Non-emptiness
        res = sp.optimize.linprog(c=np.zeros(mat.shape[1]), A_ub=mat, b_ub=rhs_vec, method="simplex")
        if not res.success:
            raise AttributeError("Error in evaluation of non-emptiness of the PolyhedralSet. Recieved " +
                                 res.message + " output from scipy.optimize.linprog linear program solver.")
        # === Boundedness
        if res.status == 3:
            # scipy linprog status == 3 indicates unboundedness
            raise AttributeError("PolyhedralSet appears to be unbounded. "
                                 "PyROS requires a bounded UncertaintySet object."
                                 "Please ensure the UncertaintySet is bounded.")


        self.coefficients_mat = lhs_coefficients_mat
        self.rhs_vec = rhs_vec
        self.type = "polyhedral"

    @property
    def dim(self):
        return len(self.coefficients_mat[0])

    @property
    def geometry(self):
        return 1

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the PolyhedralSet uncertainty set.

        Args:
            uncertain_params: uncertain parameter objects for writing constraint objects
        """

        # === Ensure valid dimensions of lhs and rhs w.r.t uncertain_params
        if np.asarray(self.coefficients_mat).shape[1] != len(uncertain_params):
            raise AttributeError("Columns of coefficients_mat matrix "
                                 "must equal length of uncertain parameters list.")

        set_i = []
        for i in range(len(self.coefficients_mat)):
            set_i.append(i)

        conlist = ConstraintList()
        conlist.construct()

        for i in set_i:
            constraint = 0
            for j in range(len(uncertain_params)):
                constraint += float(self.coefficients_mat[i][j]) * uncertain_params[j]
            conlist.add(constraint <= float(self.rhs_vec[i]))

        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        add_bounds_for_uncertain_parameters(separation_model=model, config=config)
        return


class BudgetSet(PolyhedralSet):
    """
    Budget uncertainty set
    """

    def __init__(self, budget_membership_mat,  rhs_vec):
        """
        BudgetSet constructor

        Args:
            budget_membership_mat: A matrix with 0-1 entries. Each row represents a budget constraint designating which parameter participates in the given budget. L is the total number of budget constraints.
            rhs_vec: length L vector of right-hand side values for the budgets
        """
        # === Non-zero number of columns
        mat = np.asarray(budget_membership_mat)
        rhs = np.asarray(rhs_vec)

        if len(mat.shape) == 1:
            cols = mat.shape
        else:
            cols = mat.shape[1]
        if cols == 0:
            raise AttributeError("Budget membership matrix must have non-zero number of columns.")
        # === Assert is valid matrix (same number of columns across all rows
        if not all(len(row) == cols for row in budget_membership_mat):
                raise AttributeError("Budget membership matrix must be a valid matrix, "
                                     "e.g. same number of column entries across rows.")
        # === Matrix dimension compatibility
        if mat.shape[0] != rhs.shape[0] :
               raise AttributeError("Rows of lhs_coefficients_mat matrix must equal rows of rhs_vec lists.")
        # === Ensure a 0-1 matrix
        if any(not np.isclose(elem, 0) and not np.isclose(elem, 1) for row in budget_membership_mat for elem in row):
            raise AttributeError("Budget membership matrix must be a matrix of 0's and 1's.")
        # === No all zero rows
        if all(elem == 0 for row in budget_membership_mat for elem in row):
               raise AttributeError("All zero rows are not permitted in the budget membership matrix.")

        # === Ensure 0 <= rhs_i for all i
        if any(rhs_vec[i] < 0 for i in range(len(rhs_vec))):
            raise AttributeError("RHS vector entries must be >= 0.")
        # === Non-emptiness is implied by the set

        self.coefficients_mat = budget_membership_mat
        self.rhs_vec = rhs_vec

        self.type = "budget"

    @property
    def dim(self):
        return np.asarray(self.coefficients_mat).shape[1]

    @property
    def geometry(self):
        return 1


    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the BudgetSet uncertainty set.

        Args:
            uncertain_params: uncertain parameter objects for writing constraint objects
        """

        # === Ensure matrix cols == len uncertain params
        if np.asarray(self.coefficients_mat).shape[1] != len(uncertain_params):
               raise AttributeError("Budget membership matrix must have compatible "
                                    "dimensions with uncertain parameters vector.")

        conlist = PolyhedralSet.set_as_constraint(self, uncertain_params)
        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        membership_mat = set.coefficients_mat
        rhs_vec = set.rhs_vec
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            col = column(membership_mat, i)
            ub = min(list(col[j] * rhs_vec[j] for j in range(len(rhs_vec))))
            lb = 0
            p.setlb(lb)
            p.setub(ub)
        return


class FactorModelSet(UncertaintySet):
    """
    Factor model uncertainty set
    """

    def __init__(self, origin, number_of_factors, psi_mat, beta):
        """
        FactorModelSet constructor

        Args:
            origin: vector (list) of nominal parameter values
            number_of_factors: natural number representing the dimensionality of the hypercube where independent factors (psi) reside
            psi: matrix of factors with strictly positive entries
            beta: in [0,1], parameter limiting how many factors achieve their extreme values. beta = 0 means equal numbers of cassi vars in [-1,1]^num_of_factors achieve their lower and upper bounds. beta = 1 reduces to num_of_factors-dimensional hypercube.
        """
        mat = np.asarray(psi_mat)
        # === Numeric valued arrays
        if not all(isinstance(elem, (int, float)) for elem in origin):
            raise AttributeError("All elements of origin vector must be numeric.")
        if not all(isinstance(elem, (int, float)) for row in psi_mat for elem in row):
            raise AttributeError("All elements of psi_mat vector must be numeric.")
        if not isinstance(beta, (int, float)):
            raise AttributeError("Beta parameter must be numeric.")
        if not isinstance(number_of_factors, (int)):
            raise AttributeError("number_of_factors must be integer.")
        # === Ensure dimensions of psi are n x F
        if mat.shape != (len(origin), number_of_factors):
                raise AttributeError("Psi matrix must be of dimensions n x F where n is dim(uncertain_params)"
                "and F is number_of_factors.")
        # === Ensure beta in [0,1]
        if beta > 1 or beta < 0:
            raise AttributeError("Beta parameter must be in [0,1].")
        # === No all zero columns of psi_mat
        for idx in range(mat.shape[1]):
            if all(np.isclose(elem, 0) for elem in mat[:,idx]):
                raise AttributeError("Psi matrix cannot have all zero columns.")
        # === Psi must be strictly positive entries
        for idx in range(mat.shape[1]):
            if any(elem < 0 for elem in mat[:,idx]):
                raise AttributeError("Psi matrix cannot have any negative entries. All factors must be strictly positive.")

        self.origin = origin
        self.number_of_factors = number_of_factors
        self.psi_mat = psi_mat
        self.beta = beta
        self.type = "factor_model"

    @property
    def dim(self):
        return len(self.origin)

    @property
    def geometry(self):
        return 1

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the FactorModelSet uncertainty set.

        Args:
            uncertain_params: uncertain parameter objects for writing constraint objects
        """
        model = kwargs['model']

        # === Ensure dimensions
        if len(uncertain_params) != len(self.origin):
                raise AttributeError("Dimensions of origin and uncertain_param lists must be equal.")

        # Make F-dim cassi variable
        n = []
        for i in range(self.number_of_factors):
            n.append(i)
        model.util.cassi = Var(n, initialize=0, bounds=(-1, 1))

        conlist = ConstraintList()
        conlist.construct()

        disturbances = []
        for i in range(len(uncertain_params)):
            disturbances.append(sum(self.psi_mat[i][j] * model.util.cassi[j] for j in n))

        # Make n equality constraints
        for i in range(len(uncertain_params)):
            conlist.add(self.origin[i] + disturbances[i] == uncertain_params[i])
        conlist.add(sum(model.util.cassi[i] for i in n) <= +self.beta * self.number_of_factors)
        conlist.add(sum(model.util.cassi[i] for i in n) >= -self.beta * self.number_of_factors)
        return conlist

    def point_in_set(self, uncertain_params, point, **kwargs):
        """
        FactorModelSet class-specific method for identifying if a point is in the set

        Args:
             uncertain_params: uncertain parameter objects
             point: the point to query if it is in the set or not
        """
        inv_psi = np.linalg.pinv(self.psi_mat)
        diff = np.asarray(list(point[i] - self.origin[i] for i in range(len(point))))
        cassis = np.dot(inv_psi, np.transpose(diff))

        if abs(sum(cassi for cassi in cassis)) <= self.beta * self.number_of_factors and \
            all(cassi >= -1 and cassi <= 1 for cassi in cassis):
            return True
        else:
            return False

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        nom_val = config.nominal_uncertain_param_vals
        psi_mat = set.psi_mat
        F = set.number_of_factors
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            sum_psi_row = sum(abs(psi_mat[i][j]) for j in range(F))
            lb = nom_val[i] - sum_psi_row
            ub = nom_val[i] + sum_psi_row
            p.setlb(lb)
            p.setub(ub)
        return


class AxisAlignedEllipsoidalSet(UncertaintySet):
    '''
    Axis-aligned ellipsoidal uncertainty set
    '''
    def __init__(self, center, half_lengths, scale=1):
        """
        AxisAlignedEllipsoidalSet constructor

        Args:
            origin: vector (list) of nominal parameter values.
            half_lengths: list of half-length values in each dimension (e.g. for each uncertain parameter)
            scale: right-hand side value for the ellipsoid
        """
        # === Valid data in lists
        if not all(isinstance(elem, (int, float)) for elem in half_lengths):
            raise AttributeError("Vector of half-lengths must be real-valued and numeric.")
        if not all(isinstance(elem, (int, float)) for elem in center):
            raise AttributeError("Vector center must be real-valued and numeric.")
        if not isinstance(scale, (int, float)):
            raise AttributeError("Ellipse scale must be a real-valued numeric.")
        # === Valid variance dimensions
        if not len(center) == len(half_lengths):
            raise AttributeError("Variances and center of ellipsoid must have same dimensions.")
        # === Ensure scale is non-negative
        if scale < 0:
            raise AttributeError("Scale of ellipse (rhs) must be non-negative.")

        self.center=center
        self.half_lengths=half_lengths
        self.scale=scale
        self.type="ellipsoidal"

    @property
    def dim(self):
        return len(self.center)

    @property
    def geometry(self):
        return 2

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the AxisAlignedEllipsoid uncertainty set.

        Args:
           uncertain_params: uncertain parameter objects for writing constraint objects
        """
        if len(uncertain_params) != len(self.center):
               raise AttributeError("Center of ellipsoid must be same dimensions as vector of uncertain parameters.")
        # square and invert half lengths
        inverse_squared_half_lengths = list(1.0/(a**2) for a in self.half_lengths)
        # Calculate row vector of differences
        diff_squared = []
        # === Assume VarList uncertain_param_vars
        for idx, i in enumerate(uncertain_params):
           if uncertain_params[idx].is_indexed():
               for index in uncertain_params[idx]:
                   diff_squared.append((uncertain_params[idx][index] - self.center[idx])**2)
           else:
               diff_squared.append((uncertain_params[idx] - self.center[idx])**2)

        # Calculate inner product of difference vector and variance matrix
        constraint = sum([x * y for x, y in zip(inverse_squared_half_lengths, diff_squared)])

        conlist = ConstraintList()
        conlist.construct()
        conlist.add(constraint <= self.scale)
        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''

        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            nom_value = config.nominal_uncertain_param_vals[i]
            half_length = config.uncertainty_set.half_lengths[i]
            p.setlb(nom_value - half_length)
            p.setub(nom_value + half_length)

        return


class EllipsoidalSet(UncertaintySet):
    """
    Ellipsoidal uncertainty set
    """

    def __init__(self, center, shape_matrix, scale=1):
        """
        EllipsoidalSet constructor

        Args:
            center: vector (list) of nominal parameter values.
            shape_matrix: positive-definite matrix
            scale: right-hand side value for the ellipsoid
        """

        # === Valid data in lists/matrixes
        if not all(isinstance(elem, (int, float)) for row in shape_matrix for elem in row):
            raise AttributeError("Matrix inv_covar must be real-valued and numeric.")
        if not all(isinstance(elem, (int, float)) for elem in center):
            raise AttributeError("Vector center must be real-valued and numeric.")
        if not isinstance(scale, (int, float)):
            raise AttributeError("Ellipse scale must be a real-valued numeric.")
        # === Valid matrix dimensions
        num_cols = len(shape_matrix[0])
        if not all(len(row) == num_cols for row in shape_matrix):
               raise AttributeError("Inverse covariance matrix must have valid matrix dimensions.")
        # === Ensure shape_matrix is a square matrix
        array_shape_mat = np.asarray(shape_matrix)
        if array_shape_mat.shape[0] != array_shape_mat.shape[1]:
                raise AttributeError("Inverse covariance matrix must be square.")
        # === Ensure dimensions of inv_covar are same as dimensions of uncertain_params
        if array_shape_mat.shape[1] != len(center):
                raise AttributeError("Inverse covariance matrix must be "
                                     "same dimensions as vector of uncertain parameters.")
        # === Symmetric shape_matrix
        if not np.all(np.abs(array_shape_mat-array_shape_mat.T) < 1e-8):
            raise AttributeError("Inverse covariance matrix must be symmetric.")
        # === Ensure scale is non-negative
        if scale < 0:
            raise AttributeError("Scale of ellipse (rhs) must be non-negative.")
        # === Check if shape matrix is invertible
        try:
            np.linalg.inv(shape_matrix)
        except np.linalg.LinAlgError as err:
            raise("Error with shape matrix supplied to EllipsoidalSet object being singular. %s" % err)
        # === Check is shape matrix is positive semidefinite
        if not all(np.linalg.eigvals(shape_matrix) >= 0):
            raise("Non positive-semidefinite shape matrix.")


        self.center = center
        self.shape_matrix = shape_matrix
        self.scale = scale
        self.type = "ellipsoidal"

    @property
    def dim(self):
        return len(self.center)

    @property
    def geometry(self):
        return 2

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the EllipsoidalSet uncertainty set.

        Args:
           uncertain_params: uncertain parameter objects for writing constraint objects
        """
        inv_covar = np.linalg.inv(self.shape_matrix)

        if len(uncertain_params) != len(self.center):
               raise AttributeError("Center of ellipsoid must be same dimensions as vector of uncertain parameters.")

        # Calculate row vector of differences
        diff = []
        # === Assume VarList uncertain_param_vars
        for idx, i in enumerate(uncertain_params):
            if uncertain_params[idx].is_indexed():
                for index in uncertain_params[idx]:
                    diff.append(uncertain_params[idx][index] - self.center[idx])
            else:
                diff.append(uncertain_params[idx] - self.center[idx])

        # Calculate inner product of difference vector and covar matrix
        product1 = []
        for i in range(len(inv_covar)):
            dot = sum([x * y for x, y in zip(diff, column(inv_covar, i))])
            product1.append(dot)
        constraint = sum([x * y for x, y in zip(product1, diff)])

        conlist = ConstraintList()
        conlist.construct()
        conlist.add(constraint <= self.scale)
        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        scale = set.scale
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            nom_value = config.nominal_uncertain_param_vals[i]
            P = set.shape_matrix
            P_ii = P[i][i]
            try:
                deviation = np.power(1.0/P_ii, 0.5)
            except:
                raise AttributeError("Shape matrix must be non-degenerate.")
            p.setlb(nom_value - deviation*scale)
            p.setub(nom_value + deviation*scale)

        return


class DiscreteSet(UncertaintySet):
    """
    Discrete uncertainty set
    """

    def __init__(self, scenarios):
        """
        DiscreteSet constructor

        Args:
            scenarios: vector (list) of discrete scenarios for each uncertain param realization. This is a list of dimensions (number of discrete scenarios) X (number uncertain parameters)
        """

        # === Non-empty
        if len(scenarios) == 0:
            raise AttributeError("Scenarios list must be non-empty.")
        # === Each scenario must be of real numbers
        if not all(isinstance(elem, (int, float)) for d in scenarios for elem in d):
            raise AttributeError("Each scenario must consist of real-number values for each parameter.")
        # === Confirm all scenarios are of same dimensionality
        dim = len(scenarios[0])
        if not all(len(d)==dim for d in scenarios):
               raise AttributeError("All points in list of scenarios must be same dimension.")

        self.scenarios = scenarios  # set of discrete points which are distinct realizations of uncertain params
        self.type = "discrete"

    @property
    def dim(self):
        return len(self.scenarios[0])

    @property
    def geometry(self):
        return 4

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the EllipsoidalSet uncertainty set.

        Args:
           uncertain_params: uncertain parameter objects for writing constraint objects
        """
        # === Ensure point is of correct dimensionality as the uncertain parameters
        dim = len(uncertain_params)
        if any(len(d) != dim for d in self.scenarios):
                raise AttributeError("All scenarios must have same dimensions as uncertain parameters.")

        conlist = ConstraintList()
        conlist.construct()

        for n in list(range(len(self.scenarios))):
            for i in list(range(len(uncertain_params))):
                conlist.add(uncertain_params[i] == self.scenarios[n][i])

        conlist.deactivate()
        return conlist

    def point_in_set(self, uncertain_params, point, **kwargs):
        """
        DiscreteSet class-specific method for identifying if a point is in the set

        Args:
             uncertain_params: uncertain parameter objects
             point: the point to query if it is in the set or not
        """
        # Round all double precision to a tolerance
        num_decimals = 8
        rounded_scenarios = list(list(round(num, num_decimals) for num in d) for d in self.scenarios)
        rounded_point = list(round(num, num_decimals) for num in point)

        return any(rounded_point==rounded_d for rounded_d in rounded_scenarios)

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        '''
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
            config: the config object for the PyROS solver instance
        '''
        set = config.uncertainty_set
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            min_i = min(s[i] for s in set.scenarios)
            max_i = max(s[i] for s in set.scenarios)
            p.setlb(min_i)
            p.setub(max_i)
        return


class IntersectionSet(UncertaintySet):
    """
    A class for set intersections of PyROS UncertaintySets
    """

    def __init__(self, **kwargs):
        """
        IntersectedSet constructor

        Args:
            **kwargs: keyword arguments for specifying all PyROS UncertaintySet objects to be intersected
        """
        if not all(isinstance(a_set, UncertaintySet) for a_set in kwargs.values()):
            raise ValueError("SetIntersection objects can only be constructed via UncertaintySet objects.")

        # === dim must be defined on all UncertaintySet objects
        all_sets = list(a_set for a_set in kwargs.values())
        if len(all_sets) < 2:
            raise AttributeError("SetIntersection requires 2 or more UncertaintySet objects.")

        a_dim = all_sets[0].dim
        if not all(uncertainty_set.dim == a_dim for uncertainty_set in all_sets):
            raise AttributeError("Uncertainty sets being intersected must have equal dimension.")

        self.all_sets = all_sets
        self.type = "intersection"

    @property
    def dim(self):
        return self.all_sets[0].dim

    @property
    def geometry(self):
        return max(self.all_sets[i].geometry for i in range(len(self.all_sets)))

    def point_in_set(self, uncertain_params, point, **kwargs):
        """
        IntersectedSet class-specific method for identifying if a point is in the set

        Args:
             uncertain_params: uncertain parameter objects
             point: the point to query if it is in the set or not
        """
        if all(a_set.point_in_set(uncertain_params=uncertain_params, point=point) for a_set in self.all_sets):
            return True
        else:
            return False

    def is_empty_intersection(self, uncertain_params, nlp_solver):
        """
        Determine if intersection is empty

        Args:
            uncertain_params: list of uncertain parameters
            nlp_solver: a Pyomo Solver object for solving NLPs
        """

        # === Non-emptiness check for the set intersection
        is_empty_intersection = True
        if any(a_set.type == "discrete" for a_set in self.all_sets):
            disc_sets = (a_set for a_set in self.all_sets if a_set.type == "discrete")
            disc_set = min(disc_sets, key=lambda x: len(x.scenarios))  # minimum set of scenarios
            # === Ensure there is at least one scenario from this discrete set which is a member of all other sets
            for scenario in disc_set.scenarios:
                if all(a_set.point_in_set(uncertain_params, scenario) for a_set in self.all_sets):
                    is_empty_intersection = False
                    break
        else:
            # === Compile constraints and solve NLP
            m = ConcreteModel()
            m.param_vars = Var(uncertain_params.index_set())
            for a_set in self.all_sets:
                m.add_component(a_set.type + "_constraints", a_set.set_as_constraint(uncertain_params=m.param_vars))
            try:
                res = nlp_solver.solve(m)
            except:
                raise ValueError("Solver terminated with an error while checking set intersection non-emptiness.")
            if check_optimal_termination(res):
                is_empty_intersection = False
        return is_empty_intersection

    # === Define pairwise intersection function
    @staticmethod
    def intersect(Q1, Q2, uncertain_params):
        """
        Binary function intersecting two UncertaintySet objects
        Args:
            Q1: uncertainty set 1
            Q2: uncertainty set 2
            uncertain_params: list of param objects to build constraints in point_in_set to determine if
            there is overlap between discrete and discrete/continuous sets.
        """
        constraints = ConstraintList()
        constraints.construct()

        for set in (Q1, Q2):
            other = Q1 if set is Q2 else Q2
            if set.type == "discrete":
                intersected_scenarios = []
                for point in set.scenarios:
                    if other.point_in_set(point=point, uncertain_params=uncertain_params):
                        intersected_scenarios.append(point)
                return DiscreteSet(scenarios=intersected_scenarios)

        # === This case is if both sets are continuous
        return IntersectionSet(set1=Q1, set2=Q2)

        return

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Function to generate constraints for the IntersectedSet uncertainty set.
        Args:
            uncertain_params: list of uncertain param objects participating in the sets to be intersected
        """
        try:
            nlp_solver = kwargs["config"].local_solver
        except:
            raise AttributeError("set_as_constraint for SetIntersection requires access to an NLP solver via"
                                 "the PyROS Solver config.")
        is_empty_intersection = self.is_empty_intersection(uncertain_params=uncertain_params, nlp_solver=nlp_solver)

        def _intersect(Q1, Q2):
            return self.intersect(Q1, Q2, uncertain_params)

        if not is_empty_intersection:
            Qint = functools.reduce(_intersect, self.all_sets)

            if Qint.type == "discrete":
                return Qint.set_as_constraint(uncertain_params=uncertain_params)
            else:
                conlist = ConstraintList()
                conlist.construct()
                for set in Qint.all_sets:
                    for con in list(set.set_as_constraint(uncertain_params=uncertain_params).values()):
                        conlist.add(con.expr)
                return conlist
        else:
            raise AttributeError("Set intersection is empty, cannot proceed with PyROS robust optimization.")

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        """
        Add bounds on uncertain parameters

        Args:
            model: The model to add bounds on for the uncertain parameter variable objects
        """

        add_bounds_for_uncertain_parameters(separation_model=model, config=config)
        return
