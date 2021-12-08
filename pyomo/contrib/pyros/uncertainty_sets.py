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
- IntersectedSet
"""

import abc
import functools
import math
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
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

class Geometry(Enum):
    '''
    Enum defining uncertainty set geometries
    '''
    LINEAR = 1
    CONVEX_NONLINEAR = 2
    GENERAL_NONLINEAR = 3
    DISCRETE_SCENARIOS = 4


class UncertaintySet(object, metaclass=abc.ABCMeta):
    '''
    Base class for custom user-defined uncertainty sets.
    '''

    def __init__(self, **kwargs):
        """
        Constructor for UncertaintySet base class

        Args:
             kwargs: Use the kwargs for specifying data for the UncertaintySet object. This data should be used in defining constraints in the 'set_as_constraint' function.
        """
        return

    @property
    @abc.abstractmethod
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def geometry(self):
        """
        UncertaintySet geometry:
        1 is linear,
        2 is convex nonlinear,
        3 is general nonlinear,
        4 is discrete.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        raise NotImplementedError

    def is_bounded(self, config):
        """
        Return True if the uncertainty set is bounded, else False.
        """
        # === Determine bounds on all uncertain params
        bounding_model = ConcreteModel()
        bounding_model.util = Block() # So that boundedness checks work for Cardinality and FactorModel sets
        bounding_model.uncertain_param_vars = IndexedVar(range(len(config.uncertain_params)), initialize=1)
        for idx, param in enumerate(config.uncertain_params):
            bounding_model.uncertain_param_vars[idx].set_value(
                param.value, skip_validation=True)

        bounding_model.add_component("uncertainty_set_constraint",
                                     config.uncertainty_set.set_as_constraint(
                                         uncertain_params=bounding_model.uncertain_param_vars,
                                         model=bounding_model,
                                         config=config
                                     ))

        for idx, param in enumerate(list(bounding_model.uncertain_param_vars.values())):
            bounding_model.add_component("lb_obj_" + str(idx), Objective(expr=param, sense=minimize))
            bounding_model.add_component("ub_obj_" + str(idx), Objective(expr=param, sense=maximize))

        for o in bounding_model.component_data_objects(Objective):
            o.deactivate()

        for i in range(len(bounding_model.uncertain_param_vars)):
            for limit in ("lb", "ub"):
                getattr(bounding_model, limit + "_obj_" + str(i)).activate()
                res = config.global_solver.solve(bounding_model, tee=False)
                getattr(bounding_model, limit + "_obj_" + str(i)).deactivate()
                if not check_optimal_termination(res):
                    return False
        return True

    def is_nonempty(self, config):
        """
        Return True if the uncertainty set is nonempty, else False.
        """
        return self.is_bounded(config)

    def is_valid(self, config):
        """
        Return True if the uncertainty set is bounded and non-empty, else False.
        """
        return self.is_nonempty(config=config) and self.is_bounded(config=config)

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

    def point_in_set(self, point):
        """
        Calculates if supplied ``point`` is contained in the uncertainty set. Returns True or False.

        Args:
            point: The point being checked for membership in the set.
                   The coordinates of the point should be supplied in the same order as the elements of ``uncertain_params``
                   that is to be supplied to the PyROS solve statement.
                   This point must match the dimension of the uncertain parameters of the set.
        """

        # === Ensure point is of correct dimensionality as the uncertain parameters
        if len(point) != self.dim:
            raise AttributeError("Point must have same dimensions as uncertain parameters.")

        m = ConcreteModel()
        the_params = []
        for i in range(self.dim):
            m.add_component("x_%s" % i, Var(initialize=point[i]))
            the_params.append(getattr(m, "x_%s" % i))

        # === Generate constraint for set
        set_constraint = self.set_as_constraint(uncertain_params=the_params)

        # === value() returns True if the constraint is satisfied, False else.
        is_in_set = all(value(con.expr) for con in set_constraint.values())

        return is_in_set

    @staticmethod
    def add_bounds_on_uncertain_parameters(**kwargs):
        """
        Numerical bounds on uncertain parameters are used in separation. This method should take a separation-type model
        and update the .lb() and .ub() property for each `uncertain_param_var` member of the model to a numerical value.
        This could be an inferred bound based on the uncertainty set itself, or a big-M type bound.
        If the bounds need to be numerically determined, return an empty list. See PolyhedralSet and IntersectedSet as examples.
        :param kwargs: the separation model and uncertainty set objects should be passed here.
        :return:
        """
        config = kwargs.pop('config')
        model = kwargs.pop('model')
        _set = config.uncertainty_set
        parameter_bounds = _set.parameter_bounds
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            p.setlb(parameter_bounds[i][0])
            p.setub(parameter_bounds[i][1])


class BoxSet(UncertaintySet):
    """
    Hyper-rectangle (a.k.a. "Box")
    """

    def __init__(self, bounds):
        """
        BoxSet constructor

        Args:
            bounds: A list of tuples providing lower and upper bounds (lb, ub) for each uncertain parameter, in the same order as the 'uncertain_params' required input that is to be supplied to the PyROS solve statement.
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
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.bounds)

    @property
    def geometry(self):
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        return self.bounds

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


class CardinalitySet(UncertaintySet):
    """
    Cardinality-constrained (a.k.a "Gamma") uncertainty set
    """

    def __init__(self, origin, positive_deviation, gamma):
        """
        CardinalitySet constructor

        Args:
            origin: The origin of the set (e.g., the nominal point).
            positive_deviation: Vector (``list``) of maximal deviations of each parameter.
            gamma: Scalar to bound the total number of uncertain parameters that can maximally deviate from their respective 'origin'. Setting 'gamma = 0' reduces the set to the 'origin' point. Setting 'gamma' to be equal to the number of parameters produces the hyper-rectangle [origin, origin+positive_deviation]
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
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.origin)

    @property
    def geometry(self):
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """

        nom_val = self.origin
        deviation = self.positive_deviation
        gamma = self.gamma
        parameter_bounds = [(nom_val[i], nom_val[i] + min(gamma, 1) * deviation[i]) for i in range(len(nom_val))]
        return parameter_bounds

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
        set_i = list(range(len(uncertain_params)))
        model.util.cassi = Var(set_i, initialize=0, bounds=(0, 1))

        # Make n equality constraints
        conlist = ConstraintList()
        conlist.construct()
        for i in set_i:
            conlist.add(self.origin[i] + self.positive_deviation[i] * model.util.cassi[i] == uncertain_params[i])

        conlist.add(sum(model.util.cassi[i] for i in set_i) <= self.gamma)

        return conlist

    def point_in_set(self, point):
        """
        Calculates if supplied ``point`` is contained in the uncertainty set. Returns True or False.

        Args:
            point: the point being checked for membership in the set
        """

        cassis = []
        for i in range(self.dim):
            if self.positive_deviation[i] > 0:
                cassis.append((point[i] - self.origin[i])/self.positive_deviation[i])

        if sum(cassi for cassi in cassis) <= self.gamma and \
            all(cassi >= 0 and cassi <= 1 for cassi in cassis):
            return True
        else:
            return False


class PolyhedralSet(UncertaintySet):
    """
    Polyhedral uncertainty set
    """

    def __init__(self, lhs_coefficients_mat, rhs_vec):
        """
        PolyhedralSet constructor

        Args:
            lhs_coefficients_mat: Matrix of left-hand side coefficients for the linear inequality constraints defining the polyhedral set.
            rhs_vec: Vector (``list``) of right-hand side values for the linear inequality constraints defining the polyhedral set.
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
            raise AttributeError("User-defined PolyhedralSet was determined to be empty. "
                                 "Please check the set of constraints supplied during set construction.")
        # === Boundedness
        if res.status == 3:
            # scipy linprog status == 3 indicates unboundedness
            raise AttributeError("User-defined PolyhedralSet was determined to be unbounded. "
                                 "Please augment the set of constraints supplied during set construction.")


        self.coefficients_mat = lhs_coefficients_mat
        self.rhs_vec = rhs_vec
        self.type = "polyhedral"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.coefficients_mat[0])

    @property
    def geometry(self):
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        PolyhedralSet bounds are not computed at set construction because they cannot be algebraically determined
        and require access to an optimization solver.
        """
        # For the PolyhedralSet, these are numerically determined
        # in the algorithm therefore they cannot presently be determined at construction of the set.
        return []

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

        set_i = list(range(len(self.coefficients_mat)))

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
        add_bounds_for_uncertain_parameters(model=model, config=config)
        return


class BudgetSet(PolyhedralSet):
    """
    Budget uncertainty set
    """

    def __init__(self, budget_membership_mat,  rhs_vec):
        """
        BudgetSet constructor

        Args:
            budget_membership_mat: A matrix with 0-1 entries to designate which uncertain parameters participate in each budget constraint. Here, each row is associated with a separate budget constraint.
            rhs_vec: Vector (``list``) of right-hand side values for the budget constraints.
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

        # === Add constraints such that uncertain params are >= 0
        # === This adds >=0 bound on all parameters in the set
        cols = mat.shape[1]
        identity = np.identity(cols) * -1
        for row in identity:
            budget_membership_mat.append(row.tolist())

        for i in range(identity.shape[0]):
            rhs_vec.append(0)

        self.coefficients_mat = budget_membership_mat
        self.rhs_vec = rhs_vec

        self.type = "budget"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return np.asarray(self.coefficients_mat).shape[1]

    @property
    def geometry(self):
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        membership_mat = np.asarray(self.coefficients_mat)
        rhs_vec = self.rhs_vec
        parameter_bounds = []
        for i in range(membership_mat.shape[1]):
            col = column(membership_mat, i)
            ub = min(list(col[j] * rhs_vec[j] for j in range(len(rhs_vec))))
            lb = 0
            parameter_bounds.append((lb, ub))
        return parameter_bounds

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
        # In this case, we use the UncertaintySet class method because we have numerical parameter_bounds
        UncertaintySet.add_bounds_on_uncertain_parameters(model=model, config=config)

class FactorModelSet(UncertaintySet):
    """
    Factor model (a.k.a. "net-alpha" model) uncertainty set
    """

    def __init__(self, origin, number_of_factors, psi_mat, beta):
        """
        FactorModelSet constructor

        Args:
            origin: Vector (``list``) of uncertain parameter values around which deviations are restrained.
            number_of_factors: Natural number representing the dimensionality of the space to which the set projects.
            psi: Matrix with non-negative entries designating each uncertain parameter's contribution to each factor. Here, each row is associated with a separate uncertain parameter and each column with a separate factor.
            beta: Number in [0,1] representing the fraction of the independent factors that can simultaneously attain their extreme values. Setting 'beta = 0' will enforce that as many factors will be above 0 as there will be below 0 (i.e., "zero-net-alpha" model). Setting 'beta = 1' produces the hyper-rectangle [origin - psi e, origin + psi e], where 'e' is the vector of ones.
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
                raise AttributeError("Psi matrix cannot have any negative entries. All factors must be non-negative.")

        self.origin = origin
        self.number_of_factors = number_of_factors
        self.psi_mat = psi_mat
        self.beta = beta
        self.type = "factor_model"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.origin)

    @property
    def geometry(self):
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        nom_val = self.origin
        psi_mat = self.psi_mat

        F = self.number_of_factors
        beta_F = self.beta * F
        floor_beta_F = math.floor(beta_F)
        parameter_bounds = []
        for i in range(len(nom_val)):
            non_decreasing_factor_row = sorted(psi_mat[i], reverse=True)
            # deviation = sum_j=1^floor(beta F) {psi_if_j} + (beta F - floor(beta F)) psi_{if_{betaF +1}}
            # because indexing starts at 0, we adjust the limit on the sum and the final factor contribution
            if beta_F - floor_beta_F == 0:
                deviation = sum(non_decreasing_factor_row[j] for j in range(floor_beta_F - 1))
            else:
                deviation = sum(non_decreasing_factor_row[j] for j in range(floor_beta_F - 1)) + (
                            beta_F - floor_beta_F) * psi_mat[i][floor_beta_F]
            lb = nom_val[i] - deviation
            ub = nom_val[i] + deviation
            if lb > ub:
                raise AttributeError("The computed lower bound on uncertain parameters must be less than or equal to the upper bound.")
            parameter_bounds.append((lb, ub))
        return parameter_bounds

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
        n = list(range(self.number_of_factors))
        model.util.cassi = Var(n, initialize=0, bounds=(-1, 1))

        conlist = ConstraintList()
        conlist.construct()

        disturbances = [sum(self.psi_mat[i][j] * model.util.cassi[j] for j in n)
                        for i in range(len(uncertain_params))]

        # Make n equality constraints
        for i in range(len(uncertain_params)):
            conlist.add(self.origin[i] + disturbances[i] == uncertain_params[i])
        conlist.add(sum(model.util.cassi[i] for i in n) <= +self.beta * self.number_of_factors)
        conlist.add(sum(model.util.cassi[i] for i in n) >= -self.beta * self.number_of_factors)
        return conlist


    def point_in_set(self, point):
        """
        Calculates if supplied ``point`` is contained in the uncertainty set. Returns True or False.

        Args:
             point: the point being checked for membership in the set
        """
        inv_psi = np.linalg.pinv(self.psi_mat)
        diff = np.asarray(list(point[i] - self.origin[i] for i in range(len(point))))
        cassis = np.dot(inv_psi, np.transpose(diff))

        if abs(sum(cassi for cassi in cassis)) <= self.beta * self.number_of_factors and \
            all(cassi >= -1 and cassi <= 1 for cassi in cassis):
            return True
        else:
            return False


class AxisAlignedEllipsoidalSet(UncertaintySet):
    '''
    Axis-aligned ellipsoidal uncertainty set
    '''
    def __init__(self, center, half_lengths):
        """
        AxisAlignedEllipsoidalSet constructor

        Args:
            center: Vector (``list``) of uncertain parameter values around which deviations are restrained.
            half_lengths: Vector (``list``) of half-length values representing the maximal deviations for each uncertain parameter.
        """
        # === Valid data in lists
        if not all(isinstance(elem, (int, float)) for elem in half_lengths):
            raise AttributeError("Vector of half-lengths must be real-valued and numeric.")
        if not all(isinstance(elem, (int, float)) for elem in center):
            raise AttributeError("Vector center must be real-valued and numeric.")
        if any(elem <= 0 for elem in half_lengths):
            raise AttributeError("Half length values must be > 0.")
        # === Valid variance dimensions
        if not len(center) == len(half_lengths):
            raise AttributeError("Half lengths and center of ellipsoid must have same dimensions.")

        self.center=center
        self.half_lengths=half_lengths
        self.type="ellipsoidal"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.center)

    @property
    def geometry(self):
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        nom_value = self.center
        half_length =self.half_lengths
        parameter_bounds = [(nom_value[i] - half_length[i], nom_value[i] + half_length[i]) for i in range(len(nom_value))]
        return parameter_bounds

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
        conlist.add(constraint <= 1)
        return conlist


class EllipsoidalSet(UncertaintySet):
    """
    Ellipsoidal uncertainty set
    """

    def __init__(self, center, shape_matrix, scale=1):
        """
        EllipsoidalSet constructor

        Args:
            center: Vector (``list``) of uncertain parameter values around which deviations are restrained.
            shape_matrix: Positive semi-definite matrix, effectively a covariance matrix for
            constraint and bounds determination.
            scale: Right-hand side value for the ellipsoid.
        """

        # === Valid data in lists/matrixes
        if not all(isinstance(elem, (int, float)) for row in shape_matrix for elem in row):
            raise AttributeError("Matrix shape_matrix must be real-valued and numeric.")
        if not all(isinstance(elem, (int, float)) for elem in center):
            raise AttributeError("Vector center must be real-valued and numeric.")
        if not isinstance(scale, (int, float)):
            raise AttributeError("Ellipse scale must be a real-valued numeric.")
        # === Valid matrix dimensions
        num_cols = len(shape_matrix[0])
        if not all(len(row) == num_cols for row in shape_matrix):
               raise AttributeError("Shape matrix must have valid matrix dimensions.")
        # === Ensure shape_matrix is a square matrix
        array_shape_mat = np.asarray(shape_matrix)
        if array_shape_mat.shape[0] != array_shape_mat.shape[1]:
                raise AttributeError("Shape matrix must be square.")
        # === Ensure dimensions of shape_matrix are same as dimensions of uncertain_params
        if array_shape_mat.shape[1] != len(center):
                raise AttributeError("Shape matrix must be "
                                     "same dimensions as vector of uncertain parameters.")
        # === Symmetric shape_matrix
        if not np.all(np.abs(array_shape_mat-array_shape_mat.T) < 1e-8):
            raise AttributeError("Shape matrix must be symmetric.")
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
        # === Ensure matrix is not degenerate, for determining inferred bounds
        try:
            for i in range(len(shape_matrix)):
                np.power(shape_matrix[i][i], 0.5)
        except:
            raise AttributeError("Shape matrix must be non-degenerate.")

        self.center = center
        self.shape_matrix = shape_matrix
        self.scale = scale
        self.type = "ellipsoidal"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.center)

    @property
    def geometry(self):
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        scale = self.scale
        nom_value = self.center
        P = self.shape_matrix
        parameter_bounds = [(nom_value[i] - np.power(P[i][i], 0.5) * scale,
                             nom_value[i] + np.power(P[i][i], 0.5) * scale) for i in range(self.dim)]
        return parameter_bounds

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
        product1 = [sum([x * y for x, y in zip(diff, column(inv_covar, i))]) for i in range(len(inv_covar))]
        constraint = sum([x * y for x, y in zip(product1, diff)])

        conlist = ConstraintList()
        conlist.construct()
        conlist.add(constraint <= self.scale)
        return conlist


class DiscreteScenarioSet(UncertaintySet):
    """
    Set of discrete scenarios (i.e., finite collection of realizations)
    """

    def __init__(self, scenarios):
        """
        DiscreteScenarioSet constructor

        Args:
            scenarios: Vector (``list``) of discrete scenarios where each scenario represents a realization of the uncertain parameters.
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

        # Standardize to list of tuples
        self.scenarios = list(tuple(s) for s in scenarios)  # set of discrete points which are distinct realizations of uncertain params
        self.type = "discrete"

    @property
    def dim(self):
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return len(self.scenarios[0])

    @property
    def geometry(self):
        return Geometry.DISCRETE_SCENARIOS

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        """
        parameter_bounds = [(min(s[i] for s in self.scenarios),
                             max(s[i] for s in self.scenarios)) for i in range(self.dim)]
        return parameter_bounds

    def is_bounded(self, config):
        '''
        DiscreteScenarios is bounded by default due to finiteness of the set.
        :param config:
        :return: True
        '''
        return True

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

    def point_in_set(self, point):
        """
        Calculates if supplied ``point`` is contained in the uncertainty set. Returns True or False.

        Args:
             point: the point being checked for membership in the set
        """
        # Round all double precision to a tolerance
        num_decimals = 8
        rounded_scenarios = list(list(round(num, num_decimals) for num in d) for d in self.scenarios)
        rounded_point = list(round(num, num_decimals) for num in point)

        return any(rounded_point==rounded_d for rounded_d in rounded_scenarios)


class IntersectionSet(UncertaintySet):
    """
    Set stemming from intersecting previously constructed sets of any type
    """

    def __init__(self, **kwargs):
        """
        IntersectionSet constructor

        Args:
            **kwargs: Keyword arguments for specifying all PyROS UncertaintySet objects to be intersected.
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
        """
        Dimension of the uncertainty set, i.e., number of parameters in “uncertain_params” list.
        """
        return self.all_sets[0].dim

    @property
    def geometry(self):
        return max(self.all_sets[i].geometry.value for i in range(len(self.all_sets)))

    @property
    def parameter_bounds(self):
        """
        Bounds on the realizations of the uncertain parameters, as inferred from the uncertainty set.
        IntersectedSet bounds are not computed at set construction because they cannot be algebraically determined
        and require access to an optimization solver.
        """
        # For the IntersectedSet, these are numerically determined
        # in the algorithm therefore they cannot presently be determined at construction of the set.
        return []

    def point_in_set(self, point):
        """
        Calculates if supplied ``point`` is contained in the uncertainty set. Returns True or False.

        Args:
             point: the point being checked for membership in the set
        """
        if all(a_set.point_in_set(point=point) for a_set in self.all_sets):
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
                if all(a_set.point_in_set(point=scenario) for a_set in self.all_sets):
                    is_empty_intersection = False
                    break
        else:
            # === Compile constraints and solve NLP
            m = ConcreteModel()
            m.obj = Objective(expr=0) # dummy objective required if using baron
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
    def intersect(Q1, Q2):
        """
        Binary function intersecting two UncertaintySet objects
        Args:
            Q1: uncertainty set 1
            Q2: uncertainty set 2
        """
        constraints = ConstraintList()
        constraints.construct()

        for set in (Q1, Q2):
            other = Q1 if set is Q2 else Q2
            if set.type == "discrete":
                intersected_scenarios = []
                for point in set.scenarios:
                    if other.point_in_set(point=point):
                        intersected_scenarios.append(point)
                return DiscreteScenarioSet(scenarios=intersected_scenarios)

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
            nlp_solver = kwargs["config"].global_solver
        except:
            raise AttributeError("set_as_constraint for SetIntersection requires access to an NLP solver via"
                                 "the PyROS Solver config.")
        is_empty_intersection = self.is_empty_intersection(uncertain_params=uncertain_params, nlp_solver=nlp_solver)

        def _intersect(Q1, Q2):
            return self.intersect(Q1, Q2)

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

        add_bounds_for_uncertain_parameters(model=model, config=config)
        return
