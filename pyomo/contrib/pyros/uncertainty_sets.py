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
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters


valid_num_types = tuple(native_numeric_types)


def validate_arg_type(
        arg_name,
        arg_val,
        valid_types,
        valid_type_desc=None,
        is_entry_of_arg=False,
        ):
    """
    Perform type validation of an argument to a function/method.
    If type is not valid, raise a TypeError with an appropriate
    message.

    Parameters
    ----------
    arg_name : str
        Name of argument to be displayed in exception message.
    arg_val : object
        Value of argument to be checked.
    valid_types : type or tuple of types
        Valid types for the argument value.
    valid_type_desc : str or None, optional
        Description of valid types for the argument value;
        this description is included in the exception message.
    is_entry_of_arg : bool, optional
        Is the argument value passed an entry of the argument
        described by `arg_name` (such as entry of an array or list).
        This will be indicated in the exception message.
        The default is `False`.

    Raises
    ------
    ValueError
        If the argument value is not a valid type.
    """
    if not isinstance(arg_val, valid_types):
        if valid_type_desc is not None:
            type_phrase = f"not {valid_type_desc}"
        else:
            valid_type_str = ", ".join(dtype.__name__ for dtype in valid_types)
            type_phrase = f"not of any of the valid types ({valid_type_str})"

        if is_entry_of_arg:
            raise TypeError(
                f"Entry '{arg_val}' of the argument `{arg_name}` "
                f"is {type_phrase} (provided type '{type(arg_val).__name__}')"
            )
        else:
            raise TypeError(
                f"Argument `{arg_name}` is {type_phrase} "
                f"(provided type '{type(arg_val).__name__}')"
            )


def is_ragged(arr, arr_types=None):
    """
    Determine whether an array-like (such as a list or ndarray)
    is ragged.

    NOTE: if ndarrays are considered are considered arr types,
    we do not consider zero-dimensional arrays to be as such.
    """
    arr_types = (list, np.ndarray, tuple) if arr_types is None else arr_types

    is_zero_dim_arr = isinstance(arr, np.ndarray) and len(arr.shape) == 0
    if not isinstance(arr, arr_types) or is_zero_dim_arr:
        return False

    entries_are_seqs = []
    for entry in arr:
        if np.ndarray in arr_types and isinstance(entry, np.ndarray):
            # account for 0-D arrays (treat as non-arrays)
            entries_are_seqs.append(len(entry.shape) > 0)
        else:
            entries_are_seqs.append(isinstance(entry, arr_types))

    if not any(entries_are_seqs):
        return False
    if not all(entries_are_seqs):
        return True

    entries_ragged = [is_ragged(entry for entry in arr)]
    if any(entries_ragged):
        return True
    else:
        return any(
            np.array(arr[0]).shape != np.array(entry).shape for entry in arr
        )


def validate_dimensions(arr_name, arr, dim, display_value=False):
    """
    Validate dimension of an array-like object.
    Raise Exception if validation fails.
    """
    if is_ragged(arr):
        raise ValueError(
            f"Argument `{arr_name}` should not be a ragged array-like "
            "(nested sequence of lists, tuples, arrays of different shape)"
        )

    # check dimensions matched
    array = np.asarray(arr)
    if len(array.shape) != dim:
        val_str = f" from provided value {str(arr)}" if display_value else ""
        raise ValueError(
            f"Argument `{arr_name}` must be a "
            f"{dim}-dimensional array-like "
            f"(detected {len(array.shape)} dimensions{val_str})"
        )
    elif array.shape[-1] == 0:
        raise ValueError(
            f"Last dimension of argument `{arr_name}` must be non-empty "
            f"(detected shape {array.shape})"
        )


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
        Constructor for an N-dimensional `BoxSet`.

        Parameters
        ----------
        bounds : (N, 2) array_like
            Lower and upper bounds for each uncertain
            parameter (i.e. each dimension of the set).
            The order of the dimensions corresponds to
            the order of the uncertain parameters of interest.

        Examples
        --------
        1-D box set (interval):
        >>> interval = BoxSet(bounds=[(1, 2)])
        >>> box_set.bounds
        [(1, 2)]

        2-D box set with bounds specified by Numpy array:
        >>> import numpy as np
        >>> box_set = BoxSet(bounds=np.array([[1, 2], [3, 4]]))
        >>> box_set.bounds
        array([[1, 2],
               [3, 4]])
        """
        # validate shape
        validate_dimensions("bounds", bounds, 2)
        bounds_arr = np.asarray(bounds)
        if bounds_arr.shape[-1] != 2:
            raise ValueError(
                "Argument `bounds` to BoxSet constructor should be of shape "
                f"(...,2), but detected shape (...,{bounds_arr.shape[-1]})"
            )

        # validate bound types, values
        for lb_ub_pair in bounds:
            for val in lb_ub_pair:
                validate_arg_type(
                    "bounds", val, valid_num_types,
                    valid_type_desc="a valid numeric type",
                    is_entry_of_arg=True,
                )

            lb, ub = lb_ub_pair[0], lb_ub_pair[1]
            if lb > ub:
                raise ValueError(
                    f"Lower bound {lb} exceeds upper bound {ub}"
                )

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
        Constructor for the N-dimensional cardinality set

        Parameters
        ----------
        origin : (N,) array_like
            Origin of the set (e.g. nominal parameter values).
        positive_deviation : (N,) array_like
            Maximal deviations in each dimension (i.e. for each
            uncertain parameter).
        gamma : numeric type
            Scalar which provides an upper bound for the number
            of uncertain parameters which may maximally deviate
            from their respective origin values.
            Setting `gamma` to 0 reduces the set to a singleton
            containing the center, while setting `gamma` to N
            reduces the set to a hyperrectangle whose bounds
            are [origin, origin + max deviation] in each dimension.

        Examples
        --------
        3-D cardinality set:
        >>> gamma_set = CardinalitySet([0, 0, 0], [1.0, 2.0, 1.5], 1)
        >>> gamma_set.origin
        [0, 0, 0]
        >>> gamma_set.positive_deviation
        [1.0, 2.0, 1.5]
        >>> gamma_set.gamma
        1
        """
        # === Dimension of positive_deviations and origin must be same
        origin_len = len(origin)
        deviation_len = len(positive_deviation)
        if len(origin) != len(positive_deviation):
            raise ValueError(
                f"Arguments `origin` (length {origin_len}) and "
                f" `positive_deviation` (length {deviation_len}) "
                "do not have the same length "
            )

        for orig_val, dev_val in zip(origin, positive_deviation):
            # validate numeric type
            validate_arg_type(
                "origin", orig_val, valid_num_types,
                "a valid numeric type", True,
            )
            validate_arg_type(
                "positive_deviation", dev_val, valid_num_types,
                "a valid numeric type", True,
            )
            if dev_val < 0:
                raise ValueError(
                    f"Entry '{dev_val}' of the argument `positive_deviation` "
                    "is not non-negative (provided value {dev_val})"
                )

        # validate gamma
        validate_arg_type(
            "gamma", gamma, valid_num_types,
            "a valid numeric type", False,
        )
        if gamma < 0 or gamma > origin_len:
            raise ValueError(
                "For origin and deviations provided, argument "
                f"`gamma` must be a number between 0 and {origin_len} "
                f"(provided value {gamma})"
            )

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
        Constructor for an N-dimensional polyhedral set.

        Arguments
        ---------
        lhs_coefficients_mat : (M, N) array_like
            Left-hand side coefficients for the linear
            inequality constraints defining the polyhedral set.
        rhs_vec : (N,) array_like
            Right-hand side values for the linear inequality
            constraints defining the polyhedral set.
        """
        lhs_matrix_arr = np.asarray(lhs_coefficients_mat)
        rhs_vec_arr = np.asarray(rhs_vec)

        validate_dimensions("lhs_coefficients_mat", lhs_matrix_arr, 2)
        validate_dimensions("rhs_vec", rhs_vec_arr, 1)

        # ensure number of rows of matrix matches number of
        # entries of rhs vector
        if lhs_matrix_arr.shape[0] != rhs_vec_arr.shape[0]:
            raise ValueError(
                "Argument `lhs_coefficients_mat` "
                f"({lhs_matrix_arr.shape[0]} rows) "
                "does not have same number of rows as length of "
                f"`rhs_vec` ({len(rhs_vec)} entries)"
            )

        # validate entry types
        for row, rhs_entry in zip(lhs_coefficients_mat, rhs_vec):
            for coeff in row:
                validate_arg_type(
                    "lhs_coefficients_mat",
                    coeff,
                    valid_num_types,
                    "a valid numeric type",
                    is_entry_of_arg=True,
                )
            validate_arg_type(
                "rhs_vec",
                rhs_entry,
                valid_num_types,
                "a valid numeric type",
                is_entry_of_arg=True,
            )

        # === Matrix is not all zeros
        if np.all(np.isclose(lhs_coefficients_mat, 0)):
            raise ValueError(
                "Argument `lhs_coefficients_mat` must have"
                "at least one nonzero entry"
            )

        # solve LP to verify set is nonempty; check results
        # perhaps this can be moved to an is_em
        res = sp.optimize.linprog(
            c=np.zeros(lhs_matrix_arr.shape[0]),
            A_ub=lhs_matrix_arr,
            b_ub=rhs_vec,
            method="simplex",
            bounds=(None, None),
        )
        if res.status == 1 or res.status == 4:
            raise ValueError(
                "Could not verify nonemptiness of the "
                "polyhedral set (`scipy.optimize.linprog(method=simplex)` "
                f" status {res.status}) "
            )
        elif res.status == 2:
            raise ValueError(
                "PolyhedralSet defined by `lhs_coefficients_mat` and "
                "`rhs_vec` is empty. Check arguments"
            )
        elif res.status == 3:
            raise ValueError(
                "PolyhedralSet defined by `lhs_coefficients_mat` and "
                "`rhs_vec` is unbounded. Check arguments"
            )

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

    def __init__(self, budget_membership_mat, rhs_vec):
        """
        Constructor for an N-dimensional BudgetSet.

        Parameters
        ----------
        budget_member_mat : (M, N) array_like
            Incidence matrix of the budget constraints.
            Each row corresponds to a single budget constraint,
            and defines which uncertain parameters
            (which dimensions) participate in that row's constraint.
        rhs_vec : (N,) array_like
            Right-hand side values for the budget constraints.
        """
        # === Non-zero number of columns
        budget_mat_arr = np.asarray(budget_membership_mat)
        rhs_vec_arr = np.asarray(rhs_vec)

        validate_dimensions("budget_membership_mat", budget_mat_arr, 2)
        validate_dimensions("rhs_vec", rhs_vec_arr, 1)

        # === Matrix dimension compatibility
        if budget_mat_arr.shape[0] != rhs_vec_arr.shape[0]:
            raise ValueError(
                "Argument `budget_membership_mat` "
                f"({budget_mat_arr.shape[0]} rows) "
                "does not have same number of rows as entries of "
                f"`rhs_vec` ({rhs_vec_arr.shape[0]} entries)"
            )
        # === Ensure a 0-1 matrix
        for row, rhs_entry in zip(budget_membership_mat, rhs_vec):
            for entry in row:
                validate_arg_type(
                    "budget_membership_mat",
                    entry,
                    valid_num_types,
                    "a valid numeric type",
                    True,
                )
                if not np.any(np.isclose(entry, [0, 1])):
                    raise ValueError(
                        f"Entry {entry} of argument `budget_membership_mat`"
                        " is not 0 or 1"
                    )
            if np.allclose(row, 0):
                raise ValueError(
                   "Each row of argument `budget_membership_mat` should "
                   "have at least one nonzero entry"
                )
            validate_arg_type(
                "rhs_vec",
                rhs_entry,
                valid_num_types,
                "a valid numeric type",
                True,
            )
            if rhs_entry < 0:
                raise ValueError(
                    f"Entry {rhs_entry} of argument `rhs_vec` is negative. "
                    "Ensure all entries are nonnegative"
                )

        # add rows for nonnegativity constraints
        # on the uncertain parameters to the budget matrix and rhs vector
        num_cols = budget_mat_arr.shape[1]
        neg_identity = np.identity(num_cols) * -1
        for row in neg_identity:
            budget_membership_mat.append(row.tolist())
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
        Constructor for an N-dimensional factor model set.

        Parameters
        ----------
        origin : (N,) array_like
            Uncertain parameter values around which deviations are
            restrained.
        number_of_factors : int
            Natural number representing the dimensionality of the
            space to which the set projects.
        psi : (N, `number_of_factors`) array_like
            Matrix with nonnegative entires designating each
            uncertain parameter's contribution to each  factor.
            Each row is associated with a separate uncertain parameter.
            Each column with a separate factor.
        beta : numeric type
            Number in [0, 1] representing the fraction of the
            independent factors that can simultaneously attain
            their extreme values.
            Setting 'beta = 0' will enforce that as many
            factors will be above 0 as there will be below 0
            (i.e., "zero-net-alpha" model). Setting 'beta = 1'
            produces the hyper-rectangle
            [origin - psi e, origin + psi e],
            where 'e' is a vector of ones.
        """
        validate_dimensions("psi_mat", psi_mat, 2)
        validate_dimensions("origin", origin, 1)

        psi_mat_arr = np.asarray(psi_mat)

        validate_arg_type(
            "beta",
            beta,
            valid_num_types,
            "a valid numeric type",
            False,
        )
        validate_arg_type(
            "number_of_factors",
            number_of_factors,
            int,
            "an integer",
            False,
        )

        # ensure psi matrix shape matches origin dimensions
        # and number of factors
        if psi_mat_arr.shape != (len(origin), number_of_factors):
            raise ValueError(
                "Psi matrix for factor model set with "
                f"{number_of_factors} factors and "
                f"origin with {len(origin)} entries "
                f"should be of shape {number_of_factors, len(origin)} "
                f"(detected shape {psi_mat_arr.shape})"
            )

        for row, orig_entry in zip(psi_mat, origin):
            for entry in row:
                validate_arg_type(
                    "psi_mat",
                    entry,
                    valid_num_types,
                    "a valid numeric type",
                    True,
                )
                if entry < 0:
                    raise ValueError(
                        f"Entry {entry} of argument `psi_mat` is negative. "
                        "Check that all entries are nonnegative"
                    )
            validate_arg_type(
                "origin",
                orig_entry,
                valid_num_types,
                "a valid numeric type",
                True,
            )

        for column in psi_mat_arr.T:
            if np.allclose(column, 0):
                raise ValueError(
                    "Each column of argument `psi_mat` should have at least "
                    "one nonzero entry"
                )

        # === Ensure beta in [0,1]
        if beta > 1 or beta < 0:
            raise ValueError(
                "Beta parameter must be in [0, 1] "
                f"(provided value {beta})"
            )

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

        Parameters
        ----------
        center : (N,) array_like
            Center of the ellipsoid.
        half_lengths : (N,) aray_like
            Semi-axis lengths of the ellipsoid. Each value
            specifies the maximal deviation of its corresponding
            uncertain parameter from the central point.
        """
        validate_dimensions("center", center, 1)
        validate_dimensions("half_lengths", half_lengths, 1)

        # check parity of lengths
        if not len(center) == len(half_lengths):
            raise ValueError(
                f"Arguments `center` (length {len(center)}) and "
                f"`half_lengths` (length {len(half_lengths)}) "
                "are not of the same length"
            )

        # validate entry types
        for half_len, center_val in zip(half_lengths, center):
            validate_arg_type(
                "half_lengths",
                half_len,
                valid_num_types,
                "a valid numeric type",
                True,
            )
            validate_arg_type(
                "center",
                center_val,
                valid_num_types,
                "a valid numeric type",
                True,
            )

            if half_len < 0:
                raise ValueError(
                    f"Entry {half_len} of argument `half_lengths` "
                    "is negative. Ensure all half-lengths are nonnegative"
                )

        self.center = center
        self.half_lengths = half_lengths
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
        nom_value = self.center
        half_length =self.half_lengths
        parameter_bounds = [(nom_value[i] - half_length[i], nom_value[i] + half_length[i]) for i in range(len(nom_value))]
        return parameter_bounds

    def set_as_constraint(self, uncertain_params, model=None, config=None):
        """
        Generate constraint(s) for the `AxisAlignedEllipsoidSet`
        class.

        Args:
            uncertain_params: uncertain parameter objects for writing
            constraint objects. Indexed parameters are accepted, and
            are unpacked for constraint generation.
        """
        all_params = list()

        # expand all uncertain parameters to a list.
        # this accounts for the cases in which `uncertain_params`
        # consists of indexed model components,
        # or is itself a single indexed component
        if not isinstance(uncertain_params, (tuple, list)):
            uncertain_params = [uncertain_params]

        all_params = []
        for uparam in uncertain_params:
            all_params.extend(uparam.values())

        if len(all_params) != len(self.center):
            raise AttributeError(
                f"Center of ellipsoid is of dimension {len(self.center)},"
                f" but vector of uncertain parameters is of dimension"
                f" {len(all_params)}"
            )

        zip_all = zip(all_params, self.center, self.half_lengths)
        diffs_squared = list()

        # now construct the constraints
        conlist = ConstraintList()
        conlist.construct()
        for param, ctr, half_len in zip_all:
            if half_len > 0:
                diffs_squared.append((param - ctr) ** 2 / (half_len) ** 2)
            else:
                # equality constraints for parameters corresponding to
                # half-lengths of zero
                conlist.add(param == ctr)

        conlist.add(sum(diffs_squared) <= 1)

        return conlist


class EllipsoidalSet(UncertaintySet):
    """
    Ellipsoidal uncertainty set
    """

    def __init__(self, center, shape_matrix, scale=1):
        """
        EllipsoidalSet constructor.

        Parameters
        ----------
        center : (N,) array-like
            Center of the ellipsoid.
        shape_matrix : (N, N) array-like
            A positive definite matrix characterizing the shape
            and orientation of the ellipsoid.
        scale : numeric type, optional
            Square of the factor by which to scale the semi-axes
            of the ellipsoid (i.e. the eigenvectors of the shape
            matrix).
        """
        shape_matrix_arr = np.asarray(shape_matrix)
        center_arr = np.asarray(center)

        validate_dimensions("center", center_arr, 1)
        validate_dimensions("shape_matrix", shape_matrix_arr, 2)

        # check lengths match
        if shape_matrix_arr.shape[0] != shape_matrix_arr.shape[-1]:
            raise ValueError(
                "Argument `shape_matrix` should be a square matrix "
                f"(detected shape {shape_matrix_arr.shape})"
            )
        if center_arr.shape[0] != shape_matrix_arr.shape[0]:
            raise ValueError(
                f"Arguments `center` ({center_arr.shape[0]} entries) "
                "does not have as many entries as there are rows in "
                f"`shape_matrix` ({shape_matrix_arr.shape[0]} rows) "
            )

        # validate types
        for entry, row in zip(center, shape_matrix):
            validate_arg_type(
                "center",
                entry,
                valid_num_types,
                "a valid numeric type",
                True,
            )
            for coeff in row:
                validate_arg_type(
                    "shape_matrix",
                    coeff,
                    valid_num_types,
                    "a valid numeric type",
                    True,
                )
        validate_arg_type(
            "scale",
            scale,
            valid_num_types,
            "a valid numeric type",
            False,
        )

        # validate scale
        if scale < 0:
            raise ValueError(
                f"Argument `scale` (value {scale}) should be non-negative "
            )

        # ---------- CHECK SHAPE MATRIX POSITIVE DEFINITE
        # check symmetric
        if not np.allclose(shape_matrix_arr, shape_matrix_arr.T, atol=1e-8):
            raise ValueError("Shape matrix must be symmetric.")

        # check invertible (Exception raised)
        np.linalg.inv(shape_matrix)

        # check positive semi-definite.
        # since also invertible, means positive definite
        eigvals = np.linalg.eigvals(shape_matrix)
        if np.min(eigvals) < 0:
            raise ValueError(
                "Non positive-definite shape matrix "
                f"(detected eigenvalues {eigvals})"
            )

        # check roots of diagonal entries accessible
        # (should theoretically be true if positive definite)
        for idx in range(len(shape_matrix_arr)):
            diag_entry = shape_matrix[idx][idx]
            if np.isnan(np.power(diag_entry, 0.5)):
                raise ValueError(
                    "Cannot evaluate square root of the diagonal entry "
                    f"{diag_entry} of argument `shape_matrix`. "
                    "Check that this entry is nonnegative"
                )

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
        parameter_bounds = [(nom_value[i] - np.power(P[i][i] * scale, 0.5),
                             nom_value[i] + np.power(P[i][i] * scale, 0.5)) for i in range(self.dim)]
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
        Constructor for an N-dimensional DiscreteScenarioSet.

        Parameters
        ----------
        scenarios : (M, N) array-like
            A sequence of M distinct uncertain parameter realizations.
        """
        validate_dimensions("scenarios", scenarios, 2, display_value=True)
        for pt in scenarios:
            for val in pt:
                validate_arg_type(
                    "scenarios",
                    val,
                    valid_num_types,
                    "a valid numeric type",
                    True,
                )

        # Standardize to list of tuples
        self.scenarios = list(tuple(s) for s in scenarios)
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

    def __init__(self, **unc_sets):
        """
        IntersectionSet constructor

        Parameters
        ----------
        **uncertainty_sets : dict
            PyROS UncertaintySet objects of which to construct
            an intersection.
        """
        for idx, (kwd, unc_set) in enumerate(unc_sets.items()):
            validate_arg_type(
                kwd,
                unc_set,
                UncertaintySet,
                "an UncertaintySet object",
                False,
            )
            if idx == 0:
                set_dim = unc_set.dim
            else:
                if unc_set.dim != set_dim:
                    raise ValueError(
                        "UncertaintySet objects from which to construct "
                        "an intersection must be of the same dimension "
                        f" (detected dimensions {unc_set.dim} and {set_dim})"
                    )

        if len(unc_sets) < 2:
            raise ValueError(
                "IntersectionSet construction requires at least 2 "
                "UncertaintySet objects "
                f"(detected {len(uncertainty_sets)} objects)"
            )

        self.all_sets = list(unc_sets.values())
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
