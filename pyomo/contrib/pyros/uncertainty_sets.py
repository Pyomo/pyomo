#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This module defines the :class:`~UncertaintySet` abstract base class,
used for representing the uncertainty set of a robust
optimization problem to be solved with PyROS,
and a suite of pre-implemented concrete subclasses,
based on uncertainty sets often used in the robust optimization
literature.
"""

import abc
import math
import functools
from numbers import Integral
from collections import namedtuple
from collections.abc import Iterable, MutableSequence
from enum import Enum

from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
    Block,
    ConstraintList,
    ConcreteModel,
    maximize,
    minimize,
    Var,
    VarData,
    NonNegativeReals,
)
from pyomo.core.expr import mutable_expression, native_numeric_types, value
from pyomo.core.util import quicksum, dot_product
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import (
    copy_docstring,
    PARAM_IS_CERTAIN_REL_TOL,
    PARAM_IS_CERTAIN_ABS_TOL,
    POINT_IN_UNCERTAINTY_SET_TOL,
    standardize_component_data,
)
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.common.errors import InfeasibleConstraintException


def standardize_uncertain_param_vars(obj, dim):
    """
    Standardize an object castable to a list of VarData objects
    representing uncertain model parameters,
    and check that the length of the resulting list is equal
    to the specified dimension.

    Parameters
    ----------
    obj : Var, VarData, or iterable of Var/VarData
        Object to standardize.
    dim : int
        Specified dimension.

    Returns
    -------
    var_data_list : list of VarData
        Standard variable list.
    """
    var_data_list = standardize_component_data(
        obj=obj,
        valid_ctype=Var,
        valid_cdatatype=VarData,
        ctype_validator=None,
        cdatatype_validator=None,
        allow_repeats=False,
        from_iterable=obj,
    )
    if len(var_data_list) != dim:
        raise ValueError(
            f"Passed {len(var_data_list)} VarData objects representing "
            "the uncertain parameters, but the uncertainty set is of "
            f"dimension {dim}."
        )

    return var_data_list


def _setup_standard_uncertainty_set_constraint_block(
    block, uncertain_param_vars, dim, num_auxiliary_vars=None
):
    """
    Set up block to prepare for declaration of uncertainty
    set constraints.

    Parameters
    ----------
    block : BlockData or None
        Block to be prepared. If `None`, a new concrete block
        is instantiated.
    uncertain_param_vars : list of VarData or None
        Variables representing the main uncertain parameters.
        If `None`, then a new IndexedVar object consisting of
        `dim` members is declared on `block`.
    dim : int
        Dimension of the uncertainty set of interest.
    num_auxiliary_vars : int
        Number of variables representing auxiliary uncertain
        parameters to be declared.

    Returns
    -------
    block : BlockData
        Prepared block.
    param_var_data_list : list of VarData
        Variable data objects representing the main uncertain
        parameters.
    con_list : ConstraintList
        Empty ConstraintList, to which the uncertainty set constraints
        should be added later.
    auxiliary_var_list : list of VarData
        Variable data objects representing the auxiliary uncertain
        parameters.
    """
    if block is None:
        block = Block(concrete=True)

    if uncertain_param_vars is None:
        uncertain_param_indexed_var = Var(range(dim))
        block.add_component(
            unique_component_name(block, "uncertain_param_indexed_var"),
            uncertain_param_indexed_var,
        )
        param_var_data_list = list(uncertain_param_indexed_var.values())
    else:
        # resolve arguments
        param_var_data_list = standardize_uncertain_param_vars(
            uncertain_param_vars, dim=dim
        )
    con_list = ConstraintList()
    block.add_component(
        unique_component_name(block, "uncertainty_set_conlist"), con_list
    )

    auxiliary_var_list = []
    if num_auxiliary_vars is not None:
        auxiliary_param_var = Var(range(num_auxiliary_vars))
        block.add_component(
            unique_component_name(block, "auxiliary_param_var"), auxiliary_param_var
        )
        auxiliary_var_list = list(auxiliary_param_var.values())

    return block, param_var_data_list, con_list, auxiliary_var_list


UncertaintyQuantification = namedtuple(
    "UncertaintyQuantification",
    ("block", "uncertainty_cons", "uncertain_param_vars", "auxiliary_vars"),
)
UncertaintyQuantification.__doc__ = """
    A collection of modeling components
    generated or addressed by the `set_as_constraint` method of
    an uncertainty set object.

    The UncertaintyQuantification class was generated using
    the Python :py:func:`~collections.namedtuple` factory function,
    so the standard :py:func:`~collections.namedtuple`
    attributes and methods
    (e.g., :py:meth:`~collections.somenamedtuple._asdict`)
    are available.

    Parameters
    ----------
    block : BlockData
        Block on which the uncertainty set constraints
        were added.
    uncertainty_cons : list of ConstraintData
        The added uncertainty set constraints.
    uncertain_param_vars : list of VarData
        Variables representing the (main) uncertain parameters.
    auxiliary_vars : list of VarData
        Variables representing the auxiliary uncertain parameters.
"""
UncertaintyQuantification.block.__doc__ = (
    "Block on which the uncertainty set constraints were added."
)
UncertaintyQuantification.uncertainty_cons.__doc__ = (
    "The added uncertainty set constraints."
)
UncertaintyQuantification.uncertain_param_vars.__doc__ = (
    "Variables representing the (main) uncertain parameters."
)
UncertaintyQuantification.auxiliary_vars.__doc__ = (
    "Variables representing the auxiliary uncertain parameters."
)


def validate_arg_type(
    arg_name,
    arg_val,
    valid_types,
    valid_type_desc=None,
    is_entry_of_arg=False,
    check_numeric_type_finite=True,
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
    valid_types : type or iterable of types
        Valid types for the argument value.
    valid_type_desc : str or None, optional
        Description of valid types for the argument value;
        this description is included in the exception message.
    is_entry_of_arg : bool, optional
        Is the argument value passed an entry of the argument
        described by `arg_name` (such as entry of an array or list).
        This will be indicated in the exception message.
        The default is `False`.
    check_numeric_type_finite : bool, optional
        If the valid types comprise a sequence of numeric types,
        check that the argument value is finite (and also not NaN),
        as well. The default is `True`.

    Raises
    ------
    TypeError
        If the argument value is not a valid type.
    ValueError
        If the finiteness check on a numerical value returns
        a negative result.
    """
    # convert to tuple if necessary
    if isinstance(valid_types, Iterable):
        valid_types = tuple(valid_types)
    if not isinstance(arg_val, valid_types):
        if valid_type_desc is not None:
            type_phrase = f"not {valid_type_desc}"
        else:
            if not isinstance(valid_types, Iterable):
                valid_types = [valid_types]
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

    # check for finiteness, if desired
    if check_numeric_type_finite:
        if isinstance(valid_types, type):
            numeric_types_required = valid_types in native_numeric_types
        else:
            numeric_types_required = set(valid_types).issubset(native_numeric_types)
        if numeric_types_required and (math.isinf(arg_val) or math.isnan(arg_val)):
            if is_entry_of_arg:
                raise ValueError(
                    f"Entry '{arg_val}' of the argument `{arg_name}` "
                    f"is not a finite numeric value"
                )
            else:
                raise ValueError(
                    f"Argument `{arg_name}` is not a finite numeric value "
                    f"(provided value '{arg_val}')"
                )


def is_ragged(arr, arr_types=None):
    """
    Return True if the array-like `arr` is ragged, False otherwise.

    NOTE: if Numpy ndarrays are considered to be arr types,
    then zero-dimensional arrays are not considered to be as such.

    Parameters
    ----------
    arr : array_like
        Array to check.
    arr_types : None or iterable of type
        Types of entries of `arr` to be considered subarrays.
        If `None` is specified, then this is set to
        ``(list, numpy.ndarray, tuple)``.

    Returns
    -------
    bool
        True if ragged, False otherwise.
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

    entries_ragged = [is_ragged(entry) for entry in arr]
    if any(entries_ragged):
        return True
    else:
        return any(np.array(arr[0]).shape != np.array(entry).shape for entry in arr)


def validate_dimensions(arr_name, arr, dim, display_value=False):
    """
    Validate dimension of an array-like object.

    Parameters
    ----------
    arr_name : str
        Name of the array to validate.
    arr : array_like
        Array to validate.
    dim : int
        Required dimension of the array.
    display_value : bool, optional
        True to include the array string representation
        in exception messages, False otherwise.

    Raises
    ------
    ValueError
        If `arr` is ragged or not of the required dimension `dim`.
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


def validate_array(
    arr,
    arr_name,
    dim,
    valid_types,
    valid_type_desc=None,
    required_shape=None,
    required_shape_qual="",
):
    """
    Validate shape and entry types of an array-like object.

    Parameters
    ----------
    arr : array_like
        Object to validate.
    arr_name : str
        A name/descriptor of the object to validate.
        Usually, this is the name of an object attribute
        to which the array is meant to be set.
    dim : int
        Required dimension of the array-like object.
    valid_types : set[type]
        Allowable type(s) for each entry of the array.
    valid_type_desc : str or None, optional
        Descriptor for the allowable types.
    required_shape : list or None, optional
        Specification of the length of the array in each dimension.
        If `None` is provided, no specifications are imposed.
        If a `list` is provided, then each entry of the list must be
        an `int` specifying the required length in the dimension
        corresponding to the position of the entry
        or `None` (meaning no requirement for the length in the
        corresponding dimension).
    required_shape_qual : str, optional
        Clause/phrase expressing reason `arr` should be of shape
        `required_shape`, e.g. "to match the set dimension".

    Raises
    ------
    ValueError
        If the Numpy array to which `arr` is cast is not of shape
        `required_shape`.
    """
    np_arr = np.array(arr, dtype=object)
    validate_dimensions(arr_name, np_arr, dim, display_value=False)

    def generate_shape_str(shape, required_shape):
        shape_str = ""
        assert len(shape) == len(required_shape)
        for idx, (sval, rsval) in enumerate(zip(shape, required_shape)):
            if rsval is None:
                shape_str += "..."
            else:
                shape_str += f"{sval}"
            if idx < len(shape) - 1:
                shape_str += ","
        return "(" + shape_str + ")"

    # validate shape requirements
    if required_shape is not None:
        assert len(required_shape) == dim
        for idx, size in enumerate(required_shape):
            if size is not None and size != np_arr.shape[idx]:
                req_shape_str = generate_shape_str(required_shape, required_shape)
                actual_shape_str = generate_shape_str(np_arr.shape, required_shape)
                required_shape_qual = (
                    # add a preceding space, if needed
                    f" {required_shape_qual}"
                    if required_shape_qual
                    else ""
                )
                raise ValueError(
                    f"Attribute '{arr_name}' should be of shape "
                    f"{req_shape_str}{required_shape_qual}, but detected shape "
                    f"{actual_shape_str}"
                )

    for val in np_arr.flat:
        validate_arg_type(
            arr_name,
            val,
            valid_types,
            valid_type_desc=valid_type_desc,
            is_entry_of_arg=True,
        )


class Geometry(Enum):
    """
    Geometry classifications for PyROS uncertainty set objects.
    """

    LINEAR = 1
    CONVEX_NONLINEAR = 2
    GENERAL_NONLINEAR = 3
    DISCRETE_SCENARIOS = 4


class UncertaintySet(object, metaclass=abc.ABCMeta):
    """
    An object representing an uncertainty set to be passed to the
    PyROS solver.

    An `UncertaintySet` object should be viewed as merely a container
    for data needed to parameterize the set it represents,
    such that the object's attributes do not reference the
    components of a Pyomo modeling object.
    """

    # True if parameter_bounds attribute returns
    # exact bounding box, False otherwise
    _PARAMETER_BOUNDS_EXACT = False

    @property
    @abc.abstractmethod
    def dim(self):
        """
        Dimension of the uncertainty set (number of uncertain
        parameters in a corresponding optimization model of interest).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def geometry(self):
        """
        Geometry of the uncertainty set. See the `Geometry` class
        documentation.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameter_bounds(self):
        """
        Bounds for the value of each uncertain parameter constrained
        by the set (i.e. bounds for each set dimension).

        Returns
        -------
        : list of tuple
            If the bounds can be calculated, then the list is of
            length `N`, and each entry is a pair of numeric
            (lower, upper) bounds for the corresponding
            (Cartesian) coordinate. Otherwise, the list is empty.
        """
        raise NotImplementedError

    def _create_bounding_model(self):
        """
        Make uncertain parameter value bounding problems (optimize
        value of each uncertain parameter subject to constraints on the
        uncertain parameters).

        Returns
        -------
        model : ConcreteModel
            Bounding model, with an indexed mimimization sense
            Objective with name 'param_var_objectives' consisting
            of `N` entries, all of which have been deactivated.
        """
        model = ConcreteModel()

        # construct param vars, initialize to nominal point
        model.param_vars = Var(range(self.dim))

        # add constraints
        self.set_as_constraint(uncertain_params=model.param_vars, block=model)

        @model.Objective(range(self.dim))
        def param_var_objectives(self, idx):
            return model.param_vars[idx]

        # deactivate all objectives
        model.param_var_objectives.deactivate()

        return model

    def is_bounded(self, config):
        """
        Determine whether the uncertainty set is bounded.

        Parameters
        ----------
        config : ConfigDict
            PyROS solver configuration.

        Returns
        -------
        : bool
            True if the uncertainty set is certified to be bounded,
            and False otherwise.

        Notes
        -----
        This check is carried out by checking if all parameter bounds
        are finite.

        If no parameter bounds are available, the following processes
        are run to perform the check:
        (i) feasibility-based bounds tightening is used to obtain
        parameter bounds, and if not all bound are found,
        (ii) solving a sequence of maximization and minimization
        problems (in which the objective for each problem is the value
        of a single uncertain parameter).
        If any of the optimization models cannot be solved successfully
        to optimality, then False is returned.

        This method is invoked by ``self.validate()``.
        """
        # use parameter bounds if they are available
        param_bounds_arr = self.parameter_bounds
        if param_bounds_arr:
            all_bounds_finite = np.all(np.isfinite(param_bounds_arr))
        else:
            # use FBBT
            param_bounds_arr = np.array(
                self._fbbt_parameter_bounds(config), dtype="float"
            )
            all_bounds_finite = np.isfinite(param_bounds_arr).all()

            if not all_bounds_finite:
                # get bounds that need to be solved
                index = np.isnan(param_bounds_arr)
                # solve bounding problems for bounds that have not been found
                opt_bounds_arr = np.array(
                    self._compute_exact_parameter_bounds(
                        solver=config.global_solver, index=index
                    ),
                    dtype="float",
                )
                # combine with previously found bounds
                param_bounds_arr[index] = opt_bounds_arr[index]
                all_bounds_finite = np.isfinite(param_bounds_arr).all()

        # log result
        if not all_bounds_finite:
            config.progress_logger.error(
                "Computed coordinate value bounds are not all finite. "
                f"Got bounds: {param_bounds_arr}"
            )

        return all_bounds_finite

    def is_nonempty(self, config):
        """
        Determine whether the uncertainty set is nonempty.

        Parameters
        ----------
        config : ConfigDict
            PyROS solver configuration.

        Returns
        -------
        : bool
            True if the uncertainty set is nonempty,
            and False otherwise.
        """
        # check if nominal point is in set for quick test
        if config.nominal_uncertain_param_vals:
            set_nonempty = self.point_in_set(config.nominal_uncertain_param_vals)
        else:
            # construct feasibility problem and solve otherwise
            self._solve_feasibility(config.global_solver)
            set_nonempty = True

        # log result
        if not set_nonempty:
            config.progress_logger.error(
                "Nominal point is not within the uncertainty set. "
                f"Got nominal point: {config.nominal_uncertain_param_vals}"
            )

        return set_nonempty

    def validate(self, config):
        """
        Validate the uncertainty set with a nonemptiness
        and boundedness check.

        Parameters
        ----------
        config : ConfigDict
            PyROS solver configuration.

        Raises
        ------
        ValueError
            If nonemptiness check or boundedness check fails.
        """
        if not self.is_nonempty(config=config):
            raise ValueError(f"Nonemptiness check failed for uncertainty set {self}.")

        if not self.is_bounded(config=config):
            raise ValueError(f"Boundedness check failed for uncertainty set {self}.")

    @abc.abstractmethod
    def set_as_constraint(self, uncertain_params=None, block=None):
        """
        Construct a block of Pyomo constraint(s) defining the
        uncertainty set on variables representing the uncertain
        parameters, for use in a two-stage robust optimization
        problem or subproblem (such as a PyROS separation subproblem).

        Parameters
        ----------
        uncertain_params : None, Var, or list of Var, optional
            Variable objects representing the (main) uncertain
            parameters. If `None` is passed, then
            new variable objects are constructed.
        block : BlockData or None, optional
            Block on which to declare the constraints and any
            new variable objects. If `None` is passed, then a new
            block is constructed.

        Returns
        -------
        UncertaintyQuantification
            A collection of the components added or addressed.
        """
        pass

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the uncertainty set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        is_in_set : bool
            True if the point lies in the uncertainty set,
            False otherwise.

        Notes
        -----
        This method is invoked at the outset of a PyROS solver call to
        determine whether a user-specified nominal parameter realization
        lies in the uncertainty set.
        """
        validate_array(
            arr=point,
            arr_name="point",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="numeric type",
            required_shape=[self.dim],
            required_shape_qual="to match the set dimension",
        )

        m = ConcreteModel()
        uncertainty_quantification = self.set_as_constraint(block=m)
        for var, val in zip(uncertainty_quantification.uncertain_param_vars, point):
            var.set_value(val)

        # since constraint expressions are relational,
        # `value()` returns True if constraint satisfied, False else
        # NOTE: this check may be inaccurate if there are auxiliary
        #       variables and they have not been initialized to
        #       feasible values
        is_in_set = all(
            value(con.expr) for con in uncertainty_quantification.uncertainty_cons
        )

        return is_in_set

    def _compute_exact_parameter_bounds(self, solver, index=None):
        """
        Compute lower and upper coordinate value bounds
        for every dimension of `self` by solving a bounding model.

        Parameters
        ----------
        solver : Pyomo solver type
            Optimizer to invoke on the bounding problems.
        index : list of 2-tuple of bool, optional
            A list of tuples for each index of the coordinates for
            which to compute bounds. A lower or upper bound is
            computed for any value that is True, while False
            indicates that the bound should be skipped.
            If None is passed, then the argument is set to
            ``[(True, True)]*self.dim``, so that the bounds
            for all coordinates are computed.

        Returns
        -------
        param_bounds : list of tuple of float
            Each entry of the list is a 2-tuple
            containing the lower and upper bound for
            the corresponding dimension.

        Raises
        ------
        ValueError
            If solver failed to compute a bound for a
            coordinate.
        """
        if index is None:
            index = [(True, True)] * self.dim

        # create bounding model and get all objectives
        bounding_model = self._create_bounding_model()
        objs_to_optimize = bounding_model.param_var_objectives.items()

        param_bounds = []
        for idx, obj in objs_to_optimize:
            # activate objective for corresponding dimension
            obj.activate()
            bounds = []

            # solve for lower bound, then upper bound
            # solve should be successful
            for i, sense in enumerate((minimize, maximize)):
                # check if the LB or UB should be solved
                if not index[idx][i]:
                    bounds.append(None)
                    continue
                obj.sense = sense
                res = solver.solve(bounding_model, load_solutions=False)
                if check_optimal_termination(res):
                    bounding_model.solutions.load_from(res)
                else:
                    raise ValueError(
                        "Could not compute "
                        f"{'lower' if sense == minimize else 'upper'} "
                        f"bound in dimension {idx + 1} of {self.dim}. "
                        f"Solver status summary:\n {res.solver}."
                    )
                bounds.append(value(obj))

            # add parameter bounds for current dimension
            param_bounds.append(tuple(bounds))

            # ensure sense is minimize when done, deactivate
            obj.sense = minimize
            obj.deactivate()

        return param_bounds

    def _fbbt_parameter_bounds(self, config):
        """
        Obtain parameter bounds of the uncertainty set using FBBT.
        The bounds returned by FBBT may be inexact.

        Parameters
        ----------
        config : ConfigDict
            PyROS solver configuration.

        Returns
        -------
        param_bounds : list of tuple
            List, of length `N`, containing
            (lower bound, upper bound) pairs
            for the uncertain parameters.
        """
        bounding_model = self._create_bounding_model()

        # calculate bounds with FBBT
        try:
            fbbt(bounding_model)
        except InfeasibleConstraintException as fbbt_infeasible_con_exception:
            config.progress_logger.error(
                "Encountered the following exception "
                f"while computing parameter bounds with FBBT "
                f"for uncertainty set {self}:\n  "
                f"{fbbt_infeasible_con_exception!r}"
            )

        param_bounds = [
            (var.lower, var.upper) for var in bounding_model.param_vars.values()
        ]

        return param_bounds

    def _solve_feasibility(self, solver):
        """
        Construct and solve feasibility problem using uncertainty set
        constraints and parameter bounds using `set_as_constraint` and
        `_add_bounds_on_uncertain_parameters` of self.

        Parameters
        ----------
        solver : Pyomo solver
            Optimizer capable of solving bounding problems to
            global optimality.

        Raises
        ------
        ValueError
            If feasibility problem fails to solve.
        """
        model = ConcreteModel()
        model.u = Var(within=NonNegativeReals)

        # construct param vars
        model.param_vars = Var(range(self.dim))

        # add bounds on param vars
        self._add_bounds_on_uncertain_parameters(model.param_vars, global_solver=solver)

        # add constraints
        self.set_as_constraint(uncertain_params=model.param_vars, block=model)

        # add objective with dummy variable model.u
        @model.Objective(sense=minimize)
        def feasibility_objective(self):
            return model.u

        # solve feasibility problem
        res = solver.solve(model, load_solutions=False)
        if not check_optimal_termination(res):
            raise ValueError(
                "Could not successfully solve feasibility problem. "
                f"Solver status summary:\n {res.solver}."
            )

    def _add_bounds_on_uncertain_parameters(
        self, uncertain_param_vars, global_solver=None
    ):
        """
        Specify declared bounds for Vars representing the uncertain
        parameters constrained to an uncertainty set.

        Parameters
        ----------
        global_solver : None or Pyomo solver, optional
            Optimizer capable of solving bounding problems to
            global optimality. If the coordinate bounds for the
            set can be retrieved through `self.parameter_bounds`,
            then None can be passed.
        uncertain_param_vars : Var, VarData, or list of Var/VarData
            Variables representing the uncertain parameter objects.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """
        uncertain_param_vars = standardize_uncertain_param_vars(
            uncertain_param_vars, self.dim
        )

        parameter_bounds = self.parameter_bounds
        if not parameter_bounds:
            parameter_bounds = self._compute_exact_parameter_bounds(global_solver)

        for (lb, ub), param_var in zip(parameter_bounds, uncertain_param_vars):
            param_var.setlb(lb)
            param_var.setub(ub)

    def compute_auxiliary_uncertain_param_vals(self, point, solver=None):
        """
        Compute auxiliary uncertain parameter values for a given point.
        The point need not be in the uncertainty set.

        Parameters
        ----------
        point : (N,) array-like
            Point of interest.
        solver : Pyomo solver, optional
            If needed, a Pyomo solver with which to compute the
            auxiliary values.

        Returns
        -------
        aux_space_pt : numpy.ndarray
            Computed auxiliary uncertain parameter values.
        """
        raise NotImplementedError(
            f"Auxiliary parameter computation not supported for {type(self).__name__}."
        )

    def _is_coordinate_fixed(self, config, index=None):
        """
        Test whether each Cartesian coordinate of interest
        of the uncertainty set is constrained to a single value.

        Parameters
        ----------
        config : ConfigDict
            PyROS solver options. Should at least contain attribute
            `global_solver`.
        index : iterable of int, optional
            Positional indices of the coordinates to check.
            If `None` is passed, then `index` is set to
            ``list(range(self.dim))``, so that all coordinates
            are checked.

        Returns
        -------
        list of bool
            Same length as ``index``.
            An entry of the list is True if the corresponding
            coordinate is constrained to a single value,
            False otherwise.
        """

        def _values_close(a, b):
            return math.isclose(
                a, b, rel_tol=PARAM_IS_CERTAIN_ABS_TOL, abs_tol=PARAM_IS_CERTAIN_REL_TOL
            )

        param_bounds = self.parameter_bounds
        if not (param_bounds and self._PARAMETER_BOUNDS_EXACT):
            # we need the exact bounding box
            param_bounds = self._compute_exact_parameter_bounds(
                solver=config.global_solver, index=index
            )
        else:
            index = list(range(len(param_bounds))) if index is None else index
            param_bounds = [param_bounds[idx] for idx in index]
        return [_values_close(lb, ub) for lb, ub in param_bounds]


class UncertaintySetList(MutableSequence):
    """
    Wrapper around a list of uncertainty sets, all of which have
    an immutable common dimension.

    Parameters
    ----------
    uncertainty_sets : iterable, optional
        Sequence of uncertainty sets.
    name : str or None, optional
        Name of the uncertainty set list.
    min_length : int or None, optional
        Minimum required length of the sequence. If `None` is
        provided, then the minimum required length is set to 0.
    """

    def __init__(self, uncertainty_sets=[], name=None, min_length=None):
        """Initialize self (see class docstring)."""
        self._name = name
        self._min_length = 0 if min_length is None else min_length

        # check minimum length requirement satisfied
        initlist = list(uncertainty_sets)
        if len(initlist) < self._min_length:
            raise ValueError(
                f"Attempting to initialize uncertainty set list "
                f"{self._name!r} "
                f"of minimum required length {self._min_length} with an "
                f"iterable of length {len(initlist)}"
            )

        # validate first entry of initial list.
        # The common dimension is set to that of the first entry
        # if validation is successful
        self._dim = None
        if initlist:
            self._validate(initlist[0])

        # now initialize the list
        self._list = []
        self.extend(initlist)

    def __len__(self):
        """Length of the list contained in self."""
        return len(self._list)

    def __repr__(self):
        """Return repr(self)."""
        return f"{self.__class__.__name__}({repr(self._list)})"

    def __getitem__(self, idx):
        """Return self[idx]."""
        return self._list[idx]

    def __setitem__(self, idx, value):
        """Set self[idx] = value."""
        if self._index_is_valid(idx):
            # perform validation and length check only if
            # index is valid, so that exceptions due to
            # index referencing (wrong type, out of range)
            # are raised in update attempt
            self._validate(value)
            self._check_length_update(idx, value)

        self._list[idx] = value

    def __delitem__(self, idx):
        """Perform del self[idx]."""
        if self._index_is_valid(idx):
            self._check_length_update(idx, [])
        del self._list[idx]

    def clear(self):
        """Remove all items from the list."""
        self._check_length_update(slice(0, len(self)), [])
        self._list.clear()

    def insert(self, idx, value):
        """Insert an object before index denoted by idx."""
        if isinstance(idx, Integral):
            # index should be valid. Validate value before
            # inserting
            self._validate(value, single_item=True)
        self._list.insert(idx, value)

    def _index_is_valid(self, idx, allow_int_only=False):
        """
        Object to be used as list index is within range of
        list contained within self.

        Parameters
        ----------
        idx : object
            List index. Usually an integer type or slice.
        allow_int_only : bool, optional
            Being an integral type is a necessary condition
            for validity. The default is True.

        Returns
        -------
        : bool
            True if index is valid, False otherwise.
        """
        try:
            self._list[idx]
        except (TypeError, IndexError):
            slice_valid = False
        else:
            slice_valid = True

        # if only integer types allowed, then must be an integer type
        int_req_satisfied = not allow_int_only or isinstance(idx, Integral)

        return slice_valid and int_req_satisfied

    def _check_length_update(self, idx, value):
        """
        Check whether the update ``self[idx] = value`` reduces the
        length of self to a value smaller than the minimum length.

        Raises
        ------
        ValueError
            If minimum length requirement is violated by the update.
        """
        if isinstance(idx, Integral):
            slice_len = 1
        else:
            slice_len = len(self._list[idx])

        val_len = len(value) if isinstance(value, Iterable) else 1
        new_len = len(self) + val_len - slice_len
        if new_len < self._min_length:
            raise ValueError(
                f"Length of uncertainty set list {self._name!r} must "
                f"be at least {self._min_length}"
            )

    def _validate(self, value, single_item=False):
        """
        Validate item or sequence of items to be inserted into self.

        Parameters
        ----------
        value : object
            Object to validate.
        single_item : bool, optional
            Do not allow validation of iterables of objects
            (e.g. a list of ``UncertaintySet`` objects).
            The default is `False`.

        Raises
        ------
        TypeError
            If object passed is not of the appropriate type
            (``UncertaintySet``, or an iterable thereof).
        ValueError
            If object passed is (or contains) an ``UncertaintySet``
            whose dimension does not match that of other uncertainty
            sets in self.
        """
        if not single_item and isinstance(value, Iterable):
            for val in value:
                self._validate(val, single_item=True)
        else:
            validate_arg_type(
                self._name,
                value,
                UncertaintySet,
                "An `UncertaintySet` object",
                is_entry_of_arg=True,
            )
            if self._dim is None:
                # common dimension is now set
                self._dim = value.dim
            else:
                # ensure set added matches common dimension
                if value.dim != self._dim:
                    raise ValueError(
                        f"Uncertainty set list with name {self._name!r} "
                        f"contains UncertaintySet objects of dimension "
                        f"{self._dim}, but attempting to add set of dimension "
                        f"{value.dim}"
                    )

    @property
    def dim(self):
        """Dimension of all uncertainty sets contained in self."""
        return self._dim


class BoxSet(UncertaintySet):
    """
    A hyper-rectangle (i.e., "box").

    Parameters
    ----------
    bounds : (N, 2) array_like
        Lower and upper bounds for each dimension of the set.

    Examples
    --------
    1D box set (interval):

    >>> from pyomo.contrib.pyros import BoxSet
    >>> interval = BoxSet(bounds=[(1, 2)])
    >>> interval.bounds
    array([[1, 2]])

    2D box set:

    >>> box_set = BoxSet(bounds=[[1, 2], [3, 4]])
    >>> box_set.bounds
    array([[1, 2],
           [3, 4]])

    5D hypercube with bounds 0 and 1 in each dimension:

    >>> hypercube_5d = BoxSet(bounds=[[0, 1] for idx in range(5)])
    >>> hypercube_5d.bounds
    array([[0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 1]])
    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, bounds):
        """Initialize self (see class docstring)."""
        self.bounds = bounds

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "box"

    @property
    def bounds(self):
        """
        (N, 2) numpy.ndarray : Lower and upper bounds for each dimension
        of the set.

        The bounds of a `BoxSet` instance can be changed, such that
        the dimension of the set remains unchanged.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, val):
        validate_array(
            arr=val,
            arr_name="bounds",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=[None, 2],
        )

        bounds_arr = np.array(val)

        # box set dimension is immutable
        if hasattr(self, "_bounds") and bounds_arr.shape[0] != self.dim:
            raise ValueError(
                "Attempting to set bounds of a box set of dimension "
                f"{self.dim} to a value of dimension {bounds_arr.shape[0]}"
            )
        self._bounds = np.array(val)

    @property
    def dim(self):
        """
        int : Dimension `N` of the box set.
        """
        return len(self.bounds)

    @property
    def geometry(self):
        """
        Geometry of the box set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the box set.
        This is numerically equivalent to the `bounds` attribute.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        return [tuple(bound) for bound in self.bounds]

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )

        vardata_bound_zip = zip(param_var_list, self.bounds)
        for idx, (param_var, (lb, ub)) in enumerate(vardata_bound_zip):
            uncertainty_conlist.add((lb, param_var, ub))

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_list,
            uncertainty_cons=list(uncertainty_conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    def validate(self, config):
        """
        Check BoxSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            or ``self.parameter_bounds`` has LB > UB.)
        """
        bounds_arr = np.array(self.parameter_bounds)

        # check bounds are valid
        # this includes a finiteness check
        validate_array(
            arr=bounds_arr,
            arr_name="bounds",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=[None, 2],
        )

        # check LB <= UB
        for lb, ub in bounds_arr:
            if lb > ub:
                raise ValueError(f"Lower bound {lb} exceeds upper bound {ub}")


class CardinalitySet(UncertaintySet):
    """
    A cardinality-constrained (i.e., "gamma") set.

    Parameters
    ----------
    origin : (N,) array_like
        Origin of the set (e.g., nominal uncertain parameter values).
    positive_deviation : (N,) array_like
        Maximal non-negative coordinate deviation from the origin
        in each dimension.
    gamma : numeric type
        Upper bound for the number of uncertain parameters which
        may realize their maximal deviations from the origin
        simultaneously.

    Examples
    --------
    A 3D cardinality set:

    >>> from pyomo.contrib.pyros import CardinalitySet
    >>> gamma_set = CardinalitySet(
    ...     origin=[0, 0, 0],
    ...     positive_deviation=[1.0, 2.0, 1.5],
    ...     gamma=1,
    ... )
    >>> gamma_set.origin
    array([0, 0, 0])
    >>> gamma_set.positive_deviation
    array([1. , 2. , 1.5])
    >>> gamma_set.gamma
    1
    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, origin, positive_deviation, gamma):
        """Initialize self (see class docstring)."""
        self.origin = origin
        self.positive_deviation = positive_deviation
        self.gamma = gamma

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "cardinality"

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Origin of the cardinality set
        (e.g. nominal parameter values).
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(
            arr=val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )

        # dimension of the set is immutable
        val_arr = np.array(val)
        if hasattr(self, "_origin"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'origin' of cardinality "
                    f"set of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._origin = val_arr

    @property
    def positive_deviation(self):
        """
        (N,) numpy.ndarray : Maximal coordinate deviations from the
        origin in each dimension. All entries are nonnegative.
        """
        return self._positive_deviation

    @positive_deviation.setter
    def positive_deviation(self, val):
        validate_array(
            arr=val,
            arr_name="positive_deviation",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )

        val_arr = np.array(val)

        # dimension of the set is immutable
        if hasattr(self, "_origin"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'positive_deviation' of "
                    f"cardinality set of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._positive_deviation = val_arr

    @property
    def gamma(self):
        """
        numeric type : Upper bound for the number of uncertain
        parameters which may maximally deviate from their respective
        origin values simultaneously. Must be a numerical value ranging
        from 0 to the set dimension `N`.

        Note that, mathematically, setting `gamma` to 0 reduces the set
        to a singleton containing the center, while setting `gamma` to
        the set dimension `N` makes the set mathematically equivalent
        to a `BoxSet` with bounds
        ``numpy.array([origin, origin + positive_deviation]).T``.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        validate_arg_type(
            "gamma", val, native_numeric_types, "a valid numeric type", False
        )

        self._gamma = val

    @property
    def dim(self):
        """
        int : Dimension `N` of the cardinality set.
        """
        return len(self.origin)

    @property
    def geometry(self):
        """
        Geometry of the cardinality set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the cardinality set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        nom_val = self.origin
        deviation = self.positive_deviation
        gamma = self.gamma
        parameter_bounds = [
            (nom_val[i], nom_val[i] + min(gamma, 1) * deviation[i])
            for i in range(len(nom_val))
        ]
        return parameter_bounds

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        # resolve arguments
        block, param_var_data_list, conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=self.dim,
            )
        )

        cardinality_zip = zip(
            self.origin, self.positive_deviation, aux_var_list, param_var_data_list
        )
        for orig_val, pos_dev, auxvar, param_var in cardinality_zip:
            conlist.add(orig_val + pos_dev * auxvar == param_var)

        conlist.add(quicksum(aux_var_list) <= self.gamma)

        for aux_var in aux_var_list:
            aux_var.setlb(0)
            aux_var.setub(1)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=list(conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    @copy_docstring(UncertaintySet.compute_auxiliary_uncertain_param_vals)
    def compute_auxiliary_uncertain_param_vals(self, point, solver=None):
        validate_array(
            arr=point,
            arr_name="point",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="numeric type",
            required_shape=[self.dim],
            required_shape_qual="to match the set dimension",
        )
        point_arr = np.array(point)

        is_dev_nonzero = self.positive_deviation != 0
        aux_space_pt = np.empty(self.dim)
        aux_space_pt[is_dev_nonzero] = (
            point_arr[is_dev_nonzero] - self.origin[is_dev_nonzero]
        ) / self.positive_deviation[is_dev_nonzero]
        aux_space_pt[self.positive_deviation == 0] = 0

        return aux_space_pt

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the cardinality set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        aux_space_pt = self.compute_auxiliary_uncertain_param_vals(point)
        return (
            np.all(point == self.origin + self.positive_deviation * aux_space_pt)
            and aux_space_pt.sum() <= self.gamma
            and np.all(0 <= aux_space_pt)
            and np.all(aux_space_pt <= 1)
        )

    def validate(self, config):
        """
        Check CardinalitySet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            ``self.positive_deviation`` has negative values,
            or ``self.gamma`` is out of range).
        """
        orig_val = self.origin
        pos_dev = self.positive_deviation
        gamma = self.gamma

        # check origin, positive deviation, and gamma are valid
        # this includes a finiteness check
        validate_array(
            arr=orig_val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )
        validate_array(
            arr=pos_dev,
            arr_name="positive_deviation",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )
        validate_arg_type(
            "gamma", gamma, native_numeric_types, "a valid numeric type", False
        )

        # check deviation is positive
        for dev_val in pos_dev:
            if dev_val < 0:
                raise ValueError(
                    f"Entry {dev_val} of attribute 'positive_deviation' "
                    f"is negative value"
                )

        # check gamma between 0 and n
        if gamma < 0 or gamma > self.dim:
            raise ValueError(
                "Cardinality set attribute "
                f"'gamma' must be a real number between 0 and dimension "
                f"{self.dim} "
                f"(provided value {gamma})"
            )


class PolyhedralSet(UncertaintySet):
    """
    A bounded convex polyhedron or polytope.

    Parameters
    ----------
    lhs_coefficients_mat : (M, N) array_like
        Left-hand side coefficients for the linear
        inequality constraints defining the polyhedral set.
    rhs_vec : (M,) array_like
        Right-hand side values for the linear inequality
        constraints defining the polyhedral set.
        Each entry is an upper bound for the quantity
        ``lhs_coefficients_mat @ x``, where `x` is an (N,)
        array representing any point in the polyhedral set.

    Examples
    --------
    2D polyhedral set with 4 defining inequalities:

    >>> from pyomo.contrib.pyros import PolyhedralSet
    >>> pset = PolyhedralSet(
    ...     lhs_coefficients_mat=[[-1, 0], [0, -1], [-1, 1], [1, 0]],
    ...     rhs_vec=[0, 0, 0, 1],
    ... )
    >>> pset.coefficients_mat
    array([[-1,  0],
           [ 0, -1],
           [-1,  1],
           [ 1,  0]])
    >>> pset.rhs_vec
    array([0, 0, 0, 1])
    """

    def __init__(self, lhs_coefficients_mat, rhs_vec):
        """Initialize self (see class docstring)."""
        # set attributes to copies of the originals
        self.coefficients_mat = lhs_coefficients_mat
        self.rhs_vec = rhs_vec

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "polyhedral"

    @property
    def coefficients_mat(self):
        """
        (M, N) numpy.ndarray : Coefficient matrix for the (linear)
        inequality constraints defining the polyhedral set.

        In tandem with the `rhs_vec` attribute, this matrix should
        be such that the polyhedral set is nonempty and bounded.
        Such a check is performed only at instance construction.
        """
        return self._coefficients_mat

    @coefficients_mat.setter
    def coefficients_mat(self, val):
        validate_array(
            arr=val,
            arr_name="coefficients_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        lhs_coeffs_arr = np.array(val)

        # check no change in set dimension
        if hasattr(self, "_coefficients_mat"):
            if lhs_coeffs_arr.shape[1] != self.dim:
                raise ValueError(
                    f"Polyhedral set attribute 'coefficients_mat' must have "
                    f"{self.dim} columns to match set dimension "
                    f"(provided matrix with {lhs_coeffs_arr.shape[1]} columns)"
                )

        # check shape match with rhs vector
        if hasattr(self, "_rhs_vec"):
            if lhs_coeffs_arr.shape[0] != self.rhs_vec.size:
                raise ValueError(
                    "PolyhedralSet attribute 'coefficients_mat' "
                    f"must have {self.rhs_vec.size} rows "
                    f"to match shape of attribute 'rhs_vec' "
                    f"(provided {lhs_coeffs_arr.shape[0]} rows)"
                )
        self._coefficients_mat = lhs_coeffs_arr

    @property
    def rhs_vec(self):
        """
        (M,) numpy.ndarray : Right-hand side values (upper bounds) for
        the (linear) inequality constraints defining the polyhedral set.
        """
        return self._rhs_vec

    @rhs_vec.setter
    def rhs_vec(self, val):
        validate_array(
            arr=val,
            arr_name="rhs_vec",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        rhs_vec_arr = np.array(val)

        # ensure shape of coefficients matrix
        # and rhs vec match
        if hasattr(self, "_coefficients_mat"):
            if len(val) != self.coefficients_mat.shape[0]:
                raise ValueError(
                    "PolyhedralSet attribute 'rhs_vec' "
                    f"must have {self.coefficients_mat.shape[0]} entries "
                    f"to match shape of attribute 'coefficients_mat' "
                    f"(provided {rhs_vec_arr.size} entries)"
                )

        self._rhs_vec = rhs_vec_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the polyhedral set.
        """
        return len(self.coefficients_mat[0])

    @property
    def geometry(self):
        """
        Geometry of the polyhedral set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the polyhedral set.

        Currently, an empty `list` is returned, as the bounds cannot, in
        general, be computed without access to an optimization solver.
        """
        return []

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_data_list, conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block, uncertain_param_vars=uncertain_params, dim=self.dim
            )
        )

        for row, rhs_val in zip(self.coefficients_mat, self.rhs_vec):
            lhs_expr = dot_product(row, param_var_data_list, index=range(row.size))
            conlist.add(lhs_expr <= rhs_val)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=list(conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    def validate(self, config):
        """
        Check PolyhedralSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            or ``self.coefficients_mat`` has column of zeros).
            If bounded and nonempty checks fail.
        """
        lhs_coeffs_arr = self.coefficients_mat
        rhs_vec_arr = self.rhs_vec

        # check lhs matrix and rhs vector are valid
        # this includes a finiteness check
        validate_array(
            arr=lhs_coeffs_arr,
            arr_name="coefficients_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_array(
            arr=rhs_vec_arr,
            arr_name="rhs_vec",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        # check no column is all zeros. otherwise, set is unbounded
        cols_with_all_zeros = np.nonzero(np.all(lhs_coeffs_arr == 0, axis=0))[0]
        if cols_with_all_zeros.size > 0:
            col_str = ", ".join(str(val) for val in cols_with_all_zeros)
            raise ValueError(
                "Attempting to set attribute 'coefficients_mat' to value "
                f"with all entries zero in columns at indexes: {col_str}. "
                "Ensure column has at least one nonzero entry"
            )

        # check boundedness and nonemptiness
        super().validate(config)


class BudgetSet(UncertaintySet):
    """
    A budget set.

    Parameters
    ----------
    budget_membership_mat : (L, N) array_like
        Incidence matrix of the budget constraints.
        Each row corresponds to a single budget constraint,
        and defines which uncertain parameters
        (which dimensions) participate in that row's constraint.
    rhs_vec : (L,) array_like
        Budget limits (upper bounds) with respect to
        the origin of the set.
    origin : (N,) array_like or None, optional
        Origin of the budget set. If `None` is provided, then
        the origin is set to the zero vector.

    Examples
    --------
    3D budget set with one budget constraint and
    no origin chosen (hence origin defaults to 3D zero vector):

    >>> from pyomo.contrib.pyros import BudgetSet
    >>> budget_set = BudgetSet(
    ...     budget_membership_mat=[[1, 1, 1]],
    ...     rhs_vec=[2],
    ... )
    >>> budget_set.budget_membership_mat
    array([[1, 1, 1]])
    >>> budget_set.budget_rhs_vec
    array([2])
    >>> budget_set.origin
    array([0., 0., 0.])

    3D budget set with two budget constraints and custom origin:

    >>> budget_custom = BudgetSet(
    ...     budget_membership_mat=[[1, 0, 1], [0, 1, 0]],
    ...     rhs_vec=[1, 1],
    ...     origin=[2, 2, 2],
    ... )
    >>> budget_custom.budget_membership_mat
    array([[1, 0, 1],
           [0, 1, 0]])
    >>> budget_custom.budget_rhs_vec
    array([1, 1])
    >>> budget_custom.origin
    array([2, 2, 2])
    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, budget_membership_mat, rhs_vec, origin=None):
        """Initialize self (see class docstring)."""
        self.budget_membership_mat = budget_membership_mat
        self.budget_rhs_vec = rhs_vec
        self.origin = np.zeros(self.dim) if origin is None else origin

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "budget"

    @property
    def coefficients_mat(self):
        """
        (L + N, N) numpy.ndarray : Coefficient matrix of all polyhedral
        constraints defining the budget set. Composed from the incidence
        matrix used for defining the budget constraints and a
        coefficient matrix for individual uncertain parameter
        nonnegativity constraints.

        This attribute cannot be set. The budget constraint
        incidence matrix may be altered through the
        `budget_membership_mat` attribute.
        """
        return np.append(self.budget_membership_mat, -np.identity(self.dim), axis=0)

    @property
    def rhs_vec(self):
        """
        (L + N,) numpy.ndarray : Right-hand side vector for polyhedral
        constraints defining the budget set. This also includes entries
        for nonnegativity constraints on the uncertain parameters.

        This attribute cannot be set, and is automatically determined
        given other attributes.
        """
        return np.append(
            self.budget_rhs_vec + self.budget_membership_mat @ self.origin, -self.origin
        )

    @property
    def budget_membership_mat(self):
        """
        (L, N) numpy.ndarray : Incidence matrix of the budget
        constraints.  Each row corresponds to a single budget
        constraint and defines which uncertain parameters
        participate in that row's constraint.
        """
        return self._budget_membership_mat

    @budget_membership_mat.setter
    def budget_membership_mat(self, val):
        validate_array(
            arr=val,
            arr_name="budget_membership_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        lhs_coeffs_arr = np.array(val)

        # check dimension match
        if hasattr(self, "_budget_membership_mat"):
            if lhs_coeffs_arr.shape[1] != self.dim:
                raise ValueError(
                    f"BudgetSet attribute 'budget_membership_mat' "
                    "must have "
                    f"{self.dim} columns to match set dimension "
                    f"(provided matrix with {lhs_coeffs_arr.shape[1]} columns)"
                )

        # check shape match with rhs vector
        if hasattr(self, "_budget_rhs_vec"):
            if lhs_coeffs_arr.shape[0] != self.budget_rhs_vec.size:
                raise ValueError(
                    "BudgetSet attribute 'budget_membership_mat' "
                    f"must have {self.budget_rhs_vec.size} rows "
                    f"to match shape of attribute 'budget_rhs_vec' "
                    f"(provided {lhs_coeffs_arr.shape[0]} rows)"
                )
        # matrix is valid; update
        self._budget_membership_mat = lhs_coeffs_arr

    @property
    def budget_rhs_vec(self):
        """
        (L,) numpy.ndarray : Budget limits (upper bounds)
        with respect to the origin.
        """
        return self._budget_rhs_vec

    @budget_rhs_vec.setter
    def budget_rhs_vec(self, val):
        validate_array(
            arr=val,
            arr_name="budget_rhs_vec",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        rhs_vec_arr = np.array(val)

        # ensure shape of coefficients matrix
        # and rhs vec match
        if hasattr(self, "_budget_membership_mat"):
            if len(val) != self.budget_membership_mat.shape[0]:
                raise ValueError(
                    "Budget set attribute 'budget_rhs_vec' "
                    f"must have {self.budget_membership_mat.shape[0]} entries "
                    f"to match shape of attribute 'budget_membership_mat' "
                    f"(provided {rhs_vec_arr.size} entries)"
                )

        self._budget_rhs_vec = rhs_vec_arr

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Origin of the budget set.
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(
            arr=val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        origin_arr = np.array(val)

        # ensure shape of coefficients matrix
        # and rhs vec match
        if len(val) != self.dim:
            raise ValueError(
                "Budget set attribute 'origin' "
                f"must have {self.dim} entries "
                f"to match set dimension "
                f"(provided {origin_arr.size} entries)"
            )

        self._origin = origin_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the budget set.
        """
        return self.budget_membership_mat.shape[1]

    @property
    def geometry(self):
        """
        Geometry of the budget set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the budget set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        bounds = []
        for orig_val, col in zip(self.origin, self.budget_membership_mat.T):
            lb = orig_val
            ub = orig_val + np.min(self.budget_rhs_vec[col == 1])
            bounds.append((lb, ub))

        return bounds

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, **kwargs):
        return PolyhedralSet.set_as_constraint(self, **kwargs)

    def validate(self, config):
        """
        Check BudgetSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            ``self.budget_membership_mat`` contains a
            full column or row of zeros,
            or ``self.budget_rhs_vec`` has negative values).
        """
        lhs_coeffs_arr = self.budget_membership_mat
        rhs_vec_arr = self.budget_rhs_vec
        orig_val = self.origin

        # check budget matrix, budget limits, and origin are valid
        # this includes a finiteness check
        validate_array(
            arr=lhs_coeffs_arr,
            arr_name="budget_membership_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_array(
            arr=rhs_vec_arr,
            arr_name="budget_rhs_vec",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_array(
            arr=orig_val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        # check no row, col, are all zeros and all values are 0-1.
        # ensure all entries are 0-1 values
        uniq_entries = np.unique(lhs_coeffs_arr)
        non_bool_entries = uniq_entries[(uniq_entries != 0) & (uniq_entries != 1)]
        if non_bool_entries.size > 0:
            raise ValueError(
                "Attempting to set attribute `budget_membership_mat` to value "
                "containing entries that are not 0-1 values "
                f"(example: {non_bool_entries[0]}). "
                "Ensure all entries are of value 0 or 1"
            )

        # check no row is all zeros
        rows_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=1) == 0)[0]
        if rows_with_zero_sums.size > 0:
            row_str = ", ".join(str(val) for val in rows_with_zero_sums)
            raise ValueError(
                "Attempting to set attribute `budget_membership_mat` to value "
                f"with all entries zero in rows at indexes: {row_str}. "
                "Ensure each row and column has at least one nonzero entry"
            )

        # check no column is all zeros
        cols_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=0) == 0)[0]
        if cols_with_zero_sums.size > 0:
            col_str = ", ".join(str(val) for val in cols_with_zero_sums)
            raise ValueError(
                "Attempting to set attribute `budget_membership_mat` to value "
                f"with all entries zero in columns at indexes: {col_str}. "
                "Ensure each row and column has at least one nonzero entry"
            )

        # ensure all rhs entries are nonnegative
        for entry in rhs_vec_arr:
            if entry < 0:
                raise ValueError(
                    f"Entry {entry} of attribute 'budget_rhs_vec' is "
                    "negative. Ensure all entries are nonnegative"
                )


class FactorModelSet(UncertaintySet):
    """
    A factor model (i.e., "net-alpha" model) set.

    Parameters
    ----------
    origin : (N,) array_like
        Uncertain parameter values around which deviations are
        restrained.
    number_of_factors : int
        Natural number representing the dimension of the
        space to which the set projects.
    psi_mat : (N, F) array_like
        Matrix, of full column rank, designating each uncertain
        parameter's contribution to each factor.
        Each row is associated with a separate uncertain
        parameter.  Each column is associated with a separate factor.
        Number of columns `F` of `psi_mat` should be equal to
        `number_of_factors`.
        Since `psi_mat` is expected to be full column rank,
        we require `F <= N`.
    beta : numeric type
        Real value between 0 and 1 specifying the fraction of the
        independent factors that can simultaneously attain
        their extreme values.

    Examples
    --------
    A 4D factor model set with a 2D factor space:

    >>> from pyomo.contrib.pyros import FactorModelSet
    >>> import numpy as np
    >>> fset = FactorModelSet(
    ...     origin=np.zeros(4),
    ...     number_of_factors=2,
    ...     psi_mat=[[0, 0.1], [0, 0.1], [0.1, 0], [0.1, 0]],
    ...     beta=0.5,
    ... )
    >>> fset.origin
    array([0., 0., 0., 0.])
    >>> fset.number_of_factors
    2
    >>> fset.psi_mat
    array([[0. , 0.1],
           [0. , 0.1],
           [0.1, 0. ],
           [0.1, 0. ]])
    >>> fset.beta
    0.5
    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, origin, number_of_factors, psi_mat, beta):
        """Initialize self (see class docstring)."""
        self.origin = origin
        self.number_of_factors = number_of_factors
        self.beta = beta
        self.psi_mat = psi_mat

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "factor_model"

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Uncertain parameter values around which
        deviations are restrained.
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(
            arr=val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )

        # dimension of the set is immutable
        val_arr = np.array(val)
        if hasattr(self, "_origin"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'origin' of factor model "
                    f"set of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._origin = val_arr

    @property
    def number_of_factors(self):
        """
        int : Natural number representing the dimension `F`
        of the space to which the set projects.

        This attribute is immutable, may only be set at
        object construction, and must be equal to the number of
        columns of the factor loading matrix ``self.psi_mat``.
        Therefore, since we also require that ``self.psi_mat``
        be full column rank, `number_of_factors`
        must not exceed the set dimension.
        """
        return self._number_of_factors

    @number_of_factors.setter
    def number_of_factors(self, val):
        if hasattr(self, "_number_of_factors"):
            raise AttributeError("Attribute 'number_of_factors' is immutable")
        else:
            # validate type and value
            validate_arg_type("number_of_factors", val, Integral)
            if val < 1:
                raise ValueError(
                    "Attribute 'number_of_factors' must be a positive int "
                    f"(provided value {val})"
                )
        self._number_of_factors = val

    @property
    def psi_mat(self):
        """
        (N, F) numpy.ndarray : Factor loading matrix, i.e., a full
        column rank matrix for which each entry indicates how strongly
        the factor corresponding to the entry's column is related
        to the uncertain parameter corresponding to the entry's row.
        Since `psi_mat` is expected to be full column rank,
        we require `F <= N`.
        """
        return self._psi_mat

    @psi_mat.setter
    def psi_mat(self, val):
        validate_array(
            arr=val,
            arr_name="psi_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        psi_mat_arr = np.array(val)

        # validate shape (check it matches set dimensions)
        # origin and number of factors already set
        if psi_mat_arr.shape != (self.dim, self.number_of_factors):
            raise ValueError(
                "Psi matrix for factor model set "
                f"should be of shape {self.dim, self.number_of_factors} "
                f"to match the set and factor model space dimensions "
                f"(provided shape {psi_mat_arr.shape})"
            )

        self._psi_mat = psi_mat_arr

    @property
    def beta(self):
        """
        numeric type : Real number ranging from 0 to 1 representing the
        fraction of the independent factors that can simultaneously
        attain their extreme values.

        Note that, mathematically, setting ``beta = 0`` will enforce
        that as many factors will be above 0 as there will be below 0
        (i.e., "zero-net-alpha" model). If ``beta = 1``,
        then the set is numerically equivalent to a `BoxSet` with bounds
        ``[self.origin - psi @ np.ones(F), self.origin + psi @ np.ones(F)].T``.
        """
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val

    @property
    def dim(self):
        """
        int : Dimension `N` of the factor model set.
        """
        return len(self.origin)

    @property
    def geometry(self):
        """
        Geometry of the factor model set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the factor model set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        F = self.number_of_factors
        psi_mat = self.psi_mat

        # evaluate some important quantities
        beta_F = self.beta * self.number_of_factors
        crit_pt_type = int((beta_F + F) / 2)
        beta_F_fill_in = (beta_F + F) - 2 * crit_pt_type - 1

        # argsort rows of psi_mat in descending order
        row_wise_args = np.argsort(-psi_mat, axis=1)

        parameter_bounds = []
        for idx, orig_val in enumerate(self.origin):
            # number nonnegative values in row
            M = len(psi_mat[idx][psi_mat[idx] >= 0])

            # argsort psi matrix row in descending order
            sorted_psi_row_args = row_wise_args[idx]
            sorted_psi_row = psi_mat[idx, sorted_psi_row_args]

            # now evaluate max deviation from origin
            # (depends on number nonneg entries and critical point type)
            if M > crit_pt_type:
                max_deviation = (
                    sorted_psi_row[:crit_pt_type].sum()
                    + beta_F_fill_in * sorted_psi_row[crit_pt_type]
                    - sorted_psi_row[crit_pt_type + 1 :].sum()
                )
            elif M < F - crit_pt_type:
                max_deviation = (
                    sorted_psi_row[: F - crit_pt_type - 1].sum()
                    - beta_F_fill_in * sorted_psi_row[F - crit_pt_type - 1]
                    - sorted_psi_row[F - crit_pt_type :].sum()
                )
            else:
                max_deviation = sorted_psi_row[:M].sum() - sorted_psi_row[M:].sum()

            # finally, evaluate the bounds for this dimension
            parameter_bounds.append(
                (orig_val - max_deviation, orig_val + max_deviation)
            )

        return parameter_bounds

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_data_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=self.number_of_factors,
            )
        )

        factor_zip = zip(self.origin, self.psi_mat, param_var_data_list)
        for orig_val, psi_row, param_var in factor_zip:
            psi_dot_product = dot_product(
                psi_row, aux_var_list, index=range(self.number_of_factors)
            )
            uncertainty_conlist.add(orig_val + psi_dot_product == param_var)

        # absolute value constraints on sum of auxiliary vars
        beta_F = self.beta * self.number_of_factors
        uncertainty_conlist.add((-beta_F, quicksum(aux_var_list), beta_F))

        for var in aux_var_list:
            var.setlb(-1)
            var.setub(1)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=list(uncertainty_conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    @copy_docstring(UncertaintySet.compute_auxiliary_uncertain_param_vals)
    def compute_auxiliary_uncertain_param_vals(self, point, solver=None):
        validate_array(
            arr=point,
            arr_name="point",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="numeric type",
            required_shape=[self.dim],
            required_shape_qual="to match the set dimension",
        )
        point_arr = np.array(point)

        # protect against cases where
        # `psi_mat` was recently modified entrywise
        # to a matrix that is not full column rank
        self.psi_mat = self.psi_mat

        # since `psi_mat` is full column rank,
        # the pseudoinverse uniquely determines the auxiliary values
        return np.linalg.pinv(self.psi_mat) @ (point_arr - self.origin)

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the factor model set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        aux_space_pt = self.compute_auxiliary_uncertain_param_vals(point)
        tol = POINT_IN_UNCERTAINTY_SET_TOL
        return abs(
            aux_space_pt.sum()
        ) <= self.beta * self.number_of_factors + tol and np.all(
            np.abs(aux_space_pt) <= 1 + tol
        )

    def validate(self, config):
        """
        Check FactorModelSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            ``self.psi_mat`` is not full column rank,
            or ``self.beta`` is not between 0 and 1).
        """
        orig_val = self.origin
        psi_mat_arr = self.psi_mat
        beta = self.beta

        # check origin, psi matrix, and beta are valid
        # this includes a finiteness check
        validate_array(
            arr=orig_val,
            arr_name="origin",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
        )
        validate_array(
            arr=psi_mat_arr,
            arr_name="psi_mat",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_arg_type(
            "beta", beta, native_numeric_types, "a valid numeric type", False
        )

        # check psi is full column rank
        psi_mat_rank = np.linalg.matrix_rank(psi_mat_arr)
        check_full_column_rank = psi_mat_rank == self.number_of_factors
        if not check_full_column_rank:
            raise ValueError(
                "Attribute 'psi_mat' should be full column rank. "
                f"(Got a matrix of shape {psi_mat_arr.shape} and rank {psi_mat_rank}.) "
                "Ensure `psi_mat` does not have more columns than rows, "
                "and the columns of `psi_mat` are linearly independent."
            )

        # check beta is between 0 and 1
        if beta > 1 or beta < 0:
            raise ValueError(
                "Beta parameter must be a real number between 0 "
                f"and 1 inclusive (provided value {beta})"
            )


class AxisAlignedEllipsoidalSet(UncertaintySet):
    """
    An axis-aligned ellipsoid.

    Parameters
    ----------
    center : (N,) array_like
        Center of the ellipsoid.
    half_lengths : (N,) array_like
        Semi-axis lengths of the ellipsoid.

    Examples
    --------
    3D origin-centered unit hypersphere:

    >>> from pyomo.contrib.pyros import AxisAlignedEllipsoidalSet
    >>> sphere = AxisAlignedEllipsoidalSet(
    ...     center=[0, 0, 0],
    ...     half_lengths=[1, 1, 1]
    ... )
    >>> sphere.center
    array([0, 0, 0])
    >>> sphere.half_lengths
    array([1, 1, 1])

    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, center, half_lengths):
        """Initialize self (see class docstring)."""
        self.center = center
        self.half_lengths = half_lengths

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "ellipsoidal"

    @property
    def center(self):
        """
        (N,) numpy.ndarray : Center of the ellipsoid.
        """
        return self._center

    @center.setter
    def center(self, val):
        validate_array(
            arr=val,
            arr_name="center",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        val_arr = np.array(val)

        # dimension of the set is immutable
        if hasattr(self, "_center"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'center' of "
                    f"AxisAlignedEllipsoidalSet of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._center = val_arr

    @property
    def half_lengths(self):
        """
        (N,) numpy.ndarray : Semi-axis lengths.
        """
        return self._half_lengths

    @half_lengths.setter
    def half_lengths(self, val):
        validate_array(
            arr=val,
            arr_name="half_lengths",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        val_arr = np.array(val)

        # dimension of the set is immutable
        if hasattr(self, "_center"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'half_lengths' of "
                    f"AxisAlignedEllipsoidalSet of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._half_lengths = val_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the axis-aligned ellipsoidal set.
        """
        return len(self.center)

    @property
    def geometry(self):
        """
        Geometry of the axis-aligned ellipsoidal set.
        See the `Geometry` class documentation.
        """
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the axis-aligned ellipsoidal set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        nom_value = self.center
        half_length = self.half_lengths
        parameter_bounds = [
            (nom_value[i] - half_length[i], nom_value[i] + half_length[i])
            for i in range(len(nom_value))
        ]
        return parameter_bounds

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_data_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )

        # now construct the constraints
        diffs_squared = list()
        zip_all = zip(param_var_data_list, self.center, self.half_lengths)
        for param, ctr, half_len in zip_all:
            if half_len > 0:
                diffs_squared.append((param - ctr) ** 2 / (half_len) ** 2)
            else:
                # equality constraints for parameters corresponding to
                # half-lengths of zero
                uncertainty_conlist.add(param == ctr)

        if diffs_squared:
            uncertainty_conlist.add(quicksum(diffs_squared) <= 1)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=list(uncertainty_conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    def validate(self, config):
        """
        Check AxisAlignedEllipsoidalSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            or ``self.half_lengths`` are negative).
        """
        ctr = self.center
        half_lengths = self.half_lengths

        # check center and half lengths are valid
        # this includes a finiteness check
        validate_array(
            arr=ctr,
            arr_name="center",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_array(
            arr=half_lengths,
            arr_name="half_lengths",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        # ensure half-lengths are non-negative
        for half_len in half_lengths:
            if half_len < 0:
                raise ValueError(
                    f"Entry {half_len} of 'half_lengths' "
                    "is negative. All half-lengths must be nonnegative"
                )


class EllipsoidalSet(UncertaintySet):
    """
    A general ellipsoid.

    Parameters
    ----------
    center : (N,) array-like
        Center of the ellipsoid.
    shape_matrix : (N, N) array-like
        A symmetric positive definite matrix characterizing
        the shape and orientation of the ellipsoid.
    scale : numeric type, optional
        Square of the factor by which to scale the semi-axes
        of the ellipsoid (i.e. the eigenvectors of the shape
        matrix). The default is `1`.
    gaussian_conf_lvl : numeric type, optional
        (Fractional) confidence level of the multivariate
        normal distribution with mean `center` and covariance
        matrix `shape_matrix`.
        Exactly one of `scale` and `gaussian_conf_lvl` should be
        None; otherwise, an exception is raised.

    Examples
    --------
    A 3D origin-centered unit ball:

    >>> from pyomo.contrib.pyros import EllipsoidalSet
    >>> import numpy as np
    >>> ball = EllipsoidalSet(
    ...     center=[0, 0, 0],
    ...     shape_matrix=np.eye(3),
    ...     scale=1,
    ... )
    >>> ball.center
    array([0, 0, 0])
    >>> ball.shape_matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> ball.scale
    1

    A 2D ellipsoid with custom rotation and scaling:

    >>> rotated_ellipsoid = EllipsoidalSet(
    ...     center=[1, 1],
    ...     shape_matrix=[[4, 2], [2, 4]],
    ...     scale=0.5,
    ... )
    >>> rotated_ellipsoid.center
    array([1, 1])
    >>> rotated_ellipsoid.shape_matrix
    array([[4, 2],
           [2, 4]])
    >>> rotated_ellipsoid.scale
    0.5

    A 4D 95% confidence ellipsoid:

    >>> conf_ellipsoid = EllipsoidalSet(
    ...     center=np.zeros(4),
    ...     shape_matrix=np.diag(range(1, 5)),
    ...     scale=None,
    ...     gaussian_conf_lvl=0.95,
    ... )
    >>> conf_ellipsoid.center
    array([0., 0., 0., 0.])
    >>> conf_ellipsoid.shape_matrix
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])
    >>> conf_ellipsoid.scale  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <BLANKLINE>
    ...9.4877...
    >>> conf_ellipsoid.gaussian_conf_lvl
    0.95

    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, center, shape_matrix, scale=1, gaussian_conf_lvl=None):
        """Initialize self (see class docstring)."""
        self.center = center
        self.shape_matrix = shape_matrix

        if scale is not None and gaussian_conf_lvl is None:
            self.scale = scale
        elif scale is None and gaussian_conf_lvl is not None:
            self.gaussian_conf_lvl = gaussian_conf_lvl
        else:
            raise ValueError(
                "Exactly one of `scale` and `gaussian_conf_lvl` should be "
                f"None (got {scale=}, {gaussian_conf_lvl=})"
            )

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "ellipsoidal"

    @property
    def center(self):
        """
        (N,) numpy.ndarray : Center of the ellipsoid.
        """
        return self._center

    @center.setter
    def center(self, val):
        validate_array(
            arr=val,
            arr_name="center",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        val_arr = np.array(val)

        # dimension of the set is immutable
        if hasattr(self, "_center"):
            if val_arr.size != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'center' of "
                    f"{type(self).__name__} of dimension {self.dim} "
                    f"to value of dimension {val_arr.size}"
                )

        self._center = val_arr

    @staticmethod
    def _verify_positive_definite(matrix):
        """
        Verify that a given symmetric square matrix is positive
        definite. An exception is raised if the square matrix
        is not positive definite.

        Parameters
        ----------
        matrix : (N, N) array_like
            Candidate matrix.

        Raises
        ------
        ValueError
            If matrix is not symmetric, not positive definite,
            or the square roots of the diagonal entries are
            not accessible.
        LinAlgError
            If matrix is not invertible.
        """
        matrix = np.array(matrix)

        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("Shape matrix must be symmetric.")

        # Numpy raises LinAlgError if not invertible
        np.linalg.inv(matrix)

        # check positive semi-definite.
        # since also invertible, means positive definite
        eigvals = np.linalg.eigvals(matrix)
        if np.min(eigvals) < 0:
            raise ValueError(
                "Non positive-definite shape matrix "
                f"(detected eigenvalues {eigvals})"
            )

        # check roots of diagonal entries accessible
        # (should theoretically be true if positive definite)
        for diag_entry in np.diagonal(matrix):
            if np.isnan(np.power(diag_entry, 0.5)):
                raise ValueError(
                    "Cannot evaluate square root of the diagonal entry "
                    f"{diag_entry} of argument `shape_matrix`. "
                    "Check that this entry is nonnegative"
                )

    @property
    def shape_matrix(self):
        """
        (N, N) numpy.ndarray : A positive definite matrix characterizing
        the shape and orientation of the ellipsoid.
        """
        return self._shape_matrix

    @shape_matrix.setter
    def shape_matrix(self, val):
        validate_array(
            arr=val,
            arr_name="shape_matrix",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        shape_mat_arr = np.array(val)

        # check matrix shape matches set dimension
        if hasattr(self, "_center"):
            if not all(size == self.dim for size in shape_mat_arr.shape):
                raise ValueError(
                    f"{type(self).__name__} attribute 'shape_matrix' "
                    f"must be a square matrix of size "
                    f"{self.dim} to match set dimension "
                    f"(provided matrix with shape {shape_mat_arr.shape})"
                )

        self._shape_matrix = shape_mat_arr

    @property
    def scale(self):
        """
        numeric type : Square of the factor by which to scale the
        semi-axes of the ellipsoid (i.e. the eigenvectors of the shape
        matrix).
        """
        return self._scale

    @scale.setter
    def scale(self, val):
        validate_arg_type(
            "scale", val, native_numeric_types, "a valid numeric type", False
        )

        self._scale = val
        self._gaussian_conf_lvl = sp.stats.chi2.cdf(x=val, df=self.dim)

    @property
    def gaussian_conf_lvl(self):
        """
        numeric type : (Fractional) confidence level of the
        multivariate Gaussian distribution with mean ``self.origin``
        and covariance ``self.shape_matrix`` for ellipsoidal region
        with square magnification factor ``self.scale``.
        """
        return self._gaussian_conf_lvl

    @gaussian_conf_lvl.setter
    def gaussian_conf_lvl(self, val):
        validate_arg_type(
            "gaussian_conf_lvl",
            val,
            native_numeric_types,
            "a valid numeric type",
            False,
        )

        scale_val = sp.stats.chi2.isf(q=1 - val, df=self.dim)
        if np.isnan(scale_val) or np.isinf(scale_val):
            raise ValueError(
                f"Squared scaling factor calculation for confidence level {val} "
                f"and set dimension {self.dim} returned {scale_val}. "
                "Ensure the confidence level is a value in [0, 1)."
            )

        self._gaussian_conf_lvl = val
        self._scale = scale_val

    @property
    def dim(self):
        """
        int : Dimension `N` of the ellipsoidal set.
        """
        return len(self.center)

    @property
    def geometry(self):
        """
        Geometry of the ellipsoidal set.
        See the `Geometry` class documentation.
        """
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the ellipsoidal set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        scale = self.scale
        nom_value = self.center
        P = self.shape_matrix
        parameter_bounds = [
            (
                nom_value[i] - np.power(P[i][i] * scale, 0.5),
                nom_value[i] + np.power(P[i][i] * scale, 0.5),
            )
            for i in range(self.dim)
        ]
        return parameter_bounds

    @copy_docstring(UncertaintySet.point_in_set)
    def point_in_set(self, point):
        validate_array(
            arr=point,
            arr_name="point",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="numeric type",
            required_shape=[self.dim],
            required_shape_qual="to match the set dimension",
        )
        off_center = point - self.center
        normalized_pt_radius = np.sqrt(
            off_center @ np.linalg.inv(self.shape_matrix) @ off_center
        )
        normalized_boundary_radius = np.sqrt(self.scale)
        return (
            normalized_pt_radius
            <= normalized_boundary_radius + POINT_IN_UNCERTAINTY_SET_TOL
        )

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_data_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )

        inv_shape_mat = np.linalg.inv(self.shape_matrix)
        with mutable_expression() as expr:
            for (idx1, idx2), mat_entry in np.ndenumerate(inv_shape_mat):
                expr += (
                    mat_entry
                    * (param_var_data_list[idx1] - self.center[idx1])
                    * (param_var_data_list[idx2] - self.center[idx2])
                )
        uncertainty_conlist.add(expr <= self.scale)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=list(uncertainty_conlist.values()),
            auxiliary_vars=aux_var_list,
        )

    def validate(self, config):
        """
        Check EllipsoidalSet validity.

        Raises
        ------
        ValueError
            If any uncertainty set attributes are not valid.
            (e.g., numeric values are infinite,
            ``self.shape_matrix`` is not positive semidefinite,
            or ``self.scale`` is negative).
        """
        ctr = self.center
        shape_mat_arr = self.shape_matrix
        scale = self.scale

        # check center, shape matrix, and scale are valid
        # this includes a finiteness check
        validate_array(
            arr=ctr,
            arr_name="center",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_array(
            arr=shape_mat_arr,
            arr_name="shape_matrix",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )
        validate_arg_type(
            "scale", scale, native_numeric_types, "a valid numeric type", False
        )

        # check shape matrix is positive semidefinite
        self._verify_positive_definite(shape_mat_arr)

        # ensure scale is non-negative
        if scale < 0:
            raise ValueError(
                f"{type(self).__name__} attribute "
                f"'scale' must be a non-negative real "
                f"(provided value {scale})"
            )


class DiscreteScenarioSet(UncertaintySet):
    """
    A discrete set of finitely many uncertain parameter realizations
    (or scenarios).

    Parameters
    ----------
    scenarios : (M, N) array_like
        A sequence of `M` distinct uncertain parameter realizations.

    Examples
    --------
    2D set with three scenarios:

    >>> from pyomo.contrib.pyros import DiscreteScenarioSet
    >>> discrete_set = DiscreteScenarioSet(
    ...     scenarios=[[1, 1], [2, 1], [1, 2]],
    ... )
    >>> discrete_set.scenarios
    [(1, 1), (2, 1), (1, 2)]

    """

    _PARAMETER_BOUNDS_EXACT = True

    def __init__(self, scenarios):
        """Initialize self (see class docstring)."""
        # Standardize to list of tuples
        self.scenarios = scenarios

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "discrete"

    @property
    def scenarios(self):
        """
        list of tuples : Uncertain parameter realizations comprising the
        set.  Each tuple is an uncertain parameter realization.

        Note that the `scenarios` attribute may be modified, but
        only such that the dimension of the set remains unchanged.
        """
        return self._scenarios

    @scenarios.setter
    def scenarios(self, val):
        validate_array(
            arr=val,
            arr_name="scenarios",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        scenario_arr = np.array(val)
        if hasattr(self, "_scenarios"):
            if scenario_arr.shape[1] != self.dim:
                raise ValueError(
                    f"DiscreteScenarioSet attribute 'scenarios' must have "
                    f"{self.dim} columns to match set dimension "
                    f"(provided array-like with {scenario_arr.shape[1]} "
                    "columns)"
                )

        self._scenarios = [tuple(s) for s in val]

    @property
    def dim(self):
        """
        int : Dimension `N` of the discrete scenario set.
        """
        return len(self.scenarios[0])

    @property
    def geometry(self):
        """
        Geometry of the discrete scenario set.
        See the `Geometry` class documentation.
        """
        return Geometry.DISCRETE_SCENARIOS

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the discrete scenario set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        parameter_bounds = [
            (min(s[i] for s in self.scenarios), max(s[i] for s in self.scenarios))
            for i in range(self.dim)
        ]
        return parameter_bounds

    def is_bounded(self, config):
        """
        Return True if the uncertainty set is bounded, and False
        otherwise.

        By default, the discrete scenario set is bounded,
        as the entries of all uncertain parameter scenarios
        are finite.
        """
        return True

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        # === Ensure point is of correct dimensionality as the uncertain parameters
        block, param_var_data_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )

        # no constraints declared for the discrete set;
        # instead, the param vars are fixed during separation

        return UncertaintyQuantification(
            block=block,
            uncertainty_cons=list(uncertainty_conlist.values()),
            uncertain_param_vars=param_var_data_list,
            auxiliary_vars=aux_var_list,
        )

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the discrete
        scenario set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        validate_array(
            arr=point,
            arr_name="point",
            dim=1,
            valid_types=native_numeric_types,
            valid_type_desc="numeric type",
            required_shape=[self.dim],
            required_shape_qual="to match the set dimension",
        )
        # Round all double precision to a tolerance
        num_decimals = round(-np.log10(POINT_IN_UNCERTAINTY_SET_TOL))
        rounded_scenarios = np.round(self.scenarios, decimals=num_decimals)
        rounded_point = np.round(point, decimals=num_decimals)
        return np.any(np.all(rounded_point == rounded_scenarios, axis=1))

    def validate(self, config):
        """
        Check DiscreteScenarioSet validity.

        Raises
        ------
        ValueError
            If finiteness or nonemptiness checks fail.
        """
        scenario_arr = self.scenarios

        # check that all scenarios are valid
        # this includes a nonemptiness check and a finiteness check
        # using the validate_arr method
        validate_array(
            arr=scenario_arr,
            arr_name="scenarios",
            dim=2,
            valid_types=native_numeric_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )


class IntersectionSet(UncertaintySet):
    """
    An intersection of a sequence of uncertainty sets, each of which
    is represented by an `UncertaintySet` object.

    Parameters
    ----------
    **unc_sets : dict
        PyROS `UncertaintySet` objects of which to construct
        an intersection. At least two uncertainty sets must
        be provided. All sets must be of the same dimension.

    Examples
    --------
    Intersection of origin-centered 2D box (square) and 2D
    hypersphere (circle):

    >>> from pyomo.contrib.pyros import (
    ...     BoxSet, AxisAlignedEllipsoidalSet, IntersectionSet,
    ... )
    >>> square = BoxSet(bounds=[[-1.5, 1.5], [-1.5, 1.5]])
    >>> circle = AxisAlignedEllipsoidalSet(
    ...     center=[0, 0],
    ...     half_lengths=[2, 2],
    ... )
    >>> # to construct intersection, pass sets as keyword arguments
    >>> intersection = IntersectionSet(set1=square, set2=circle)
    >>> intersection.all_sets  # doctest: +ELLIPSIS
    UncertaintySetList([...])

    """

    def __init__(self, **unc_sets):
        """Initialize self (see class docstring)."""
        self.all_sets = unc_sets

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return "intersection"

    @property
    def all_sets(self):
        """
        UncertaintySetList : List of the uncertainty sets of which to
        take the intersection. Must be of minimum length 2.

        This attribute may be set through any iterable of
        `UncertaintySet` objects, and exhibits similar behavior
        to a `list`.
        """
        return self._all_sets

    @all_sets.setter
    def all_sets(self, val):
        if isinstance(val, dict):
            the_sets = val.values()
        else:
            the_sets = list(val)

        # type validation, ensure all entries have same dimension
        all_sets = UncertaintySetList(the_sets, name="all_sets", min_length=2)

        # set dimension is immutable
        if hasattr(self, "_all_sets"):
            if all_sets.dim != self.dim:
                raise ValueError(
                    "Attempting to set attribute 'all_sets' of an "
                    f"IntersectionSet of dimension {self.dim} to a sequence "
                    f"of sets of dimension {all_sets[0].dim}"
                )

        self._all_sets = all_sets

    @property
    def dim(self):
        """
        int : Dimension of the intersection set.
        """
        return self.all_sets[0].dim

    @property
    def geometry(self):
        """
        Geometry of the intersection set.
        See the `Geometry` class documentation.
        """
        return max(self.all_sets[i].geometry.value for i in range(len(self.all_sets)))

    @property
    def parameter_bounds(self):
        """
        Uncertain parameter value bounds for the intersection
        set.

        Currently, an empty list, as the bounds cannot, in general,
        be computed without access to an optimization solver.
        """
        return []

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the intersection set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        if all(a_set.point_in_set(point=point) for a_set in self.all_sets):
            return True
        else:
            return False

    # === Define pairwise intersection function
    @staticmethod
    def intersect(Q1, Q2):
        """
        Obtain the intersection of two uncertainty sets,
        accounting for the case where either of the two sets
        is discrete.

        Parameters
        ----------
        Q1, Q2 : UncertaintySet
            Operand uncertainty sets.

        Returns
        -------
        DiscreteScenarioSet or IntersectionSet
            Intersection of the sets. A `DiscreteScenarioSet` is
            returned if both operand sets are `DiscreteScenarioSet`
            instances; otherwise, an `IntersectionSet` is returned.
        """
        for set1, set2 in zip((Q1, Q2), (Q2, Q1)):
            if isinstance(set1, DiscreteScenarioSet):
                return DiscreteScenarioSet(
                    scenarios=[pt for pt in set1.scenarios if set1.point_in_set(pt)]
                )

        # === This case is if both sets are continuous
        return IntersectionSet(set1=Q1, set2=Q2)

    @copy_docstring(UncertaintySet.set_as_constraint)
    def set_as_constraint(self, uncertain_params=None, block=None):
        block, param_var_data_list, uncertainty_conlist, aux_var_list = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )

        intersection_set = functools.reduce(self.intersect, self.all_sets)
        if isinstance(intersection_set, DiscreteScenarioSet):
            return intersection_set.set_as_constraint(
                uncertain_params=uncertain_params, block=block
            )

        all_cons, all_aux_vars = [], []
        for idx, unc_set in enumerate(intersection_set.all_sets):
            sub_block = Block()
            block.add_component(
                unique_component_name(block, f"sub_block_{idx}"), sub_block
            )
            set_quantification = unc_set.set_as_constraint(
                block=sub_block, uncertain_params=param_var_data_list
            )
            all_cons.extend(set_quantification.uncertainty_cons)
            all_aux_vars.extend(set_quantification.auxiliary_vars)

        return UncertaintyQuantification(
            block=block,
            uncertain_param_vars=param_var_data_list,
            uncertainty_cons=all_cons,
            auxiliary_vars=all_aux_vars,
        )

    def validate(self, config):
        """
        Check IntersectionSet validity.

        Raises
        ------
        ValueError
            If finiteness or nonemptiness checks fail.
        """
        the_sets = self.all_sets

        # validate each set
        for a_set in the_sets:
            a_set.validate(config)

        # check boundedness and nonemptiness of intersected set
        super().validate(config)
