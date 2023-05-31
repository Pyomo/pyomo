"""
Abstract and pre-defined classes for representing uncertainty sets (or
uncertain parameter spaces) of two-stage nonlinear robust optimization
models.

Along with a ``ConcreteModel`` object representing a deterministic model
formulation, an uncertainty set object may be passed to the PyROS solver
to obtain a solution to the model's two-stage robust optimization
counterpart.

Classes
-------
``UncertaintySet``
    Abstract base class for a generic uncertainty set. All other set
    types defined in this module are subclasses.  A user may implement
    their own uncertainty set type as a custom-written subclass.

``EllipsoidalSet``
    A hyperellipsoid.

``AxisAlignedEllipsoidalSet``
    An axis-aligned hyperellipsoid.

``PolyhedralSet``
    A bounded convex polyhedron/polytope.

``BoxSet``
    A hyperrectangle.

``BudgetSet``
    A budget set.

``CardinalitySet``
    A cardinality set (or gamma set).

``DiscreteScenarioSet``
    A discrete set of finitely many points.

``FactorModelSet``
    A factor model set (or net-alpha model set).

``IntersectionSet``
    An intersection of two or more sets, each represented by an
    ``UncertaintySet`` object.
"""


import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum

from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
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
            numeric_types_required = valid_types in valid_num_types
        else:
            numeric_types_required = set(valid_types).issubset(valid_num_types)
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
    Determine whether an array-like (such as a list or Numpy ndarray)
    is ragged.

    NOTE: if Numpy ndarrays are considered to be arr types,
    then zero-dimensional arrays are not considered to be as such.
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


def validate_array(
    arr, arr_name, dim, valid_types, valid_type_desc=None, required_shape=None
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
                raise ValueError(
                    f"Attribute '{arr_name}' should be of shape "
                    f"{req_shape_str}, but detected shape "
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


def uncertainty_sets(obj):
    if not isinstance(obj, UncertaintySet):
        raise ValueError(
            "Expected an UncertaintySet object, instead received %s" % (obj,)
        )
    return obj


def column(matrix, i):
    # Get column i of a given multi-dimensional list
    return [row[i] for row in matrix]


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
        """
        raise NotImplementedError

    def bounding_model(self, config=None):
        """
        Make uncertain parameter value bounding problems (optimize
        value of each uncertain parameter subject to constraints on the
        uncertain parameters).

        Parameters
        ----------
        config : None or ConfigDict, optional
            If a ConfigDict is provided, then it contains
            arguments passed to the PyROS solver.

        Returns
        -------
        model : ConcreteModel
            Bounding problem, with all Objectives deactivated.
        """
        model = ConcreteModel()
        model.util = Block()

        # construct param vars, initialize to nominal point
        model.param_vars = Var(range(self.dim))

        # add constraints
        model.cons = self.set_as_constraint(
            uncertain_params=model.param_vars, model=model, config=config
        )

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
        This check is carried out by solving a sequence of maximization
        and minimization problems (in which the objective for each
        problem is the value of a single uncertain parameter). If any of
        the optimization models cannot be solved successfully to
        optimality, then False is returned.

        This method is invoked during the validation step of a PyROS
        solver call.
        """
        bounding_model = self.bounding_model(config=config)
        solver = config.global_solver

        # initialize uncertain parameter variables
        for param, param_var in zip(
            config.uncertain_params, bounding_model.param_vars.values()
        ):
            param_var.set_value(param.value, skip_validation=True)

        for idx, obj in bounding_model.param_var_objectives.items():
            # activate objective for corresponding dimension
            obj.activate()

            # solve for lower bound, then upper bound
            for sense in (minimize, maximize):
                obj.sense = sense
                res = solver.solve(bounding_model, load_solutions=False, tee=False)

                if not check_optimal_termination(res):
                    return False

            # ensure sense is minimize when done, deactivate
            obj.sense = minimize
            obj.deactivate()

        return True

    def is_nonempty(self, config):
        """
        Return True if the uncertainty set is nonempty, else False.
        """
        return self.is_bounded(config)

    def is_valid(self, config):
        """
        Return True if the uncertainty set is bounded and non-empty,
        else False.
        """
        return self.is_nonempty(config=config) and self.is_bounded(config=config)

    @abc.abstractmethod
    def set_as_constraint(self, **kwargs):
        """
        Construct a (sequence of) mathematical constraint(s)
        (represented by Pyomo `Constraint` objects) on the uncertain
        parameters to represent the uncertainty set for use in a
        two-stage robust optimization problem or subproblem (such as a
        PyROS separation subproblem).

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing, at the very least, a sequence
            of `Param` or `Var` objects representing the uncertain
            parameters of interest, and any additional information
            needed to generate the constraints.
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

        # === Ensure point is of correct dimensionality as the uncertain parameters
        if len(point) != self.dim:
            raise AttributeError(
                "Point must have same dimensions as uncertain parameters."
            )

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
        Specify the numerical bounds for the uncertain parameters
        restricted by the set. Each uncertain parameter is represented
        by a Pyomo `Var` object in a model passed to this method,
        and the numerical bounds are specified by setting the
        `.lb()` and `.ub()` attributes of the `Var` object.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments consisting of a Pyomo `ConfigDict` and a
            Pyomo `ConcreteModel` object, representing a PyROS solver
            configuration and the optimization model of interest.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """
        config = kwargs.pop('config')
        model = kwargs.pop('model')
        _set = config.uncertainty_set
        parameter_bounds = _set.parameter_bounds
        for i, p in enumerate(model.util.uncertain_param_vars.values()):
            p.setlb(parameter_bounds[i][0])
            p.setub(parameter_bounds[i][1])


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
    A hyper-rectangle (a.k.a. "box").

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
            valid_types=valid_num_types,
            valid_type_desc="a valid numeric type",
            required_shape=[None, 2],
        )

        bounds_arr = np.array(val)

        for lb, ub in bounds_arr:
            if lb > ub:
                raise ValueError(f"Lower bound {lb} exceeds upper bound {ub}")

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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of box constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
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
    A cardinality-constrained (a.k.a. "gamma") set.

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
            valid_types=valid_num_types,
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
            valid_types=valid_num_types,
            valid_type_desc="a valid numeric type",
        )

        for dev_val in val:
            if dev_val < 0:
                raise ValueError(
                    f"Entry {dev_val} of attribute 'positive_deviation' "
                    f"is negative value"
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
        validate_arg_type("gamma", val, valid_num_types, "a valid numeric type", False)
        if val < 0 or val > self.dim:
            raise ValueError(
                "Cardinality set attribute "
                f"'gamma' must be a real number between 0 and dimension "
                f"{self.dim} "
                f"(provided value {val})"
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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of cardinality set constraints on
        a sequence of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. This dictionary should consist
            of a `model` entry, which maps to a `ConcreteModel`
            object representing the model of interest (parent model
            of the uncertain parameter objects).

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        # === Ensure dimensions
        if len(uncertain_params) != len(self.origin):
            raise AttributeError(
                "Dimensions of origin and uncertain_param lists must be equal."
            )

        model = kwargs['model']
        set_i = list(range(len(uncertain_params)))
        model.util.cassi = Var(set_i, initialize=0, bounds=(0, 1))

        # Make n equality constraints
        conlist = ConstraintList()
        conlist.construct()
        for i in set_i:
            conlist.add(
                self.origin[i] + self.positive_deviation[i] * model.util.cassi[i]
                == uncertain_params[i]
            )

        conlist.add(sum(model.util.cassi[i] for i in set_i) <= self.gamma)

        return conlist

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
        cassis = []
        for i in range(self.dim):
            if self.positive_deviation[i] > 0:
                cassis.append((point[i] - self.origin[i]) / self.positive_deviation[i])

        if sum(cassi for cassi in cassis) <= self.gamma and all(
            cassi >= 0 and cassi <= 1 for cassi in cassis
        ):
            return True
        else:
            return False


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

        # validate nonemptiness and boundedness here.
        # This check is only performed at construction.
        self._validate()

    def _validate(self):
        """
        Check polyhedral set attributes are such that set is nonempty
        (solve a feasibility problem).

        Raises
        ------
        ValueError
            If set is empty, or the check was not
            successfully completed due to numerical issues.
        """
        # solve LP
        res = sp.optimize.linprog(
            c=np.zeros(self.coefficients_mat.shape[1]),
            A_ub=self.coefficients_mat,
            b_ub=self.rhs_vec,
            method="simplex",
            bounds=(None, None),
        )

        # check termination
        if res.status == 1 or res.status == 4:
            raise ValueError(
                "Could not verify nonemptiness of the "
                "polyhedral set (`scipy.optimize.linprog(method=simplex)` "
                f" status {res.status}) "
            )
        elif res.status == 2:
            raise ValueError(
                "PolyhedralSet defined by 'coefficients_mat' and "
                "'rhs_vec' is empty. Check arguments"
            )

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
            valid_types=valid_num_types,
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

        # check no column is all zeros. otherwise, set is unbounded
        cols_with_all_zeros = np.nonzero(
            [np.all(col == 0) for col in lhs_coeffs_arr.T]
        )[0]
        if cols_with_all_zeros.size > 0:
            col_str = ", ".join(str(val) for val in cols_with_all_zeros)
            raise ValueError(
                "Attempting to set attribute 'coefficients_mat' to value "
                f"with all entries zero in columns at indexes: {col_str}. "
                "Ensure column has at least one nonzero entry"
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
            valid_types=valid_num_types,
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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of polyhedral constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """

        # === Ensure valid dimensions of lhs and rhs w.r.t uncertain_params
        if np.asarray(self.coefficients_mat).shape[1] != len(uncertain_params):
            raise AttributeError(
                "Columns of coefficients_mat matrix "
                "must equal length of uncertain parameters list."
            )

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
        """
        Specify the numerical bounds for each of a sequence of uncertain
        parameters, represented by Pyomo `Var` objects, in a modeling
        object. The numerical bounds are specified through the `.lb()`
        and `.ub()` attributes of the `Var` objects.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest (parent model of the uncertain parameter
            objects for which to specify bounds).
        config : ConfigDict
            PyROS solver config.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """
        add_bounds_for_uncertain_parameters(model=model, config=config)


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
            valid_types=valid_num_types,
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
            valid_types=valid_num_types,
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

        # ensure all entries are nonnegative
        for entry in rhs_vec_arr:
            if entry < 0:
                raise ValueError(
                    f"Entry {entry} of attribute 'budget_rhs_vec' is "
                    "negative. Ensure all entries are nonnegative"
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
            valid_types=valid_num_types,
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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of the constraints defining the budget
        set on a given sequence of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        # === Ensure matrix cols == len uncertain params
        if self.dim != len(uncertain_params):
            raise ValueError(
                f"Argument 'uncertain_params' must contain {self.dim}"
                "Param objects to match BudgetSet dimension"
                f"(provided {len(uncertain_params)} objects)"
            )

        return PolyhedralSet.set_as_constraint(self, uncertain_params)

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        """
        Specify the numerical bounds for each of a sequence of uncertain
        parameters, represented by Pyomo `Var` objects, in a modeling
        object. The numerical bounds are specified through the `.lb()`
        and `.ub()` attributes of the `Var` objects.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest (parent model of the uncertain parameter
            objects for which to specify bounds).
        config : ConfigDict
            PyROS solver config.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """
        # In this case, we use the UncertaintySet class method
        # because we have numerical parameter_bounds
        UncertaintySet.add_bounds_on_uncertain_parameters(model=model, config=config)


class FactorModelSet(UncertaintySet):
    """
    A factor model (a.k.a. "net-alpha" model) set.

    Parameters
    ----------
    origin : (N,) array_like
        Uncertain parameter values around which deviations are
        restrained.
    number_of_factors : int
        Natural number representing the dimensionality of the
        space to which the set projects.
    psi_mat : (N, F) array_like
        Matrix designating each uncertain parameter's contribution to
        each factor.  Each row is associated with a separate uncertain
        parameter.  Each column is associated with a separate factor.
        Number of columns `F` of `psi_mat` should be equal to
        `number_of_factors`.
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
    ...     psi_mat=np.full(shape=(4, 2), fill_value=0.1),
    ...     beta=0.5,
    ... )
    >>> fset.origin
    array([0., 0., 0., 0.])
    >>> fset.number_of_factors
    2
    >>> fset.psi_mat
    array([[0.1, 0.1],
           [0.1, 0.1],
           [0.1, 0.1],
           [0.1, 0.1]])
    >>> fset.beta
    0.5
    """

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
            valid_types=valid_num_types,
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
        int : Natural number representing the dimensionality `F`
        of the space to which the set projects.

        This attribute is immutable, and may only be set at
        object construction. Typically, the number of factors
        is significantly less than the set dimension, but no
        restriction to that end is imposed here.
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
        (N, F) numpy.ndarray : Matrix designating each
        uncertain parameter's contribution to each factor. Each row is
        associated with a separate uncertain parameter. Each column with
        a separate factor.
        """
        return self._psi_mat

    @psi_mat.setter
    def psi_mat(self, val):
        validate_array(
            arr=val,
            arr_name="psi_mat",
            dim=2,
            valid_types=valid_num_types,
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

        # check values acceptable
        for column in psi_mat_arr.T:
            if np.allclose(column, 0):
                raise ValueError(
                    "Each column of attribute 'psi_mat' should have at least "
                    "one nonzero entry"
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
        ``[origin - psi @ np.ones(F), origin + psi @ np.ones(F)].T``.
        """
        return self._beta

    @beta.setter
    def beta(self, val):
        if val > 1 or val < 0:
            raise ValueError(
                "Beta parameter must be a real number between 0 "
                f"and 1 inclusive (provided value {val})"
            )

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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of factor model constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. This dictionary should consist
            of a `model` entry, which maps to a `ConcreteModel`
            object representing the model of interest (parent model
            of the uncertain parameter objects).

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        model = kwargs['model']

        # === Ensure dimensions
        if len(uncertain_params) != len(self.origin):
            raise AttributeError(
                "Dimensions of origin and uncertain_param lists must be equal."
            )

        # Make F-dim cassi variable
        n = list(range(self.number_of_factors))
        model.util.cassi = Var(n, initialize=0, bounds=(-1, 1))

        conlist = ConstraintList()
        conlist.construct()

        disturbances = [
            sum(self.psi_mat[i][j] * model.util.cassi[j] for j in n)
            for i in range(len(uncertain_params))
        ]

        # Make n equality constraints
        for i in range(len(uncertain_params)):
            conlist.add(self.origin[i] + disturbances[i] == uncertain_params[i])
        conlist.add(
            sum(model.util.cassi[i] for i in n) <= +self.beta * self.number_of_factors
        )
        conlist.add(
            sum(model.util.cassi[i] for i in n) >= -self.beta * self.number_of_factors
        )
        return conlist

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
        inv_psi = np.linalg.pinv(self.psi_mat)
        diff = np.asarray(list(point[i] - self.origin[i] for i in range(len(point))))
        cassis = np.dot(inv_psi, np.transpose(diff))

        if abs(
            sum(cassi for cassi in cassis)
        ) <= self.beta * self.number_of_factors and all(
            cassi >= -1 and cassi <= 1 for cassi in cassis
        ):
            return True
        else:
            return False


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
            valid_types=valid_num_types,
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
            valid_types=valid_num_types,
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

        # ensure half-lengths are non-negative
        for half_len in val_arr:
            if half_len < 0:
                raise ValueError(
                    f"Entry {half_len} of 'half_lengths' "
                    "is negative. All half-lengths must be nonnegative"
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

    def set_as_constraint(self, uncertain_params, model=None, config=None):
        """
        Construct a list of ellipsoidal constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : {IndexedParam, IndexedVar, list of Param/Var}
            Uncertain parameter objects upon which the constraints
            are imposed. Indexed parameters are accepted, and
            are unpacked for constraint generation.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
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
    A general ellipsoid.

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
        matrix). The default is `1`.

    Examples
    --------
    3D origin-centered unit hypersphere:

    >>> from pyomo.contrib.pyros import EllipsoidalSet
    >>> import numpy as np
    >>> hypersphere = EllipsoidalSet(
    ...     center=[0, 0, 0],
    ...     shape_matrix=np.eye(3),
    ...     scale=1,
    ... )
    >>> hypersphere.center
    array([0, 0, 0])
    >>> hypersphere.shape_matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> hypersphere.scale
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

    """

    def __init__(self, center, shape_matrix, scale=1):
        """Initialize self (see class docstring)."""
        self.center = center
        self.shape_matrix = shape_matrix
        self.scale = scale

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
            valid_types=valid_num_types,
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
            valid_types=valid_num_types,
            valid_type_desc="a valid numeric type",
            required_shape=None,
        )

        shape_mat_arr = np.array(val)

        # check matrix shape matches set dimension
        if hasattr(self, "_center"):
            if not all(size == self.dim for size in shape_mat_arr.shape):
                raise ValueError(
                    f"EllipsoidalSet attribute 'shape_matrix' "
                    f"must be a square matrix of size "
                    f"{self.dim} to match set dimension "
                    f"(provided matrix with shape {shape_mat_arr.shape})"
                )

        self._verify_positive_definite(shape_mat_arr)
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
        validate_arg_type("scale", val, valid_num_types, "a valid numeric type", False)
        if val < 0:
            raise ValueError(
                "EllipsoidalSet attribute "
                f"'scale' must be a non-negative real "
                f"(provided value {val})"
            )

        self._scale = val

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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of ellipsoidal constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : {IndexedParam, IndexedVar, list of Param/Var}
            Uncertain parameter objects upon which the constraints
            are imposed. Indexed parameters are accepted, and
            are unpacked for constraint generation.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        inv_covar = np.linalg.inv(self.shape_matrix)

        if len(uncertain_params) != len(self.center):
            raise AttributeError(
                "Center of ellipsoid must be same dimensions as vector of uncertain parameters."
            )

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
        product1 = [
            sum([x * y for x, y in zip(diff, column(inv_covar, i))])
            for i in range(len(inv_covar))
        ]
        constraint = sum([x * y for x, y in zip(product1, diff)])

        conlist = ConstraintList()
        conlist.construct()
        conlist.add(constraint <= self.scale)
        return conlist


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
            valid_types=valid_num_types,
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

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        # === Ensure point is of correct dimensionality as the uncertain parameters
        dim = len(uncertain_params)
        if any(len(d) != dim for d in self.scenarios):
            raise AttributeError(
                "All scenarios must have same dimensions as uncertain parameters."
            )

        conlist = ConstraintList()
        conlist.construct()

        for n in list(range(len(self.scenarios))):
            for i in list(range(len(uncertain_params))):
                conlist.add(uncertain_params[i] == self.scenarios[n][i])

        conlist.deactivate()
        return conlist

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
        # Round all double precision to a tolerance
        num_decimals = 8
        rounded_scenarios = list(
            list(round(num, num_decimals) for num in d) for d in self.scenarios
        )
        rounded_point = list(round(num, num_decimals) for num in point)

        return any(rounded_point == rounded_d for rounded_d in rounded_scenarios)


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
    >>> intersection.all_sets
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

    def is_empty_intersection(self, uncertain_params, nlp_solver):
        """
        Determine if intersection is empty.

        Arguments
        ---------
        uncertain_params : list of Param or list of Var
            List of uncertain parameter objects.
        nlp_solver : Pyomo SolverFactory object
            NLP solver.

        Returns
        -------
        is_empty_intersection : bool
            True if the intersection is certified to be empty,
            and False otherwise.
        """

        # === Non-emptiness check for the set intersection
        is_empty_intersection = True
        if any(a_set.type == "discrete" for a_set in self.all_sets):
            disc_sets = (a_set for a_set in self.all_sets if a_set.type == "discrete")
            disc_set = min(
                disc_sets, key=lambda x: len(x.scenarios)
            )  # minimum set of scenarios
            # === Ensure there is at least one scenario from this discrete set which is a member of all other sets
            for scenario in disc_set.scenarios:
                if all(a_set.point_in_set(point=scenario) for a_set in self.all_sets):
                    is_empty_intersection = False
                    break
        else:
            # === Compile constraints and solve NLP
            m = ConcreteModel()
            m.obj = Objective(expr=0)  # dummy objective required if using baron
            m.param_vars = Var(uncertain_params.index_set())
            for a_set in self.all_sets:
                m.add_component(
                    a_set.type + "_constraints",
                    a_set.set_as_constraint(uncertain_params=m.param_vars),
                )
            try:
                res = nlp_solver.solve(m)
            except:
                raise ValueError(
                    "Solver terminated with an error while checking set intersection non-emptiness."
                )
            if check_optimal_termination(res):
                is_empty_intersection = False
        return is_empty_intersection

    # === Define pairwise intersection function
    @staticmethod
    def intersect(Q1, Q2):
        """
        Obtain the intersection of two uncertainty sets.

        Parameters
        ----------
        Q1, Q2 : UncertaintySet
            Operand uncertainty sets.

        Returns
        -------
        : DiscreteScenarioSet or IntersectionSet
            Intersection of the sets. A `DiscreteScenarioSet` is
            returned if both operand sets are `DiscreteScenarioSet`
            instances; otherwise, an `IntersectionSet` is returned.
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
        Construct a list of constraints on a given sequence
        of uncertain parameter objects. In advance of constructing
        the constraints, a check is performed to determine whether
        the set is empty.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. Must contain a `config` entry,
            which maps to a `ConfigDict` containing an entry
            entitled `global_solver`. The `global_solver`
            key maps to an NLP solver, purportedly with global
            optimization capabilities.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.

        Raises
        ------
        AttributeError
            If the intersection set is found to be empty.
        """
        try:
            nlp_solver = kwargs["config"].global_solver
        except:
            raise AttributeError(
                "set_as_constraint for SetIntersection requires access to an NLP solver via"
                "the PyROS Solver config."
            )
        is_empty_intersection = self.is_empty_intersection(
            uncertain_params=uncertain_params, nlp_solver=nlp_solver
        )

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
                    for con in list(
                        set.set_as_constraint(
                            uncertain_params=uncertain_params
                        ).values()
                    ):
                        conlist.add(con.expr)
                return conlist
        else:
            raise AttributeError(
                "Set intersection is empty, cannot proceed with PyROS robust optimization."
            )

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        """
        Specify the numerical bounds for each of a sequence of uncertain
        parameters, represented by Pyomo `Var` objects, in a modeling
        object. The numerical bounds are specified through the `.lb()`
        and `.ub()` attributes of the `Var` objects.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest (parent model of the uncertain parameter
            objects for which to specify bounds).
        config : ConfigDict
            PyROS solver config.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """

        add_bounds_for_uncertain_parameters(model=model, config=config)
        return
