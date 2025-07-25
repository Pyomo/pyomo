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

import collections
import functools
import inspect
import itertools

from collections.abc import Sequence
from collections.abc import Mapping

from pyomo.common.autoslots import AutoSlots
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject

initializer_map = {}
sequence_types = set()
# initialize with function, method, and method-wrapper types.
function_types = set(
    [
        type(PyomoObject.is_expression_type),
        type(PyomoObject().is_expression_type),
        type(PyomoObject.is_expression_type.__call__),
    ]
)


def Initializer(
    arg,
    allow_generators=False,
    treat_sequences_as_mappings=True,
    arg_not_specified=None,
    additional_args=0,
):
    """Standardized processing of Component keyword arguments

    Component keyword arguments accept a number of possible inputs, from
    scalars to dictionaries, to functions (rules) and generators.  This
    function standardizes the processing of keyword arguments and
    returns "initializer classes" that are specialized to the specific
    data type provided.

    Parameters
    ----------
    arg:

        The argument passed to the component constructor.  This could
        be almost any type, including a scalar, dict, list, function,
        generator, or None.

    allow_generators: bool

        If False, then we will raise an exception if ``arg`` is a generator

    treat_sequences_as_mappings: bool

        If True, then if ``arg`` is a sequence, we will treat it as if
        it were a mapping (i.e., ``dict(enumerate(arg))``).  Otherwise
        sequences will be returned back as the value of the initializer.

    arg_not_specified:

        If ``arg`` is ``arg_not_specified``, then the function will
        return None (and not an InitializerBase object).

    additional_args: int

        The number of additional arguments that will be passed to any
        function calls (provided *before* the index value).

    """
    if arg is arg_not_specified:
        return None
    if additional_args:
        if arg.__class__ in function_types:
            if allow_generators or inspect.isgeneratorfunction(arg):
                raise ValueError(
                    "Generator functions are not allowed when passing additional args"
                )
            _args = inspect.getfullargspec(arg)
            _nargs = len(_args.args)
            if inspect.ismethod(arg) and arg.__self__ is not None:
                # Ignore 'self' for bound instance methods and 'cls' for
                # @classmethods
                _nargs -= 1
            if _nargs == 1 + additional_args and _args.varargs is None:
                return ParameterizedScalarCallInitializer(arg, constant=True)
            else:
                return ParameterizedIndexedCallInitializer(arg)
        else:
            base_initializer = Initializer(
                arg=arg,
                allow_generators=allow_generators,
                treat_sequences_as_mappings=treat_sequences_as_mappings,
                arg_not_specified=arg_not_specified,
            )
            if type(base_initializer) in (
                ScalarCallInitializer,
                IndexedCallInitializer,
            ):
                # This is an edge case: if we are providing additional
                # args, but this is the first time we are seeing a
                # callable type, we will (potentially) incorrectly
                # categorize this as an IndexedCallInitializer.  Re-try
                # now that we know this is a function_type.
                return Initializer(
                    arg=base_initializer._fcn,
                    allow_generators=allow_generators,
                    treat_sequences_as_mappings=treat_sequences_as_mappings,
                    arg_not_specified=arg_not_specified,
                    additional_args=additional_args,
                )
            return ParameterizedInitializer(base_initializer)
    if arg.__class__ in initializer_map:
        return initializer_map[arg.__class__](arg)
    if arg.__class__ in sequence_types:
        if treat_sequences_as_mappings:
            return ItemInitializer(arg)
        else:
            return ConstantInitializer(arg)
    if arg.__class__ in function_types:
        # Note: we do not use "inspect.isfunction or inspect.ismethod"
        # because some function-like things (notably cythonized
        # functions) return False
        if not allow_generators and inspect.isgeneratorfunction(arg):
            raise ValueError("Generator functions are not allowed")
        # Historically pyomo.core.base.misc.apply_indexed_rule
        # accepted rules that took only the parent block (even for
        # indexed components).  We will preserve that functionality
        # here.
        #
        # I was concerned that some builtins aren't compatible with
        # getfullargspec (and would need the same try-except logic as in
        # the partial handling), but I have been unable to come up with
        # an example.  The closest was getattr(), but that falls back on
        # getattr.__call__, which does support getfullargspec.
        _args = inspect.getfullargspec(arg)
        _nargs = len(_args.args)
        if inspect.ismethod(arg) and arg.__self__ is not None:
            # Ignore 'self' for bound instance methods and 'cls' for
            # @classmethods
            _nargs -= 1
        if _nargs == 1 and _args.varargs is None:
            return ScalarCallInitializer(
                arg, constant=not inspect.isgeneratorfunction(arg)
            )
        else:
            return IndexedCallInitializer(arg)
    if hasattr(arg, '__len__'):
        if isinstance(arg, Mapping):
            initializer_map[arg.__class__] = ItemInitializer
        elif isinstance(arg, Sequence) and not isinstance(arg, str):
            sequence_types.add(arg.__class__)
        elif isinstance(arg, PyomoObject):
            # TODO: Should IndexedComponent inherit from
            # collections.abc.Mapping?
            if arg.is_component_type() and arg.is_indexed():
                initializer_map[arg.__class__] = ItemInitializer
            else:
                initializer_map[arg.__class__] = ConstantInitializer
        elif any(c.__name__ == 'ndarray' for c in arg.__class__.__mro__):
            if numpy_available and isinstance(arg, numpy.ndarray):
                sequence_types.add(arg.__class__)
        elif any(c.__name__ == 'Series' for c in arg.__class__.__mro__):
            if pandas_available and isinstance(arg, pandas.Series):
                sequence_types.add(arg.__class__)
        elif any(c.__name__ == 'DataFrame' for c in arg.__class__.__mro__):
            if pandas_available and isinstance(arg, pandas.DataFrame):
                initializer_map[arg.__class__] = DataFrameInitializer
        else:
            # Note: this picks up (among other things) all string instances
            initializer_map[arg.__class__] = ConstantInitializer
        # recursively call Initializer to pick up the new registration
        return Initializer(
            arg,
            allow_generators=allow_generators,
            treat_sequences_as_mappings=treat_sequences_as_mappings,
            arg_not_specified=arg_not_specified,
        )
    if inspect.isgenerator(arg) or hasattr(arg, 'next') or hasattr(arg, '__next__'):
        # This catches generators and iterators (like enumerate()), but
        # skips "reusable" iterators like range() as well as Pyomo
        # (finite) Set objects [they were both caught by the
        # "hasattr('__len__')" above]
        if not allow_generators:
            raise ValueError("Generators are not allowed")
        # Deepcopying generators is problematic (e.g., it generates a
        # segfault in pypy3 7.3.0).  We will immediately expand the
        # generator into a tuple and then store it as a constant.
        return ConstantInitializer(tuple(arg))
    if type(arg) is functools.partial:
        try:
            _args = inspect.getfullargspec(arg.func)
        except:
            # Inspect doesn't work for some built-in callables (notably
            # 'int').  We will just have to assume this is a "normal"
            # IndexedCallInitializer
            return IndexedCallInitializer(arg)
        _positional_args = set(_args.args)
        for key in arg.keywords:
            _positional_args.discard(key)
        if len(_positional_args) - len(arg.args) == 1 and _args.varargs is None:
            return ScalarCallInitializer(arg)
        else:
            return IndexedCallInitializer(arg)
    if isinstance(arg, InitializerBase):
        return arg
    if isinstance(arg, PyomoObject):
        # We re-check for PyomoObject here, as that picks up / caches
        # non-components like component data objects and expressions
        initializer_map[arg.__class__] = ConstantInitializer
        return ConstantInitializer(arg)
    if callable(arg) and not isinstance(arg, type):
        # We assume any callable thing could be a functor; but, we must
        # filter out types, as we use types as special identifiers that
        # should not be called (e.g., UnknownSetDimen)
        if inspect.isfunction(arg) or inspect.ismethod(arg):
            # Add this to the set of known function types and try again
            function_types.add(type(arg))
        else:
            # Try again, but use the __call__ method (for supporting
            # things like functors and cythonized functions).  __call__
            # is almost certainly going to be a method-wrapper
            arg = arg.__call__
        return Initializer(
            arg,
            allow_generators=allow_generators,
            treat_sequences_as_mappings=treat_sequences_as_mappings,
            arg_not_specified=arg_not_specified,
        )
    initializer_map[arg.__class__] = ConstantInitializer
    return ConstantInitializer(arg)


class InitializerBase(AutoSlots.Mixin, object):
    """Base class for all Initializer objects"""

    __slots__ = ()

    verified = False

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return False

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return False

    def indices(self):
        """Return a generator over the embedded indices

        This will raise a RuntimeError if this initializer does not
        contain embedded indices
        """
        raise RuntimeError(
            "Initializer %s does not contain embedded indices" % (type(self).__name__,)
        )


class ConstantInitializer(InitializerBase):
    """Initializer for constant values"""

    __slots__ = ('val', 'verified')

    def __init__(self, val):
        self.val = val
        self.verified = False

    def __call__(self, parent, idx):
        return self.val

    def constant(self):
        return True


class ItemInitializer(InitializerBase):
    """Initializer for dict-like values supporting __getitem__()"""

    __slots__ = ('_dict',)

    def __init__(self, _dict):
        self._dict = _dict

    def __call__(self, parent, idx):
        return self._dict[idx]

    def contains_indices(self):
        return True

    def indices(self):
        try:
            return self._dict.keys()
        except AttributeError:
            return range(len(self._dict))


class DataFrameInitializer(InitializerBase):
    """Initializer for pandas DataFrame values"""

    __slots__ = ('_df', '_column')

    def __init__(self, dataframe, column=None):
        self._df = dataframe
        if column is not None:
            self._column = column
        elif len(dataframe.columns) == 1:
            self._column = dataframe.columns[0]
        else:
            self._column = None

    def __call__(self, parent, idx):
        if self._column is None:
            return self._df.at[idx]
        return self._df.at[idx, self._column]

    def contains_indices(self):
        return True

    def indices(self):
        if self._column is None:
            return itertools.product(self._df.index, self._df)
        return self._df.index


class IndexedCallInitializer(InitializerBase):
    """Initializer for functions and callable objects"""

    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx, **kwargs):
        # Note: this is called by a component using data from a Set (so
        # any tuple-like type should have already been checked and
        # converted to a tuple; or flattening is turned off and it is
        # the user's responsibility to sort things out.
        if idx.__class__ is tuple:
            return self._fcn(parent, *idx, **kwargs)
        else:
            return self._fcn(parent, idx, **kwargs)


class ParameterizedIndexedCallInitializer(IndexedCallInitializer):
    """IndexedCallInitializer that accepts additional arguments"""

    __slots__ = ()

    def __call__(self, parent, idx, *args, **kwargs):
        if idx.__class__ is tuple:
            return self._fcn(parent, *args, *idx, **kwargs)
        else:
            return self._fcn(parent, *args, idx, **kwargs)


class CountedCallGenerator(object):
    """Generator implementing the "counted call" initialization scheme

    This generator implements the older "counted call" scheme, where the
    first argument past the parent block is a monotonically-increasing
    integer beginning at `start_at`.
    """

    def __init__(self, ctype, fcn, scalar, parent, idx, start_at):
        # Note: this is called by a component using data from a Set (so
        # any tuple-like type should have already been checked and
        # converted to a tuple; or flattening is turned off and it is
        # the user's responsibility to sort things out.
        self._count = start_at - 1
        if scalar:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c))
        elif idx.__class__ is tuple:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c, *idx))
        else:
            self._fcn = lambda c: self._filter(ctype, fcn(parent, c, idx))

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        return self._fcn(self._count)

    next = __next__

    @staticmethod
    def _filter(ctype, x):
        if x is None:
            raise ValueError(
                """Counted %s rule returned None instead of %s.End.
    Counted %s rules of the form fcn(model, count, *idx) will be called
    repeatedly with an increasing count parameter until the rule returns
    %s.End.  None is not a valid return value in this case due to the
    likelihood that an error in the rule can incorrectly return None."""
                % ((ctype.__name__,) * 4)
            )
        return x


class CountedCallInitializer(InitializerBase):
    """Initializer for functions implementing the "counted call" API."""

    # Pyomo has a historical feature for some rules, where the number of
    # times[*1] the rule was called could be passed as an additional
    # argument between the block and the index.  This was primarily
    # supported by Set and ConstraintList.  There were many issues with
    # the syntax, including inconsistent support for jagged (dimen=None)
    # indexing sets, inconsistent support for *args rules, and a likely
    # infinite loop if the rule returned Constraint.Skip.
    #
    # As a slight departure from previous implementations, we will ONLY
    # allow the counted rule syntax when the rule does NOT use *args
    #
    # [*1] The extra argument was one-based, and was only incremented
    #     when a valid value was returned by the rule and added to the
    #     _data.  This was fragile, as returning something like
    #     {Component}.Skip could result in an infinite loop.  This
    #     implementation deviates from that behavior and increments the
    #     counter every time the rule is called.
    #
    # [JDS 6/2019] We will support a slightly restricted but more
    # consistent form of the original implementation for backwards
    # compatibility, but I believe that we should deprecate this syntax
    # entirely.
    __slots__ = ('_fcn', '_is_counted_rule', '_scalar', '_ctype', '_start')

    def __init__(self, obj, _indexed_init, starting_index=1):
        self._fcn = _indexed_init._fcn
        self._is_counted_rule = None
        self._scalar = not obj.is_indexed()
        self._ctype = obj.ctype
        self._start = starting_index
        if self._scalar:
            self._is_counted_rule = True

    def __call__(self, parent, idx):
        # Note: this is called by a component using data from a Set (so
        # any tuple-like type should have already been checked and
        # converted to a tuple; or flattening is turned off and it is
        # the user's responsibility to sort things out.
        if self._is_counted_rule == False:
            if idx.__class__ is tuple:
                return self._fcn(parent, *idx)
            else:
                return self._fcn(parent, idx)
        if self._is_counted_rule == True:
            return CountedCallGenerator(
                self._ctype, self._fcn, self._scalar, parent, idx, self._start
            )

        # Note that this code will only be called once, and only if
        # the object is not a scalar.
        _args = inspect.getfullargspec(self._fcn)
        _nargs = len(_args.args)
        if inspect.ismethod(self._fcn) and self._fcn.__self__ is not None:
            _nargs -= 1
        _len = len(idx) if idx.__class__ is tuple else 1
        if _len + 2 == _nargs:
            self._is_counted_rule = True
        else:
            self._is_counted_rule = False
        return self.__call__(parent, idx)


class ScalarCallInitializer(InitializerBase):
    """Initializer for functions taking only the parent block argument."""

    __slots__ = ('_fcn', '_constant')

    def __init__(self, _fcn, constant=True):
        self._fcn = _fcn
        self._constant = constant

    def __call__(self, parent, idx, **kwargs):
        return self._fcn(parent, **kwargs)

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._constant


class ParameterizedScalarCallInitializer(ScalarCallInitializer):
    """ScalarCallInitializer that accepts additional arguments"""

    __slots__ = ()

    def __call__(self, parent, idx, *args, **kwargs):
        return self._fcn(parent, *args, **kwargs)


class DefaultInitializer(InitializerBase):
    """Initializer wrapper that maps exceptions to default values.


    Parameters
    ----------
    initializer: :py:class`InitializerBase`
        the Initializer instance to wrap

    default:
        the value to return inlieu of the caught exception(s)

    exceptions: Exception or tuple
        the single Exception or tuple of Exceptions to catch and return
        the default value.

    """

    __slots__ = ('_initializer', '_default', '_exceptions')

    def __init__(self, initializer, default, exceptions):
        self._initializer = initializer
        self._default = default
        self._exceptions = exceptions

    def __call__(self, parent, index, **kwargs):
        try:
            return self._initializer(parent, index, **kwargs)
        except self._exceptions:
            return self._default

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._initializer.constant()

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return self._initializer.contains_indices()

    def indices(self):
        return self._initializer.indices()


class ParameterizedInitializer(InitializerBase):
    """Wrapper to provide additional positional arguments to Initializer objects"""

    __slots__ = ('_base_initializer',)

    def __init__(self, base):
        self._base_initializer = base

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._base_initializer.constant()

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return self._base_initializer.contains_indices()

    def indices(self):
        """Return a generator over the embedded indices

        This will raise a RuntimeError if this initializer does not
        contain embedded indices
        """
        return self._base_initializer.indices()

    def __call__(self, parent, idx, *args, **kwargs):
        return self._base_initializer(parent, idx)(parent, *args, **kwargs)


class PartialInitializer(InitializerBase):
    """Partial wrapper of an InitializerBase that supplies additional arguments"""

    __slots__ = ('_fcn',)

    def __init__(self, _fcn, *args, **kwargs):
        self._fcn = functools.partial(_fcn, *args, **kwargs)

    def constant(self):
        return self._fcn.func.constant()

    def contains_indices(self):
        return self._fcn.func.contains_indices()

    def indices(self):
        return self._fcn.func.indices()

    def __call__(self, parent, idx, *args, **kwargs):
        # Note that the Initializer.__call__ API is different from the
        # rule API.  As a result, we cannot just inherit from
        # IndexedCallInitializer and must instead implement our own
        # __call__ here.
        return self._fcn(parent, idx, *args, **kwargs)


_bound_sequence_types = collections.defaultdict(None.__class__)


class BoundInitializer(InitializerBase):
    """Initializer wrapper for processing bounds (mapping scalars to 2-tuples)

    Note that this class is meant to mimic the behavior of
    :py:func:`Initializer` and will return ``None`` if the initializer
    that it is wrapping is ``None``.

    Parameters
    ----------
    arg:

        As with :py:func:`Initializer`, this is the raw argument passed
        to the component constructor.

    obj: :py:class:`Component`

        The component that "owns" the initializer.  This initializer
        will treat sequences as mappings only if the owning component is
        indexed and the sequence passed to the initializer is not of
        length 2

    """

    __slots__ = ('_initializer',)

    def __new__(cls, arg=None, obj=NOTSET):
        # The Initializer() function returns None if the initializer is
        # None.  We will mock that behavior by commandeering __new__()
        if arg is None and obj is not NOTSET:
            return None
        else:
            return super().__new__(cls)

    def __init__(self, arg, obj=NOTSET):
        if obj is NOTSET or obj.is_indexed():
            treat_sequences_as_mappings = not (
                isinstance(arg, Sequence)
                and len(arg) == 2
                and not isinstance(arg[0], Sequence)
            )
        else:
            treat_sequences_as_mappings = False
        self._initializer = Initializer(
            arg, treat_sequences_as_mappings=treat_sequences_as_mappings
        )

    def __call__(self, parent, index, **kwargs):
        val = self._initializer(parent, index, **kwargs)
        if _bound_sequence_types[val.__class__]:
            return val
        if _bound_sequence_types[val.__class__] is None:
            _bound_sequence_types[val.__class__] = isinstance(
                val, Sequence
            ) and not isinstance(val, str)
            if _bound_sequence_types[val.__class__]:
                return val
        return (val, val)

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._initializer.constant()

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return self._initializer.contains_indices()

    def indices(self):
        return self._initializer.indices()
