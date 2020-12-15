#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Utility functions
#
import functools
import inspect
import six

from six import iteritems, iterkeys
from six.moves import xrange

if six.PY2:
    getargspec = inspect.getargspec
    from collections import Sequence as collections_Sequence
    from collections import Mapping as collections_Mapping
else:
    # For our needs, getfullargspec is a drop-in replacement for
    # getargspec (which was removed in Python 3.x)
    getargspec = inspect.getfullargspec
    from collections.abc import Sequence as collections_Sequence
    from collections.abc import Mapping as collections_Mapping


from pyomo.common import DeveloperError
from pyomo.core.expr.numvalue import (
    native_types,
)

def is_functor(obj):
    """
    Returns true iff obj.__call__ is defined.
    """
    return inspect.isfunction(obj) or hasattr(obj,'__call__')


#
# The following decorator is general and should probably be promoted to
# component.py so that we can efficiently handle construction errors on
# scalar components.
#
# TODO: quantify the memory overhead here.  We create (and preserve) a
# locals() dict for *each* method that we wrap.  If that becomes
# significant, we might consider using a single global private
# environment (which would require some thought when managing any
# potential name collisions)
#
def _disable_method(fcn, msg=None):
    _name = fcn.__name__
    if msg is None:
        msg = 'access %s on' % (_name,)

    # functools.wraps doesn't preserve the function signature until
    # Python 3.4, and even then, does not preserve it accurately (e.g.,
    # calling with the incorreect number of arguments does not generate
    # an error).  For backwards compatability with Python 2.x, we will
    # create a temporary (local) function using exec that matches the
    # function signature passed in and raises an exception
    if six.PY2:
        args = str(inspect.formatargspec(*getargspec(fcn)))
    else:
        args = str(inspect.signature(fcn))
    assert args == '(self)' or args.startswith('(self,')

    # lambda comes through with a function name "<lambda>".  We will
    # use exec here to create a function (in a private namespace)
    # that will have the correct name.
    _env = {}
    _funcdef = """def %s%s:
        raise RuntimeError(
            "Cannot %s %%s '%%s' before it has been constructed (initialized)."
            %% (type(self).__name__, self.name))
""" % (_name, args, msg,)
    exec(_funcdef, _env)
    return functools.wraps(fcn)(_env[_name])


def _disable_property(fcn, msg=None):
    if msg is None:
        _gmsg = 'access property %s on' % (fcn.fget.__name__,)
    else:
        _gmsg = msg
    def getter(self, *args, **kwds):
        raise RuntimeError(
            "Cannot %s %s '%s' before it has been constructed (initialized)."
            % (_gmsg, type(self).__name__, self.name))

    if msg is None:
        _smsg = 'set property %s on' % (fcn.fget.__name__,)
    else:
        _smsg = msg
    def setter(self, *args, **kwds):
        raise RuntimeError(
            "Cannot %s %s '%s' before it has been constructed (initialized)."
            % (_smsg, type(self).__name__, self.name))

    return property(fget=getter, fset=setter, doc=fcn.__doc__)

def disable_methods(methods):
    """Class decorator to disable methods before construct is called.

    This decorator should be called to create "Abstract" scalar classes
    that override key methods to raise exceptions.  When the construct()
    method is called, the class instance changes type back to the
    original scalar component and the full class functionality is
    restored.  This prevents most class methods from having to begin with
    "`if not self.parent_component()._constructed: raise RuntimeError`"
    """
    def class_decorator(cls):
        assert(len(cls.__bases__) == 1)
        base = cls.__bases__[0]

        def construct(self, data=None):
            if hasattr(self, '_name') and self._name == self.__class__.__name__:
                self._name = base.__name__
            self.__class__ = base
            return base.construct(self, data)
        construct.__doc__ = base.construct.__doc__
        cls.construct = construct

        for method in methods:
            if type(method) is tuple:
                method, msg = method
            else:
                msg = None
            if not hasattr(base, method):
                raise DeveloperError(
                    "Cannot disable method %s on %s: not present on base class"
                    % (method, cls))
            base_method = getattr(base, method)
            if type(base_method) is property:
                setattr(cls, method, _disable_property(base_method, msg))
            else:
                setattr(cls, method, _disable_method(base_method, msg))
        return cls

    return class_decorator


#
# The following set of "Initializer" classes are a general functionality
# and should be promoted to their own module so that we can use them on
# all Components to standardize how we process component arguments.
#
def Initializer(init,
                allow_generators=False,
                treat_sequences_as_mappings=True,
                arg_not_specified=None):
    """Standardized processing of Component keyword arguments

    Component keyword arguments accept a number of possible inputs, from
    scalars to dictionaries, to functions (rules) and generators.  This
    function standardizes the processing of keyword arguments and
    returns "initializer classes" that are specialized to the specific
    data type provided.
    """
    if init.__class__ in native_types:
        if init is arg_not_specified:
            return None
        return ConstantInitializer(init)
    elif inspect.isfunction(init) or inspect.ismethod(init):
        if not allow_generators and inspect.isgeneratorfunction(init):
            raise ValueError("Generator functions are not allowed")
        # Historically pyomo.core.base.misc.apply_indexed_rule
        # accepted rules that took only the parent block (even for
        # indexed components).  We will preserve that functionality
        # here.
        _args = getargspec(init)
        _nargs = len(_args.args)
        if inspect.ismethod(init) and init.__self__ is not None:
            # Ignore 'self' for bound instance methods and 'cls' for
            # @classmethods
            _nargs -= 1
        if _nargs == 1 and _args.varargs is None:
            return ScalarCallInitializer(init)
        else:
            return IndexedCallInitializer(init)
    elif isinstance(init, collections_Mapping):
        return ItemInitializer(init)
    elif isinstance(init, collections_Sequence) \
            and not isinstance(init, six.string_types):
        if treat_sequences_as_mappings:
            return ItemInitializer(init)
        else:
            return ConstantInitializer(init)
    elif inspect.isgenerator(init) or (
            ( hasattr(init, 'next') or hasattr(init, '__next__') )
              and not hasattr(init, '__len__')):
        # This catches generators and iterators (like enumerate()), but
        # skips "reusable" iterators like range() as well as Pyomo
        # (finite) Set objects.
        if not allow_generators:
            raise ValueError("Generators are not allowed")
        # Deepcopying generators is problematic (e.g., it generates a
        # segfault in pypy3 7.3.0).  We will immediately expand the
        # generator into a tuple and then store it as a constant.
        return ConstantInitializer(tuple(init))
    elif type(init) is functools.partial:
        _args = getargspec(init.func)
        if len(_args.args) - len(init.args) == 1 and _args.varargs is None:
            return ScalarCallInitializer(init)
        else:
            return IndexedCallInitializer(init)
    else:
        return ConstantInitializer(init)


class InitializerBase(object):
    """Base class for all Initializer objects"""
    __slots__ = ()

    verified = False

    def __getstate__(self):
        """Class serializer

        This class must declare __getstate__ because it is slotized.
        This implementation should be sufficient for simple derived
        classes (where __slots__ are only declared on the most derived
        class).
        """
        return {k:getattr(self,k) for k in self.__slots__}

    def __setstate__(self, state):
        for key, val in iteritems(state):
            object.__setattr__(self, key, val)

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
        raise RuntimeError("Initializer %s does not contain embedded indices"
                           % (type(self).__name__,))


class ConstantInitializer(InitializerBase):
    """Initializer for constant values"""
    __slots__ = ('val','verified')

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
            return iterkeys(self._dict)
        except AttributeError:
            return xrange(len(self._dict))


class IndexedCallInitializer(InitializerBase):
    """Initializer for functions and callable objects"""
    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx):
        # Note: this is called by a component using data from a Set (so
        # any tuple-like type should have already been checked and
        # converted to a tuple; or flattening is turned off and it is
        # the user's responsibility to sort things out.
        if idx.__class__ is tuple:
            return self._fcn(parent, *idx)
        else:
            return self._fcn(parent, idx)



class CountedCallGenerator(object):
    """Generator implementing the "counted call" initialization scheme

    This generator implements the older "counted call" scheme, where the
    first argument past the parent block is a monotonically-increasing
    integer beginning at 1.
    """
    def __init__(self, ctype, fcn, scalar, parent, idx):
        # Note: this is called by a component using data from a Set (so
        # any tuple-like type should have already been checked and
        # converted to a tuple; or flattening is turned off and it is
        # the user's responsibility to sort things out.
        self._count = 0
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
                % ((ctype.__name__,)*4))
        return x


class CountedCallInitializer(InitializerBase):
    """Initializer for functions implementing the "counted call" API.
    """
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
    # compatability, but I believe that we should deprecate this syntax
    # entirely.
    __slots__ = ('_fcn','_is_counted_rule', '_scalar','_ctype')

    def __init__(self, obj, _indexed_init):
        self._fcn = _indexed_init._fcn
        self._is_counted_rule = None
        self._scalar = not obj.is_indexed()
        self._ctype = obj.ctype
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
                self._ctype, self._fcn, self._scalar, parent, idx)

        # Note that this code will only be called once, and only if
        # the object is not a scalar.
        _args = getargspec(self._fcn)
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
    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx):
        return self._fcn(parent)
