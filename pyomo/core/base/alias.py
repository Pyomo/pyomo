import weakref
import logging
from copy import deepcopy

from pyomo.core.base.component import Component, ComponentData

logger = logging.getLogger('pyomo.core')

__all__ = ['Alias']

#
# Key Points (for now - subject to change):
#   - Alias mimics the behavior of weakref.proxy (mostly)
#     using a weakref.ref under the hood. Aliasing an
#     object will not prevent it from being garbage collected.
#   - Alias will forward all function calls and member lookups
#     through to the underlying proxy, EXCEPT for those on
#     the component base class.
#   - Alias prevents prevents __deepcopy__ on ComponentData
#     from being called. All clones/copies are shallow,
#     that is, unless an Alias is pickled completely separate
#     from what it is referencing.
#   - Updating the referenced object is disabled for now,
#     but would be easy to update.
#   - Many magic methods do not go through __getattr__,
#     so we manually forward them.
#   - aliased_object (property), provides direct access to
#     the aliased component (unless the weakref has become
#     invalid), so that Component base class methods can
#     be called if needed.
#   - Alias inherits the ctype of the object it proxies (for now). Not
#     sure if this is good or bad.  This could be bad for solver
#     plugins (e.g., extra variables), but I think when an expression
#     is generated due to forwarding NumValue base class methods, this
#     results in the Alias (e.g., for a Var) getting replaced with its
#     aliased object in the expression. So problem writers and
#     canonical_repn could possibly never encounter Aliases (if we're
#     careful).

class Alias(Component):

    __slots__ = ('_aliased_object')

    # It would be nice if Component declared these as slots, but for
    # now we need to hard code them so they can be set during
    # Component.__init__ (and properly redirected back to the base
    # class from the overloaded __setattr__ method defined here
    _component_slots = ['_type','doc','name','_active','_constructed','_parent']

    def __init__(self, obj):
        self._aliased_object = None
        if isinstance(obj, Alias):
            obj = obj._aliased_object()
        if obj is not None:
            self._aliased_object = weakref.ref(obj)
        ctype = Alias
        if isinstance(obj, Component):
            ctype = obj.type()
        else:
            if not isinstance(obj, ComponentData):
                raise TypeError("Aliased object must be an "
                                "instance of Component or ComponentData")
            ctype = obj.parent_component().type()
        Component.__init__(self, ctype=ctype)

    @property
    def aliased_object(self):
        obj = Alias.__dict__['_aliased_object'].__get__(self)
        if obj is not None:
            obj = obj()
        return obj
    @aliased_object.setter
    def aliased_object(self, value):
        raise AttributeError("Assignment not allowed.")

    #
    # Overload methods on Component
    #

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                name = type(self)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Constructing Alias, name=%s, "
                                 "from data=%s", name, str(data))
        if self._constructed:
            return
        self._constructed = True

    def _pprint(self):
        if self.aliased_object is None:
            return ([("Proxy","None")], (), (), ())
        else:
            return ([("Proxy", self.aliased_object.name)], (), (), ())

    # Dereference so we can point to a constructed object
    # after model cloning / construction
    def __getstate__(self):
        result = super(Alias, self).__getstate__()
        result['_aliased_object'] = self.aliased_object
        return result

    def __setstate__(self, state):
        super(Alias, self).__setstate__(state)
        if self._aliased_object is not None:
            self._aliased_object = weakref.ref(self._aliased_object)

    #
    # Note: __getattr__ will only be invoked when a
    #       a method is not defined here or on base class
    #       or when an member name is not defined
    #       on this class (base class not included).
    #       Any members on the Component
    #       base class do not get redirected
    #       to the proxy, so we check there first
    #       before calling a proxy method (so that
    #       __init__ on Component works for this
    #       objects base class). This might be
    #       different if Component is slot-ized.
    #
    def __getattr__(self, name):

        # if we are proxy to a ComponentData
        # object, we need to make sure NOT to
        # invoke its deepcopy method
        if name == "__deepcopy__":
            raise AttributeError

        if name in Alias.__slots__:
            # should never happen
            assert False

        if name in Alias._component_slots:
            return super(Alias, self).__dict__[name].__get__(self)

        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return getattr(self.aliased_object, name)

    #
    # __setattr__ and __delattr__ will always be invoked
    #

    def __setattr__(self, name, attr):
        # avoid infinite recursion
        if name in Alias.__slots__:
            # this is how you access a slot while bypassing __setattr__
            return Alias.__dict__[name].__set__(self, attr)

        if name in Alias._component_slots:
            super(Alias, self).__dict__[name] = attr
            return

        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is not longer valid")
        return setattr(self.aliased_object, name, attr)

    def __delattr__(self, name):
        # avoid infinite recursion
        if name in Alias.__slots__:
            raise TypeError("'dictproxy' object does not support item deletion")

        if name in Alias._component_slots:
            del super(Alias, self).__dict__[name]

        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is not longer valid")
        return delattr(self.aliased_object, name)

    #
    # In addition to overloading __getattr__, the following magic
    # methods need to be redirected as Python does not access them
    # through __getitem__. If the aliased object does not implement
    # them an exception will be thrown
    #

    # E.g., IndexedComponent

    def __len__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__len__()

    def __contains__(self, key):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__contains__(key)

    def __iter__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__iter__()

    def __getitem__(self, key):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__getitem__(key)

    def __setitem__(self, key, value):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__setitem__(key, value)

    # E.g., NumValue

    def __float__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__float__()

    def __int__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__int__()

    def __lt__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__lt__(other)

    def __gt__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__gt__(other)

    def __le__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__le__(other)

    def __ge__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__ge__(other)

    def __eq__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__eq__(other)

    def __add__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__add__(other)

    def __sub__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__sub__(other)

    def __mul__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__mul__(other)

    def __div__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__div__(other)

    def __truediv__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__truediv__(other)

    def __pow__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__pow__(other)

    def __radd__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__radd__(other)

    def __rsub__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__rsub__(other)

    def __rmul__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__rmul__(other)

    def __rdiv__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__rdiv__(other)

    def __rtruediv__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__rtruediv__(other)

    def __rpow__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__rpow__(other)

    def __iadd__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__iadd__(other)

    def __isub__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__isub__(other)

    def __imul__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__imul__(other)

    def __idiv__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__idiv__(other)

    def __itruediv__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__itruediv__(other)

    def __ipow__(self, other):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__ipow__(other)

    def __neg__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__neg__()

    def __pos__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__pos__()

    def __abs__(self):
        if self.aliased_object is None:
            raise weakref.ReferenceError("Proxy is no longer valid")
        return self.aliased_object.__abs__()
