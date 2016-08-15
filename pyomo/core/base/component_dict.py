#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ComponentDict",)

import weakref
import collections

from pyomo.core.base.component_interface import \
    (IComponentContainer,
     _abstract_readwrite_property)

import six
from six import itervalues, iteritems

# Note that prior to Python 3, collections.MutableMappping
# is not defined with an empty __slots__
# attribute. Therefore, in Python 2, all implementations of
# this class will have a __dict__ member whether or not they
# declare __slots__. I don't believe it is worth trying to
# code a work around for the Python 2 case as we are moving
# closer to a Python 3-only world and indexed component
# storage containers are not really memory bottlenecks.
class ComponentDict(IComponentContainer,
                    collections.MutableMapping):
    """A partial implementation of the IComponentContainer
    interface that presents dict-like storage functionality.
    """
    __slots__ = ()

    def __init__(self, *args):
        self._data = {}
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            self.update(args[0])

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _data = _abstract_readwrite_property()

    #
    # Define the IComponentContainer abstract methods
    #

    def components(self):
        return itervalues(self._data)

    def component_entry_key(self, component):
        for key, val in iteritems(self._data):
            if val is component:
                return key
        raise ValueError

    #
    # Define the MutableMapping abstract methods
    #

    def __setitem__(self, key, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                item._parent = weakref.ref(self)
                if hasattr(self, "_active"):
                    self._active |= getattr(item, '_active', True)
                if key in self._data:
                    # release the current component
                    # * see __delitem__ for explanation
                    self._data[key]._parent = None
                self._data[key] = item
                return
            elif (key in self._data) and (self._data[key] is item):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                return
            # see note about allowing components to live in more than
            # one container
            raise ValueError(
                "Invalid assignment to %s type with name '%s' "
                "at key %s. A parent container has already been "
                "assigned to the component being inserted: %s"
                % (self.__class__.__name__,
                   self.cname(True),
                   key,
                   item.parent.cname(True)))
        else:
            raise TypeError(
                "Invalid assignment to type %s with index %s. "
                "The component being inserted has the wrong "
                "component type: %s"
                % (self.__class__.__name__,
                   key,
                   item.ctype))


    def __delitem__(self, key):
        obj = self._data[key]
        obj._parent = None
        del self._data[key]

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    #
    # Override a few default implementations on MutableMapping
    #

    # We want to avoid generating Pyomo expressions due to
    # comparison of values, so we convert both objects to a
    # plain dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, collections.Mapping):
            return False
        return dict((key, (type(val), id(val)))
                    for key, val in self.items()) == \
               dict((key, (type(val), id(val)))
                    for key, val in other.items())

    def __ne__(self, other):
        return not (self == other)

if __name__ == "__main__":

    class VarDict(ComponentDict):
        __slots__ = ("_ctype",
                     "_parent",
                     "_data")
        def __init__(self, *args, **kwds):
            self._ctype = "Var"
            self._parent = None
            self._data = {}
            super(VarDict, self).__init__(*args, **kwds)

    v = VarDict()
    print("issubclass: "+str(issubclass(ComponentDict,
                                        collections.MutableMapping)))
    print("issubclass: "+str(issubclass(VarDict,
                                        collections.MutableMapping)))
    print("isinstance: "+str(isinstance(v,
                                        collections.MutableMapping)))
    print(v.keys())
