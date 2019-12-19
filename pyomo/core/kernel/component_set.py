#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import six
from six import itervalues, iteritems

if six.PY3:
    from collections.abc import MutableSet as collections_MutableSet
    from collections.abc import Set as collections_Set
else:
    from collections import MutableSet as collections_MutableSet
    from collections import Set as collections_Set

class ComponentSet(collections_MutableSet):
    """
    This class is a replacement for set that allows Pyomo
    modeling components to be used as entries. The
    underlying hash is based on the Python id() of the
    object, which gets around the problem of hashing
    subclasses of NumericValue. This class is meant for
    creating sets of Pyomo components. The use of non-Pyomo
    components as entries should be avoided (as the behavior
    is undefined).

    References to objects are kept around as long as they
    are entries in the container, so there is no need to
    worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object ids after
    a deepcopy or pickle.

    *** An instance of this class should never be
    deepcopied/pickled unless it is done so along with
    its component entries (e.g., as part of a block). ***
    """
    __slots__ = ("_data",)
    def __init__(self, *args):
        self._data = dict()
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            self.update(args[0])

    def __str__(self):
        """String representation of the mapping."""
        tmp = []
        for objid, obj in iteritems(self._data):
            tmp.append(str(obj)+" (id="+str(objid)+")")
        return "ComponentSet("+str(tmp)+")"

    def update(self, args):
        """Update a set with the union of itself and others."""
        self._data.update((id(obj), obj)
                          for obj in args)

    #
    # This method must be defined for deepcopy/pickling
    # because this class relies on Python ids.
    #
    def __setstate__(self, state):
        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        assert len(state) == 1
        self._data = {id(obj):obj for obj in state['_data']}

    def __getstate__(self):
        return {'_data': tuple(self._data.values())}

    #
    # Implement MutableSet abstract methods
    #

    def __contains__(self, val):
        return self._data.__contains__(id(val))

    def __iter__(self):
        return itervalues(self._data)

    def __len__(self):
        return self._data.__len__()

    def add(self, val):
        """Add an element."""
        self._data[id(val)] = val

    def discard(self, val):
        """Remove an element. Do not raise an exception if absent."""
        if id(val) in self._data:
            del self._data[id(val)]

    #
    # Overload MutableSet default implementations
    #

    # We want to avoid generating Pyomo expressions due to
    # comparison of values, so we convert both objects to a
    # plain dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, collections_Set):
            return False
        return set((type(val), id(val))
                   for val in self) == \
               set((type(val), id(val))
                   for val in other)

    def __ne__(self, other):
        return not (self == other)

    #
    # The remaining MutableSet methods have slow default
    # implementations.
    #

    def clear(self):
        """Remove all elements from this set."""
        self._data.clear()

    def remove(self, val):
        """Remove an element. If not a member, raise a KeyError."""
        try:
            del self._data[id(val)]
        except KeyError:
            raise KeyError("Component with id '%s': %s"
                           % (id(val), str(val)))
