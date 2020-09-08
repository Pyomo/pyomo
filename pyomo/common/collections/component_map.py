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
from six import itervalues

if six.PY3:
    from collections.abc import MutableMapping as collections_MutableMapping
    from collections.abc import Mapping as collections_Mapping
else:
    from collections import MutableMapping as collections_MutableMapping
    from collections import Mapping as collections_Mapping

class ComponentMap(collections_MutableMapping):
    """
    This class is a replacement for dict that allows Pyomo
    modeling components to be used as entry keys. The
    underlying mapping is based on the Python id() of the
    object, which gets around the problem of hashing
    subclasses of NumericValue. This class is meant for
    creating mappings from Pyomo components to values. The
    use of non-Pyomo components as entry keys should be
    avoided.

    A reference to the object is kept around as long as it
    has a corresponding entry in the container, so there is
    no need to worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object ids after
    a deepcopy or pickle.

    *** An instance of this class should never be
    deepcopied/pickled unless it is done so along with the
    components for which it contains map entries (e.g., as
    part of a block). ***
    """
    __slots__ = ("_dict",)
    def __init__(self, *args, **kwds):
        # maps id(obj) -> (obj,val)
        self._dict = {}
        # handle the dict-style initialization scenarios
        self.update(*args, **kwds)

    #
    # This method must be defined for deepcopy/pickling
    # because this class relies on Python ids.
    #
    def __setstate__(self, state):
        # *** Temporary hack to allow this class to be used
        # *** in inheritance chains for both the old and new
        # *** component hierarchies.
        for cls in self.__class__.__mro__:
            if cls.__name__ == "ICategorizedObject":
                super(ComponentMap, self).__setstate__(state)
                break

        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        self._dict = \
            {id(obj):(obj,val) \
                 for obj, val in itervalues(state['_dict'])}

    def __getstate__(self):
        # *** Temporary hack to allow this class to be used
        # *** in inheritance chains for both the old and new
        # *** component hierarchies.
        try:
            super(ComponentMap, self).__getstate__
        except AttributeError:
            state = {}
        else:
            state = super(ComponentMap, self).__getstate__()
        for cls in self.__class__.__mro__:
            if cls.__name__ == "ICategorizedObject":
                break
        else:
            for i in ComponentMap.__slots__:
                state[i] = getattr(self, i)
        return state

    def __str__(self):
        """String representation of the mapping."""
        tmp = {str(c)+" (id="+str(id(c))+")":v for c,v in self.items()}
        return "ComponentMap("+str(tmp)+")"

    #
    # Implement MutableMapping abstract methods
    #

    def __getitem__(self, obj):
        try:
            return self._dict[id(obj)][1]
        except KeyError:
            raise KeyError("Component with id '%s': %s"
                           % (id(obj), str(obj)))

    def __setitem__(self, obj, val):
        self._dict[id(obj)] = (obj,val)

    def __delitem__(self, obj):
        try:
            del self._dict[id(obj)]
        except KeyError:
            raise KeyError("Component with id '%s': %s"
                           % (id(obj), str(obj)))

    def __iter__(self):
        return (obj \
                for obj, val in \
                itervalues(self._dict))

    def __len__(self):
        return self._dict.__len__()

    #
    # Overload MutableMapping default implementations
    #

    # We want to avoid generating Pyomo expressions due to
    # comparison of values, so we convert both objects to a
    # plain dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, collections_Mapping):
            return False
        return {(type(key), id(key)):val
                    for key, val in self.items()} == \
               {(type(key), id(key)):val
                    for key, val in other.items()}

    def __ne__(self, other):
        return not (self == other)

    #
    # The remaining methods have slow default
    # implementations for MutableMapping. In particular,
    # they rely KeyError catching, which is slow for this
    # class because KeyError messages use fully qualified
    # names.
    #

    def __contains__(self, obj):
        return id(obj) in self._dict

    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        self._dict.clear()

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        if key in self:
            return self[key]
        return default

    def setdefault(self, key, default=None):
        'D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D'
        if key in self:
            return self[key]
        else:
            self[key] = default
        return default


