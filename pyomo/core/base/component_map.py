#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('ComponentMap',)

import collections

class ComponentMap(collections.MutableMapping):
    """
    This class acts as replacement for dict that allows
    Pyomo modeling components to be used as entry keys. The
    underlying mapping is based on the Python id() of the
    object, so that hashing can still take place when the
    object is mutable, unhashable, etc. This class meant
    for Pyomo components to values. The use of non-Pyomo
    components as entry keys should be avoided.

    A reference to the object is kept around as long as it
    has a corresponding entry in the container so that we
    don't need to worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object id()'s after
    a deepcopy or unpickle.

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

    def __setstate__(self, state):
        """
        This method must be defined for deepcopy/pickling because this
        class relies on Python ids.
        """
        id_func = id
        # object id() may have changed after unpickling so we rebuild
        # the dictionary keys
        self._dict = \
            dict((id_func(component), (component,val)) \
                 for component, val in itervalues(state['_dict']))

    def __str__(self):
        """String representation of the mapping."""
        tmp = '{' + \
              (', '.join(component.cname(True)+": "+str(val) \
                        for component, val \
                        in itervalues(self._dict))) + \
              '}'
        return tmp

    def pprint(self, stream=None, indent=1, width=80, depth=None, verbose=False):
        """
        Pretty-print a Python object to a stream [default is sys.stdout].
        """
        if verbose:
            tmp = dict((repr(component.cname(True))+" (id="+str(id(component))+")", val)
                           for component, val \
                           in itervalues(self._dict))
        else:
            tmp = dict((repr(component.cname(True))+" (id="+str(id(component))+")", val)
                           for component, val \
                           in itervalues(self._dict))
        pprint.pprint(tmp,
                      stream=stream,
                      indent=indent,
                      width=width,
                      depth=depth)

    #
    # Implement MutableMapping abstract methods
    #

    def __getitem__(self, component):
        try:
            return self._dict[id(component)][1]
        except KeyError:
            cname = component.cname(True)
            raise KeyError("Component with name: "
                           +cname+
                           " (id=%s)" % id(component))

    def __setitem__(self, component, value):
        self._dict[id(component)] = (component,value)

    def __delitem__(self, component):
        try:
            del self._dict[id(component)]
        except KeyError:
            cname = component.cname(True)
            raise KeyError("Component with name: "
                           +cname+
                           " (id=%s)" % id(component))

    def __iter__(self):
        return (component \
                for component, value in \
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
        if not isinstance(other, collections.Mapping):
            return False
        return dict(((type(key), id(key)), val)
                    for key, val in self.items()) == \
               dict(((type(key), id(key)), val)
                    for key, val in other.items())

    def __ne__(self, other):
        return not (self == other)

    #
    # The remaining methods have slow default
    # implementations for collections.MutableMapping. In
    # particular, they rely KeyError catching, which is slow
    # for this class because KeyError messages use fully
    # qualified names.
    #

    def __contains__(self, component):
        return id(component) in self._dict

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
