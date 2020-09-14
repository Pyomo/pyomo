#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

GlobalSets = {}
def _get_global_set(name):
    return GlobalSets[name]
_get_global_set.__safe_for_unpickling__ = True

class GlobalSetBase(object):
    """The base class for all Global sets"""
    __slots__ = ()

    def __reduce__(self):
        # Cause pickle to preserve references to this object
        return _get_global_set, (self.local_name,)

    def __deepcopy__(self, memo):
        # Prevent deepcopy from duplicating this object
        return self

    def __str__(self):
        # Override str() to always print out the global set name
        return self.name

# FIXME: This mocks up part of the Set API until we can break up the set
# module to resolve circular dependencies and can make this a proper
# GlobalSet (Scalar IndexedComponent objects are indexed by
# UnindexedComponent_set, but we would like UnindexedComponent_set to be
# a proper scalar IndexedComponent).
#
#UnindexedComponent_set = set([None])
class _UnindexedComponent_set(GlobalSetBase):
    def __init__(self, name):
        self.name = name
    def __contains__(self, val):
        return val is None
    def get(self, value, default):
        if value is None:
            return value
        return default
    def __iter__(self):
        return (None,).__iter__()
    def subsets(self):
        return [ self ]
    def construct(self):
        pass
    def __len__(self):
        return 1
    def __eq__(self, other):
        return self is other
    def __ne__(self, other):
        return self is not other
    @property
    def dimen(self):
        return 0
    def isdiscrete(self):
        return True
    def isfinite(self):
        return True
    def isordered(self):
        # As this set only has a single element, it is implicitly "ordered"
        return True
UnindexedComponent_set = _UnindexedComponent_set('UnindexedComponent_set')
