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
    from collections.abc import MutableSet
else:
    from collections import MutableSet

from collections import OrderedDict

class OrderedSet(MutableSet):
    __slots__ = ('_dict')

    def __init__(self, iterable=None):
        self._dict = OrderedDict()
        if iterable is not None:
            self.update(iterable)

    def __str__(self):
        """String representation of the mapping."""
        return "OrderedSet(%s)" % (', '.join(repr(x) for x in self))


    def update(self, iterable):
        for val in iterable:
            self.add(val)

    #
    # This method must be defined for deepcopy/pickling
    # because this class is slotized.
    #
    def __setstate__(self, state):
        self._dict = state

    def __getstate__(self):
        return self._dict

    #
    # Implement MutableSet abstract methods
    #

    def __contains__(self, val):
        return val in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def add(self, val):
        """Add an element."""
        if val not in self._dict:
            self._dict[val] = None

    def discard(self, val):
        """Remove an element. Do not raise an exception if absent."""
        if val in self._dict:
            del self._dict[val]

    #
    # The remaining MutableSet methods have slow default
    # implementations.
    #

    def clear(self):
        """Remove all elements from this set."""
        self._dict.clear()

    def remove(self, val):
        """Remove an element. If not a member, raise a KeyError."""
        del self._dict[val]
