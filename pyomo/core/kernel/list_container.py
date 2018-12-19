#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.core.kernel.tuple_container import TupleContainer

import six
from six.moves import xrange as range

if six.PY3:
    from collections.abc import MutableSequence as collections_MutableSequence
else:
    from collections import MutableSequence as collections_MutableSequence

logger = logging.getLogger('pyomo.core')

class ListContainer(TupleContainer,
                    collections_MutableSequence):
    """
    A partial implementation of the IHomogeneousContainer
    interface that provides list-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level and initialize the remaining
    ICategorizedObject attributes during object creation. If
    using __slots__, a slot named "_data" must be included.

    Note that this implementation allows nested storage of
    other ICategorizedObjectContainer implementations that
    are defined with the same ctype.
    """
    __slots__ = ()

    def __init__(self, *args):
        self._data = []
        self._init(args)

    #
    # Define the MutableSequence abstract methods
    #

    def __setitem__(self, i, item):
        if item.ctype is self.ctype:
            if item._parent is None:
                # be sure the current object is properly
                # removed
                logger.warning(
                    "Implicitly replacing the entry %s (type=%s) "
                    "with a new object (type=%s). This is usually "
                    "indicative of a modeling error. To avoid this "
                    "warning, delete the original object from the "
                    "container before assigning a new object."
                    % (self[i].name,
                       self[i].__class__.__name__,
                       item.__class__.__name__))
                self._data[i]._clear_parent_and_storage_key()
                item._update_parent_and_storage_key(self, i)
                self._data[i] = item
                return
            elif self._data[i] is item:
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                return
            # see note about allowing objects to live in more than
            # one container
            raise ValueError(
                "Invalid assignment to %s type with name '%s' "
                "at index %s. A parent container has already been "
                "assigned to the object being inserted: %s"
                % (self.__class__.__name__,
                   self.name,
                   i,
                   item.parent.name))
        else:
            raise TypeError(
                "Invalid assignment to type %s with index %s. "
                "The object being inserted has the wrong "
                "category type: %s"
                % (self.__class__.__name__,
                   i,
                   item.ctype))

    def insert(self, i, item):
        """S.insert(index, object) -- insert object before index"""
        self._insert(i, item)

    def __delitem__(self, i):
        self._data[i]._clear_parent_and_storage_key()
        del self._data[i]

    #
    # Override a few default implementations on Sequence
    #
    # We want to avoid issues with parent ownership
    # as we shuffle items

    def reverse(self):
        """S.reverse() -- reverse *IN PLACE*"""
        n = len(self)
        data = self._data
        for i in range(n//2):
            data[i], data[n-i-1] = data[n-i-1], data[i]
