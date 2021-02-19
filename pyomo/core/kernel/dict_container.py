#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging

from pyomo.common.collections import OrderedDict, Mapping, MutableMapping

from pyomo.core.kernel.homogeneous_container import \
    IHomogeneousContainer

if sys.version_info[:2] >= (3,7):
    # dict became ordered in CPython 3.6 and added to the standard in 3.7
    _ordered_dict_ = dict
else:
    _ordered_dict_ = OrderedDict

from six import itervalues

logger = logging.getLogger('pyomo.core')

# Note that prior to Python 3, collections
# is not defined with an empty __slots__
# attribute. Therefore, in Python 2, all implementations of
# this class will have a __dict__ member whether or not they
# declare __slots__. I don't believe it is worth trying to
# code a work around for the Python 2 case as we are moving
# closer to a Python 3-only world these types of objects are
# not memory bottlenecks.
class DictContainer(IHomogeneousContainer, MutableMapping):
    """
    A partial implementation of the IHomogeneousContainer
    interface that provides dict-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level and initialize the remaining
    ICategorizedObject attributes during object creation. If
    using __slots__, a slot named "_data" must be included.

    Note that this implementation allows nested storage of
    other ICategorizedObjectContainer implementations that
    are defined with the same ctype.
    """
    __slots__ = ()
    _child_storage_delimiter_string = ""
    _child_storage_entry_string = "[%s]"

    def __init__(self, *args, **kwds):
        self._data = _ordered_dict_()
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            self.update(args[0])
        if len(kwds):
            self.update(**kwds)

    def _fast_insert(self, key, item):
        item._update_parent_and_storage_key(self, key)
        self._data[key] = item

    #
    # Define the ICategorizedObjectContainer abstract methods
    #

    def child(self, key):
        """Get the child object associated with a given
        storage key for this container.

        Raises:
            KeyError: if the argument is not a storage key
                for any children of this container
        """
        return self[key]

    def children(self):
        """A generator over the children of this container."""
        return itervalues(self._data)

    #
    # Define the MutableMapping abstract methods
    #

    def __setitem__(self, key, item):
        if item.ctype is self.ctype:
            if item._parent is None:
                if key in self._data:
                    logger.warning(
                        "Implicitly replacing the entry %s (type=%s) "
                        "with a new object (type=%s). This is usually "
                        "indicative of a modeling error. To avoid this "
                        "warning, delete the original object from the "
                        "container before assigning a new object."
                        % (self[key].name,
                           self[key].__class__.__name__,
                           item.__class__.__name__))
                    self._data[key]._clear_parent_and_storage_key()
                self._fast_insert(key, item)
                return
            elif (key in self._data) and (self._data[key] is item):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                return
            # see note about allowing categorized objects to
            # live in more than one container
            raise ValueError(
                "Invalid assignment to %s type with name '%s' "
                "at key %s. A parent container has already been "
                "assigned to the object being inserted: %s"
                % (self.__class__.__name__,
                   self.name,
                   key,
                   item.parent.name))
        else:
            raise TypeError(
                "Invalid assignment to type %s with index %s. "
                "The object being inserted has the wrong "
                "category type: %s"
                % (self.__class__.__name__,
                   key,
                   item.ctype))

    def __delitem__(self, key):
        self._data[key]._clear_parent_and_storage_key()
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

    def __contains__(self, key):
        return self._data.__contains__(key)

    # We want to avoid generating Pyomo expressions due to
    # comparison of values, so we convert both objects to a
    # plain dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return False
        return {key:(type(val), id(val))
                    for key, val in self.items()} == \
               {key:(type(val), id(val))
                    for key, val in other.items()}

    def __ne__(self, other):
        return not (self == other)
