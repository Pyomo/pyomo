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
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.core.base.component_interface import \
    (IComponentContainer,
     _abstract_readwrite_property)
from pyomo.core.base.component_map import ComponentMap

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
    """
    A partial implementation of the IComponentContainer
    interface that presents dict-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level and declare the remaining required
    abstract properties of the IComponentContainer base
    class plus and additional _data property.

    Note that this implementation allows nested storage of
    other IComponentContainer implementations that are
    defined with the same ctype.

    The optional keyword 'ordered' can be set to True to
    indicate that an OrderedDict should be used as the
    underlying storage dictionary.
    """
    __slots__ = ()

    def __init__(self, *args, **kwds):
        ordered = kwds.pop('ordered', False)
        if len(kwds):
            raise ValueError("Unexpected keywords used "
                             "to initialize class: %s"
                             % (str(list(kwds.keys()))))
        if ordered:
            self._data = OrderedDict()
        else:
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
        for child in self.children():
            if child._is_component:
                yield child
            else:
                for component in child.components():
                    yield component

    def child_key(self, child):
        for key, val in iteritems(self._data):
            if val is child:
                return key
        raise ValueError

    def children(self, return_key=False):
        """Iterate over the children of this container.

        Args:
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting the child storage key and the
                child object. By default, only the child
                objects are returned.

        Returns: an iterator of objects or (key,object) tuples
        """
        if return_key:
            return iteritems(self._data)
        else:
            return itervalues(self._data)

    def preorder_traversal(self,
                           active=None,
                           return_key=False,
                           root_key=None):
        """
        Generates a preorder traversal of the storage tree.

        Args:
            active (True/None): Set to True to indicate that
                only active objects should be included. The
                default value of None indicates that all
                components (including those that have been
                deactivated) should be included. *Note*: This
                flag is ignored for any objects that do not
                have an active flag.
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting of the local storage key of the
                object within its parent and the object
                itself. By default, only the objects are
                returned.
            root_key: The key to return with this object
                (when return_key is True).

        Returns: an iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        _active_flag_name = "active"
        if (active is None) or \
           getattr(self, _active_flag_name, True):
            if return_key:
                yield root_key, self
            else:
                yield self
            for key, child in self.children(return_key=True):
                if (active is None) or \
                   getattr(child, _active_flag_name, True):
                    if child._is_component:
                        if return_key:
                            yield key, child
                        else:
                            yield child
                    else:
                        assert child._is_container
                        for item in child.preorder_traversal(
                                active=active,
                                return_key=return_key,
                                root_key=key):
                            yield item

    def postorder_traversal(self,
                            active=None,
                            return_key=False,
                            root_key=None):
        """
        Generates a postorder traversal of the storage tree.

        Args:
            active (True/None): Set to True to indicate that
                only active objects should be included. The
                default value of None indicates that all
                components (including those that have been
                deactivated) should be included. *Note*: This
                flag is ignored for any objects that do not
                have an active flag.
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting of the local storage key of the
                object within its parent and the object
                itself. By default, only the objects are
                returned.
            root_key: The key to return with this object
                (when return_key is True).

        Returns: an iterator of objects or (key,object) tuples
        """
        assert active in (None, True)
        _active_flag_name = "active"
        if (active is None) or \
           getattr(self, _active_flag_name, True):
            for key, child in self.children(return_key=True):
                if (active is None) or \
                   getattr(child, _active_flag_name, True):
                    if child._is_component:
                        if return_key:
                            yield key, child
                        else:
                            yield child
                    else:
                        assert child._is_container
                        for item in child.postorder_traversal(
                                active=active,
                                return_key=return_key,
                                root_key=key):
                            yield item
            if return_key:
                yield root_key, self
            else:
                yield self

    def generate_names(self,
                       active=None,
                       descend_into=True,
                       convert=str,
                       prefix=""):
        """
        Generate a container of fully qualified names (up to
        this container) for objects stored under this
        container.

        Args:
            active (True/None): Set to True to indicate that
                only active components should be
                included. The default value of None
                indicates that all components (including
                those that have been deactivated) should be
                included. *Note*: This flag is ignored for
                any objects that do not have an active flag.
            descend_into (bool): Indicates whether or not to
                include subcomponents of any container
                objects that are not components. Default is
                True.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is str.
            prefix (str): A string to prefix names with.

        Returns:
            A component map that behaves as a dictionary
            mapping component objects to names.
        """
        assert active in (None, True)
        _active_flag_name = "active"
        names = ComponentMap()
        if (active is None) or \
           getattr(self, _active_flag_name, True):
            name_template = (prefix +
                             self._child_storage_delimiter_string +
                             self._child_storage_entry_string)
            for key, child in self.children(return_key=True):
                if (active is None) or \
                   getattr(child, _active_flag_name, True):
                    names[child] = name_template % convert(key)
                    if descend_into and child._is_container and \
                       (not child._is_component):
                        names.update(child.generate_names(
                            active=active,
                            descend_into=True,
                            convert=convert,
                            prefix=names[child]))
        return names

    #
    # Define the MutableMapping abstract methods
    #

    def __setitem__(self, key, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                item._parent = weakref.ref(self)
                if key in self._data:
                    # release the current component
                    # * see __delitem__ for explanation
                    self._data[key]._parent = None
                self._data[key] = item
                if (not getattr(self, "_active", True)) and \
                   getattr(item, "_active", False):
                    # this will notify all inactive
                    # ancestors that they are now active
                    item.activate()
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
                   self.name,
                   key,
                   item.parent.name))
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
