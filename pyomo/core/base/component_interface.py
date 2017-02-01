#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('ICategorizedObject',
           'IActiveObject',
           'IComponent',
           '_IActiveComponent',
           'IComponentContainer',
           '_IActiveComponentContainer')

import sys
import abc
import copy
import weakref

import six
from six.moves import xrange

def _not_implemented(*args, **kwds):
    raise NotImplementedError     #pragma:nocover

def _abstract_readwrite_property(**kwds):
    p = abc.abstractproperty(fget=_not_implemented,
                             fset=_not_implemented,
                             **kwds)
    if 'doc' in kwds:
        p.__doc__ = kwds['doc']
    return p

def _abstract_readonly_property(**kwds):
    p = abc.abstractproperty(fget=_not_implemented,
                             **kwds)
    if 'doc' in kwds:
        p.__doc__ = kwds['doc']
    return p

class ICategorizedObject(six.with_metaclass(abc.ABCMeta, object)):
    """
    Interface for objects that maintain a weak reference to
    a parent storage object and have a category type.

    This class is abstract. It assumes any derived class
    declares the following attributes (with or without __slots__):
        - _ctype: The objects category type.
        - _parent: A weak reference to the object's
                   parent or None.
    """
    __slots__ = ()

    # These flags can be used by implementations to speed up
    # code. The use of ABCMeta as a metaclass slows down
    # isinstance calls by an order of magnitude! So for
    # instance, use hasattr(obj, '_is_categorized') as
    # opposed to isinstance(obj, ICategorizedObject)
    _is_categorized_object = True
    _is_component = False
    _is_container = False

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    #
    # Interface
    #

    @property
    def ctype(self):
        """The object's category type."""
        return self._ctype

    @property
    def parent(self):
        """The object's parent"""
        if isinstance(self._parent, weakref.ReferenceType):
            return self._parent()
        else:
            return self._parent

    @property
    def parent_block(self):
        """The first ancestor block above this object"""
        parent = self.parent
        while (parent is not None) and \
              (not parent._is_component):
            parent = parent.parent
        return parent

    @property
    def root_block(self):
        """The root storage block above this object"""
        root_block = None
        if self._is_component and self._is_container:
            root_block = self
        parent_block = self.parent_block
        while parent_block is not None:
            root_block = parent_block
            parent_block = parent_block.parent_block
        return root_block

    def getname(self,
                fully_qualified=False,
                name_buffer={}, # HACK: ignored (required to work with some solver interfaces, but that code should change soon)
                convert=str):
        """
        Dynamically generate a name for the object its storage
        key in a parent. If there is no parent, the method will
        return None.

        Args:
            fully_qualified (bool): Generate full name from
                nested block names. Default is False.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is repr.

        Returns:
            A string representing the name of the component
            or None.
        """
        parent = self.parent
        if parent is None:
            return None

        key = parent.child_key(self)
        name = parent._child_storage_entry_string % convert(key)
        if fully_qualified:
            parent_name = parent.getname(fully_qualified=True)
            if parent_name is not None:
                return (parent_name +
                        parent._child_storage_delimiter_string +
                        name)
            else:
                return name
        else:
            return name

    @property
    def name(self):
        """The object's fully qualified name. Alias
        for obj.getname(fully_qualified=True)."""
        return self.getname(fully_qualified=True)

    @property
    def local_name(self):
        """The object's local name within the context of its
        parent. Alias for obj.getname(fully_qualified=False)."""
        return self.getname(fully_qualified=False)

    def __str__(self):
        """Convert this object to a string."""
        name = self.name
        if name is None:
            return "<"+self.__class__.__name__+">"
        else:
            return name

    #
    # The following method must be defined to allow
    # expressions and blocks to be correctly cloned.
    # (copied from JDS code in original component.py,
    # comments removed for now)
    #

    def __deepcopy__(self, memo):
        if '__block_scope__' in memo and \
                id(self) not in memo['__block_scope__']:
            _known = memo['__block_scope__']
            _new = []
            tmp = self.parent_block
            tmpId = id(tmp)
            while tmpId not in _known:
                _new.append(tmpId)
                tmp = tmp.parent_block
                tmpId = id(tmp)

            for _id in _new:
                _known[_id] = _known[tmpId]

            if not _known[tmpId]:
                # component is out-of-scope.  shallow copy only
                ans = memo[id(self)] = self
                return ans

        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    #
    # The following two methods allow implementations to be
    # pickled. These should work whether or not the
    # implementation makes use of __slots__, and whether or
    # not non-empty __slots__ declarations appear on
    # multiple classes in the inheritance chain.
    #

    def __getstate__(self):
        state = getattr(self, "__dict__", {}).copy()
        # Get all slots in the inheritance chain
        for cls in self.__class__.__mro__:
            for key in cls.__dict__.get("__slots__",()):
                state[key] = getattr(self, key)
        # make sure we don't store the __dict__ in
        # duplicate (it can be declared as a slot)
        state.pop('__dict__', None)
        # make sure not to pickle the __weakref__
        # slot if it was declared
        state.pop('__weakref__', None)
        # make sure to dereference the parent weakref
        state['_parent'] = self.parent
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            # bypass a possibly overridden __setattr__
            object.__setattr__(self, key, value)
        # make sure _parent is a weakref
        # if it is not None
        if self._parent is not None:
            self._parent = weakref.ref(self._parent)

class IActiveObject(six.with_metaclass(abc.ABCMeta, object)):
    """
    Interface for objects that support activate/deactivate
    semantics.

    This class is abstract.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    active = _abstract_readonly_property(
        doc=("A boolean indicating whether or "
             "not this object is active."))

    #
    # Interface
    #

    @abc.abstractmethod
    def activate(self, *args, **kwds):
        """Set the active attribute to True"""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def deactivate(self, *args, **kwds):
        """Set the active attribute to False"""
        raise NotImplementedError     #pragma:nocover

class IComponent(ICategorizedObject):
    """
    Interface for components that can be stored inside
    objects of type IComponentContainer.

    This class is abstract.
    """
    _is_component = True
    _is_container = False
    __slots__ = ()

    #
    # Interface
    #

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Write the component string representation to a buffer"""
        if ostream is None:
            ostream = sys.stdout
        ostream.write(self.__str__())

class _IActiveComponentMixin(IActiveObject):
    """
    To be used as an additional base class in IComponent
    implementations to add fuctionality for activating and
    deactivating the component.

    Any container that stores implementations of this type
    should use _IActiveComponentContainerMixin as a base
    class.

    This class is abstract. It assumes any derived class
    declares the following attributes (with or without __slots__):
        - _active: A boolean indicating whethor or not this
                   component is active.
    """
    __slots__ = ()

    #
    # Interface
    #

    @property
    def active(self):
        """The active status of this container."""
        return self._active

    @active.setter
    def active(self, value):
        raise AttributeError(
            "Assignment not allowed. Use the "
            "(de)activate method")

    def activate(self, _from_parent_=False):
        """Activate this component."""
        if (not self._active) and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._increment_active()
        self._active = True

    def deactivate(self, _from_parent_=False):
        """Deactivate this component."""
        if self._active and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._decrement_active()
        self._active = False

class IComponentContainer(ICategorizedObject):
    """
    A container of modeling components and possibly other
    containers.
    """
    _is_component = False
    _is_container = True
    _child_storage_delimiter_string = ""
    _child_storage_entry_string = "[%s]"
    __slots__ = ()

    #
    # Interface
    #

    @abc.abstractmethod
    def components(self, *args, **kwds):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def child_key(self, *args, **kwds):
        """Returns the lookup key associated with a child of this
        container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def child(self, *args, **kwds):
        """Returns a child of this container given a storage key."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def children(self, *args, **kwds):
        """A generator over the children of this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def preorder_traversal(self, *args, **kwds):
        """A generator over all descendents in prefix order."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def postorder_traversal(self, *args, **kwds):
        """A generator over all descendents in postfix order."""
        raise NotImplementedError     #pragma:nocover

class _IActiveComponentContainerMixin(IActiveObject):
    """
    To be used as an additional base class in
    IComponentContainer implementations to add fuctionality
    for activating and deactivating the container and its
    children.

    This class is abstract. It assumes any derived class
    declares the following attributes (with or without __slots__):
        - _active: A boolean indicating whethor or not this
                   component is active.
    """
    __slots__ = ()

    # Should be called any time a new active child is added
    # or any timing an existing child's active status
    # changes from False to True
    def _increment_active(self):
        assert self._active >= 0
        if self._active == 0:
            # the container itself is currently
            # inactive, so activate it
            self._active = 1
            # this includes notifying any parent
            # of the change in status
            parent = self.parent
            if parent is not None:
                parent._increment_active()
        self._active += 1

    # Should be called any time an active is child removed
    # or any timing an existing child's active status
    # changes from True to False
    def _decrement_active(self):
        self._active -= 1
        assert self._active >= 1

    #
    # Interface
    #

    @property
    def active(self):
        """The active status of this container."""
        return bool(self._active)

    @active.setter
    def active(self, value):
        raise AttributeError(
            "Assignment not allowed. Use the "
            "(de)activate method.")

    def activate(self, _from_parent_=False):
        """Activate this container. All children of this
        container will be activated and the active flag on
        all ancestors of this container will be set to
        True."""
        assert self._active >= 0
        if (not self.active) and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._increment_active()
        # activate all children
        self._active = 1
        for child in self.children():
            self._active += 1
            if isinstance(child, IActiveObject):
                child.activate(_from_parent_=True)

    def deactivate(self, descend_into=True, _from_parent_=False):
        """Deactivate this container and all of its children."""
        assert self._active >= 0
        if self.active and \
           (not _from_parent_):
            # inform the parent
            parent = self.parent
            if parent is not None:
                parent._decrement_active()
        self._active = 0
        # deactivate all children
        for child in self.children():
            if isinstance(child, IActiveObject):
                child.deactivate(_from_parent_=True)
