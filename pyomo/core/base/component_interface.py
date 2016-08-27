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
           '_IActiveComponentContainer',
           'IBlockStorage')

import sys
import abc
import copy
import weakref

import six
from six.moves import xrange

def _not_implemented(*args, **kwds):
    raise NotImplementedError     #pragma:nocover

def _abstract_readwrite_property(**kwds):
    return abc.abstractproperty(fget=_not_implemented,
                                fset=_not_implemented,
                                **kwds)

def _abstract_readonly_property(**kwds):
    return abc.abstractproperty(fget=_not_implemented,
                                **kwds)

_no_ctype = object()

class ICategorizedObject(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for objects that maintain a weak reference to a parent
    storage object and have a category type."""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _ctype = _abstract_readonly_property(
        doc=("A category type. Used by the parent of "
             "this component to categorize it."))

    _parent = _abstract_readwrite_property(
        doc=("A weak reference to parent object of "
             "type IComponentContainer or None."))

    @property
    def ctype(self):
        """The component category."""
        return self._ctype

    @property
    def parent(self):
        """Returns the parent component"""
        if isinstance(self._parent, weakref.ReferenceType):
            return self._parent()
        else:
            return self._parent

    @property
    def parent_block(self):
        """Return the parent storage block for this component"""
        parent = self.parent
        while (parent is not None) and \
              (not isinstance(parent, IBlockStorage)):
            parent = parent.parent
        return parent

    @property
    def root_block(self):
        """Returns the root storage block above this component"""
        root_block = None
        if isinstance(self, IBlockStorage):
            root_block = self
        parent_block = self.parent_block
        while parent_block is not None:
            root_block = parent_block
            parent_block = parent_block.parent_block
        return root_block

    def name(self,
              fully_qualified=False,
              name_buffer=None,
              convert=str):
        """
        Generate a name for the component container.

        Args:
            fully_qualified (bool): Generate full name from
                nested block names. Default is False.
            name_buffer (dict): A temporary storage
                dictionary that can be used to optimize
                iterative name generation.
            convert (function): A function that converts
                a storage key into a string representation.

        Returns:
            A string representing the name of the component
            container.
        """
        # TODO
        assert name_buffer is None

        parent = self.parent
        if parent is None:
            return None

        key = parent.child_key(self)
        if isinstance(parent, IBlockStorage):
            name = "%s" % (convert(key))
            prefix = "."
            parent_is_block = True
        else:
            assert isinstance(parent, IComponentContainer)
            name = "[%s]" % (convert(key))
            prefix = ""
            parent_is_block = False

        if (not fully_qualified) and \
           parent_is_block:
            return name
        else:
            parent_name = parent.name(fully_qualified=fully_qualified,
                                      name_buffer=name_buffer)
            if parent_name is not None:
                return parent_name + prefix + name
            else:
                return name

class IActiveObject(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for objects that support activate/deactivate
    semantics."""
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
    def activate(self):
        """Set the active attribute to True"""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def deactivate(self):
        """Set the active attribute to False"""
        raise NotImplementedError     #pragma:nocover

class IComponent(ICategorizedObject):
    """
    Interface for components that can be stored inside
    objects of type IComponentContainer."""
    __slots__ = ()

    #
    # Interface
    #

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Write the component to a buffer"""
        if ostream is None:
            ostream = sys.stdout
        ostream.write(self.__str__())

    def __str__(self):
        name = self.name(True)
        if name is None:
            return "<"+self.__class__.__name__+">"
        else:
            return name

class _IActiveComponent(IActiveObject):
    """
    To be used as an additional base class in IComponent
    implementations to add fuctionality for activating and
    deactivating the component.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _active = _abstract_readwrite_property(
        doc=("A boolean indicating whethor or not any "
             "components stored below this container "
             "are active."))

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

    def activate(self):
        """Activate this component. The active flag on the
        parent container (if one exists) is set to True."""
        self._active = True
        # the parent must also become active
        parent = self.parent
        while (parent is not None) and \
              (not parent._active):
            parent._active = True
            parent = parent.parent

    def deactivate(self):
        """Deactivate this component."""
        self._active = False

class IComponentContainer(ICategorizedObject):
    """A container of modeling components."""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    #
    # Interface
    #

    @abc.abstractmethod
    def components(self):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def child_key(self, component):
        """The lookup key associated with a child of this
        container."""
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def children(self):
        """A generator over the children of this container."""
        raise NotImplementedError     #pragma:nocover

class _IActiveComponentContainer(IActiveObject):
    """
    To be used as an additional base class in ComponentContainer
    implementations to add fuctionality for activating and
    deactivating the container and its children.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _active = _abstract_readwrite_property()

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
            "(de)activate method.")

    def activate(self):
        """Activate this container. All children of this
        container will be activated and the active flag on
        all ancestors of this container will be set to
        True."""
        self._active = True
        # all of ancestors must also become active
        parent = self.parent
        while (parent is not None) and \
              (not parent._active):
            parent._active = True
            parent = parent.parent
        # all children must be activated
        for child in self.children():
            if isinstance(child, IActiveObject):
                child.activate()

    def deactivate(self):
        """Deactivate this container and all its children."""
        self._active = False
        # all children must be deactivated
        for child in self.children():
            if isinstance(child, IActiveObject):
                child.deactivate()

class IBlockStorage(IComponentContainer,
                    _IActiveComponentContainer):
    """A container that stores multiple types."""
    __slots__ = ()

    #
    # Interface
    #

    @abc.abstractmethod
    def blocks(self):
        raise NotImplementedError     #pragma:nocover

    #
    # These methods are already declared abstract on
    # IComponentContainer, but we redeclare them here to
    # point out that the can accept a ctype
    #

    @abc.abstractmethod
    def components(self, ctype=_no_ctype):
        raise NotImplementedError     #pragma:nocover

    @abc.abstractmethod
    def children(self, ctype=_no_ctype):
        raise NotImplementedError     #pragma:nocover
