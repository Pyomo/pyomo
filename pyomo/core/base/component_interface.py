#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('IComponent',
           '_IActiveComponent',
           'IComponentContainer',
           '_IActiveComponentContainer',
           'IBlockStorage')

import abc
import copy
import weakref

import six
from six.moves import xrange

def _not_implemented(*args, **kwds):
    raise NotImplementedError

def _abstract_readwrite_property(**kwds):
    return abc.abstractproperty(fget=_not_implemented,
                                fset=_not_implemented,
                                **kwds)

def _abstract_readonly_property(**kwds):
    return abc.abstractproperty(fget=_not_implemented,
                                **kwds)

class IParentPointerObject(six.with_metaclass(abc.ABCMeta, object)):
    """Interface Objects that maintain a weak reference to a parent
    storage object."""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _parent = _abstract_readwrite_property()

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
        parent = self.parent_block
        root = parent
        while parent is not None:
            root = parent
            parent = parent.parent_block
        if (root is None) and \
           isinstance(self, IBlockStorage):
            root = self
        return root

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

class IComponent(IParentPointerObject):
    """
    Interface for components that can be stored on a
    component block or inside a component container.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _ctype = _abstract_readonly_property()

    #
    # Interface
    #

    @property
    def ctype(self):
        """Returns the component type"""
        return self._ctype

class _IActiveComponent(object):
    """
    To be used as an additional base class in IComponent
    implementations to add fuctionality for activating and
    deactivating the component on a model.
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
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to a specified value."""
        raise AttributeError("Assignment not allowed. Use the "
                             "(de)activate method")

    def activate(self):
        """Set the active attribute to True"""
        self._active = True
        parent = self.parent
        # if a component becomes active, the parent
        # must also become active
        if parent is not None:
            self.parent._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False

class IComponentContainer(IParentPointerObject):
    """A container of modeling components."""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    _ctype = _abstract_readonly_property()

    #
    # Interface
    #

    @property
    def ctype(self):
        """The component type."""
        return self._ctype

    @abc.abstractmethod
    def components(self):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError

    @abc.abstractmethod
    def child_key(self, component):
        """The lookup key associated with a child of this
        container."""
        raise NotImplementedError

    @abc.abstractmethod
    def children(self):
        """A generator over the children of this container."""
        raise NotImplementedError

class _IActiveComponentContainer(object):
    """
    To be used as an additional base class in Component
    implementations to add fuctionality for activating and
    deactivating the component on a model.
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
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        raise AttributeError(
            "Assignment not allowed. Use the "
            "(de)activate methods.")

    def activate(self):
        """Set the active attribute to True"""
        self._active = True
        for child in self.children():
            child.activate()

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False
        for child in self.children():
            children.deactivate()

class IBlockStorage(IComponentContainer, _IActiveComponent):
    __slots__ = ()

    #
    # Interface
    #

    @abc.abstractmethod
    def components(self, ctype):
        raise NotImplementedError

    @abc.abstractmethod
    def children(self, ctype):
        raise NotImplementedError

    @abc.abstractmethod
    def blocks(self):
        raise NotImplementedError

if __name__ == "__main__":

    class Cont(IComponentContainer, _IActiveComponentContainer):
        __slots__ = ("_parent",
                     "_ctype",
                     "_active",
                     "_d",
                     "_inv_d",
                     "__weakref__")
        def __init__(self, ctype):
            self._parent = None
            self._ctype = ctype
            self._active = True
            self._d = {}
            self._inv_d = {}
        def child_key(self, component):
            return self._inv_d[id(component)]
        def components(self,
                       ctype,
                       active=None,
                       sort=False,
                       descend_into=True,
                       descent_order=None):
            # TODO
            assert descent_order is None
            assert active is None
            assert sort is False
            assert descend_into is True
            for ctype in ctypes:
                for component in itervalues(self._d.get(ctype, {})):
                    yield component
        def children(self):
            for ctype in self._d:
                for child in self._d.get(ctype, {}):
                    yield child

        def insert(self, key, component):
            if component._parent is not None:
                raise ValueError(
                    "Can not store a component that has already "
                    "been assigned a parent")
            if component.ctype != self.ctype:
                raise TypeError(
                    "Component must have ctype=%s"
                    % (str(self.ctype)))
            self._d[key] = component
            self._inv_d[id(component)] = key
            component._parent = weakref.ref(self)
            self._active |= component.active
        def remove(self, component=None, key=None):
            if component is not None:
                assert key is None
                key = self._inv_d[id(component)]
            else:
                assert key is not None
                component = self._d[key]
            del self._d[key]
            del self._inv_d[id(component)]
            component._parent = None

    class C1(IComponent, _IActiveComponent):
        __slots__ = ("_parent",
                     "_ctype",
                     "_active")
        def __init__(self):
            self._parent = None
            self._ctype = "Variable"
            self._active = True

    class Block(IBlockStorage):
        __slots__ = ("_parent",
                     "_ctype",
                     "_active",
                     "_d",
                     "_inv_d",
                     "_names",
                     "__weakref__")
        def __init__(self):
            self._parent = None
            self._ctype = "Block"
            self._active = True
            self._d = {}
            self._inv_d = {}
            self._names = set()
        def child_key(self, component):
            return self._inv_d[id(component)]
        def components(self):
            return list(self._d.values())
        children = components
        def blocks(self):
            raise NotImplementedError
        def insert(self, key, component):
            if component._parent is not None:
                raise ValueError(
                    "Can not store a component that has already "
                    "been assigned a parent")
            if key in self._names:
                # TODO
                print("WARNING: Overwriting existing component!")
                self.remove(key=key)
            if component.ctype not in self._d:
                self._d[component.ctype] = {}
            self._d[component.ctype][key] = component
            self._inv_d[id(component)] = key
            self._names.add(key)
            component._parent = weakref.ref(self)
        def remove(self, component=None, key=None):
            if component is not None:
                assert key is None
                key = self._inv_d[id(component)]
            else:
                assert key is not None
                component = self._d[component.ctype][key]
            del self._d[component.ctype][key]
            del self._inv_d[id(component)]
            self._names.remove(key)
            component._parent = None

    print("\nCreating c")
    c = C1()
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.active
    c.deactivate()
    assert not c.active
    assert c.parent is None
    assert c.parent_block is None
    assert c.root_block is None

    print("\nCreating C")
    C = Cont("Variable")
    print(C.name())
    print(C.name(fully_qualified=True))
    assert C.active
    C.deactivate()
    assert not C.active
    assert C.parent is None
    assert C.parent_block is None
    assert C.root_block is None
    c.activate()
    assert c.active
    assert not C.active
    assert C.parent is None
    assert C.parent_block is None
    assert C.root_block is None
    C.insert("c", c)
    assert C.active
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.active
    assert c.parent is C
    assert c.parent_block is None
    assert c.root_block is None

    print("\nCreating b")
    b = Block()
    print(b.name())
    print(b.name(fully_qualified=True))
    assert b.active
    assert b.parent is None
    assert b.parent_block is None
    assert b.root_block is b
    b.insert("C", C)
    print(C.name())
    print(C.name(fully_qualified=True))
    assert C.parent is b
    assert C.parent_block is b
    assert C.root_block is b
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.parent is C
    assert c.parent_block is b
    assert c.root_block is b

    print("\nCreating B")
    B = Cont("Block")
    print(B.name())
    print(B.name(fully_qualified=True))
    assert B.active
    assert B.parent is None
    assert B.parent_block is None
    assert B.root_block is None
    B.insert("b", b)
    print(b.name())
    print(b.name(fully_qualified=True))
    assert b.parent is B
    assert b.parent_block is None
    assert b.root_block is b
    print(C.name())
    print(C.name(fully_qualified=True))
    assert C.parent is b
    assert C.parent_block is b
    assert C.root_block is b
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.parent is C
    assert c.parent_block is b
    assert c.root_block is b

    print("\nCreating M")
    M = Block()
    print(M.name())
    print(M.name(fully_qualified=True))
    assert M.active
    assert M.parent is None
    assert M.parent_block is None
    assert M.root_block is M
    M.insert("B", B)
    print(B.name())
    print(B.name(fully_qualified=True))
    assert B.parent is M
    assert B.parent_block is M
    assert B.root_block is M
    print(b.name())
    print(b.name(fully_qualified=True))
    assert b.parent is B
    assert b.parent_block is M
    assert b.root_block is M
    print(C.name())
    print(C.name(fully_qualified=True))
    assert C.parent is b
    assert C.parent_block is b
    assert C.root_block is M
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.parent is C
    assert c.parent_block is b
    assert c.root_block is M
    try:
        M.insert("b", b)
    except ValueError:
        pass
    else:
        assert False
    print("\nRelocating b")
    B.remove(b)
    M.insert("b", b)
    print(b.name())
    print(b.name(fully_qualified=True))
    assert b.parent is M
    assert b.parent_block is M
    assert b.root_block is M
    print(C.name())
    print(C.name(fully_qualified=True))
    assert C.parent is b
    assert C.parent_block is b
    assert C.root_block is M
    print(c.name())
    print(c.name(fully_qualified=True))
    assert c.parent is C
    assert c.parent_block is b
    assert c.root_block is M
