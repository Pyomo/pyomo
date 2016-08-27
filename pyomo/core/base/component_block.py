#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("block",
           "block_list",
           "block_dict",
           "StaticBlock")

import logging
from collections import (OrderedDict,
                         defaultdict)
import weakref

from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer,
                                                 IBlockStorage,
                                                 _no_ctype)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
import pyomo.opt

import six
from six import itervalues, iteritems

logger = logging.getLogger('pyomo.core')

class block(IBlockStorage):
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    """An implementation of the IBlockStorage interface."""
    def __init__(self):
        self.__parent = None
        self.__active = True
        self.__byctype = defaultdict(OrderedDict)

    @property
    def _parent(self):
        return self.__parent
    @_parent.setter
    def _parent(self, parent):
        self.__parent = parent

    @property
    def _active(self):
        return self.__active
    @_active.setter
    def _active(self, active):
        self.__active = active

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(self):
    #    pass

    def child_key(self, child):
        if child.ctype in self.__byctype:
            for key, val in iteritems(self.__byctype[child.ctype]):
                if val is child:
                    return key
        raise ValueError("No child entry: %s"
                         % (child))

    #
    # Define the IBlockStorage abstract methods
    #

    def children(self, *args):
        """Iterate over the children of this block. At most one
        argument can be provided which specifies the ctype of
        the children to iterate over. Otherwise, all children
        will be included."""
        if len(args) == 0:
            ctypes = self.__byctype.keys()
        elif len(args) == 1:
            ctypes = args
        else:
            raise TypeError("children expected at most 1 arguments, "
                            "got %d" % (len(args)))
        for ctype in ctypes:
            if ctype in self.__byctype:
                for child in itervalues(self.__byctype[ctype]):
                    yield child

    def components(self,
                   ctype=_no_ctype,
                   active=None,
                   descend_into=True):
        # TODO
        from pyomo.core.base.block import Block
        if ctype is _no_ctype:
            args = ()
        else:
            args = (ctype,)
        _active_flag_name = "active"
        for child in self.children(*args):
            if (active is None) or \
               (getattr(child,
                        _active_flag_name,
                        active) == active):
                if isinstance(child, IBlockStorage):
                    # child is a block
                    yield child
                elif isinstance(child, IComponentContainer):
                    # child is a container that is not a block
                    for component in child.components():
                        if (active is None) or \
                           (getattr(component,
                                    _active_flag_name,
                                    active) == active):
                            yield component
                else:
                    # child is not a container
                    yield child

        if descend_into:
            for child in self.children(Block):
                if (active is None) or \
                   (not active) or \
                   getattr(child,
                           _active_flag_name,
                           True):
                    if isinstance(child, IBlockStorage):
                        # child is a block
                        for component in child.components(
                                ctype=ctype,
                                active=active,
                                descend_into=descend_into):
                            yield component
                    else:
                        # child is a container of _only_ blocks
                        for _comp in child.components():
                            if (active is None) or \
                               (not active) or \
                               getattr(_comp,
                                       _active_flag_name,
                                       True):
                                for component in _comp.components(
                                        ctype=ctype,
                                        active=active,
                                        descend_into=descend_into):
                                    yield component

    def blocks(self,
               active=None,
               descend_into=True):
        # TODO
        from pyomo.core.base.block import Block
        if (active is None) or \
           self.active == active:
            yield self
        for component in self.components(ctype=Block,
                                         active=active,
                                         descend_into=descend_into):
            yield component

    #
    # Interface
    #

    def __setattr__(self, name, component):
        if isinstance(component, (IComponent, IComponentContainer)):
            if component._parent is None:
                if name in self.__dict__:
                    logger.warning(
                        "Implicitly replacing the component attribute "
                        "%s (type=%s) on block with a new Component "
                        "(type=%s).\nThis is usually indicative of a "
                        "modelling error.\nTo avoid this warning, delete "
                        "the original component from the block before "
                        "assigning a new component with the same name."
                        % (name,
                           type(getattr(self, name)),
                           type(component)))
                    delattr(self, name)
                self.__byctype[component.ctype][name] = component
                component._parent = weakref.ref(self)
            elif (name in self.__byctype[component.ctype]) and \
                 (self.__byctype[component.ctype][name] is component):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                pass
            else:
                raise ValueError(
                    "Invalid assignment to %s type with name '%s' "
                    "at entry %s. A parent container has already "
                    "been assigned to the component being "
                    "inserted: %s"
                    % (self.__class__.__name__,
                       self.name(True),
                       name,
                       component.parent.name(True)))
        super(block, self).__setattr__(name, component)

    def __delattr__(self, name):
        component = self.__dict__[name]
        if isinstance(component, (IComponent, IComponentContainer)):
            del self.__byctype[component.ctype][name]
            if len(self.__byctype[component.ctype]) == 0:
                del self.__byctype[component.ctype]
            component._parent = None
        super(block, self).__delattr__(name)

    # TODO
    #def write(self, ...):
    #def clone(self, ...):

class block_list(ComponentList,
                 _IActiveComponentContainer):
    """A list-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(block_list, self).__init__(*args, **kwds)

class block_dict(ComponentDict,
                 _IActiveComponentContainer):
    """A dict-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(block_dict, self).__init__(*args, **kwds)

class StaticBlock(IBlockStorage):
    """
    A helper class for implementing blocks with a static
    set of components using __slots__. Derived classes
    should assign a static set of components to the instance
    in the __init__ method before calling this base class's
    __init__ method. The set of components should be
    identified using __slots__.

    Note that this implementation is not designed for
    class hierarchies that extend more than one level
    beyond this class.
    """
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_components",
                 "__weakref__")
    def __init__(self):
        self._parent = None
        self._active = True
        # Note: We are iterating over the derived
        #       class's __slots__ here.
        for name in self.__slots__:
            obj = getattr(self, name)
            if isinstance(obj, (IComponent, IComponentContainer)):
                obj._parent = weakref.ref(self)

    #
    # Define the IComponentContainer abstract methods
    #

    # overridden by the IBlockStorage interface
    #def components(self):
    #    pass

    def child_key(self, child):
        for key in self.__slots__:
            if getattr(self, key) is child:
                return key
        raise ValueError("No child entry: %s"
                         % (child))

    # overridden by the IBlockStorage interface
    #def children(self):
    #    pass

    #
    # Define the IBlockStorage abstract methods
    #

    def children(self, *args):
        """Iterate over the children of this block. At most one
        argument can be provided which specifies the ctype of
        the children to iterate over. Otherwise, all children
        will be included."""
        if len(args) == 0:
            for name in self.__slots__:
                child = getattr(self, name)
                if isinstance(child, ICategorizedObject):
                    yield child
        elif len(args) == 1:
            ctype = args[0]
            for name in self.__slots__:
                child = getattr(self, name)
                if isinstance(child, ICategorizedObject) and \
                   child.ctype == ctype:
                    yield child
        else:
            raise TypeError("children expected at most 1 arguments, "
                            "got %d" % (len(args)))

    def components(self, *args, **kwds):
        return block.components(self, *args, **kwds)

    def blocks(self, *args, **kwds):
        return block.blocks(self, *args, **kwds)
