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

import weakref

from pyomo.core.base.component_interface import (IComponent,
                                                 IComponentContainer,
                                                 IBlockStorage)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
import pyomo.opt

import six
from six import itervalues, iteritems

class block(IBlockStorage):
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    """An implementation of the IBlockStorage interface."""
    def __init__(self):
        self.__parent = None
        self.__active = True
        self.__byctype = {}

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

    def component_entry_key(self, component):
        for key, val in iteritems(self.__byctype[component.ctype]):
            if val is component:
                return key
        raise ValueError("No component entry: %s"
                         % (component))

    #
    # Define the IBlockStorage abstract methods
    #

    def components(self,
                   ctype,
                   active=None,
                   sort=False,
                   descend_into=True,
                   descent_order=None):
            # TODO
#            assert descent_order is None
#            assert active is None
#            assert sort is False
#            assert descend_into is True
            for component in itervalues(self.__byctype.get(ctype, {})):
                if isinstance(component, IComponentContainer):
                    for component in component.components():
                        yield component
                else:
                    yield component

    def blocks(self,
               active=None,
               sort=False,
               descend_into=True,
               descent_order=None):
        from pyomo.core.base.block import Block
        assert descent_order is None
        if active is None:
            yield self
            for block in self.components(Block,
                                         active=None,
                                         sort=sort,
                                         descend_into=descend_into,
                                         descent_order=descent_order):
                yield block
        elif active:
            if not self.active:
                return
            yield self
            for block in self.components(Block,
                                         active=True,
                                         sort=sort,
                                         descend_into=descend_into,
                                         descent_order=descent_order):
                yield block
        else:
            all_active_ids = set()
            all_active_ids.update(
                id(block) for block in self.components(
                    Block,
                    active=True,
                    descend_into=descend_into))
            if not self.active:
                yield self
            for block in self.components(Block,
                                         active=None,
                                         sort=sort,
                                         descend_into=descend_into,
                                         descent_order=descent_order):
                if id(block) not in all_active_ids:
                    yield block

    #
    # Interface
    #

    def __setattr__(self, name, component):
        if isinstance(component, (IComponent, IComponentContainer)):
            if component._parent is None:
                if name in self.__dict__:
                    # TODO
                    print("WARNING: Overwriting existing component!")
                    self.del_component(name=name)
                if component.ctype not in self.__byctype:
                    self.__byctype[component.ctype] = {}
                self.__byctype[component.ctype][name] = component
                component._parent = weakref.ref(self)
            elif (name in self.__byctype.get(component.ctype, {})) and \
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
                       self.cname(True),
                       name,
                       component.parent.cname(True)))
        super(block, self).__setattr__(name, component)
    add_component = __setattr__

    def __delattr__(self, name):
        component = self.__dict__[name]
        if isinstance(component, (IComponent, IComponentContainer)):
            del self.__byctype[component.ctype][name]
            component._parent = None
        super(block, self).__delattr__(name)
    del_component = __delattr__

    def write(self,
              filename,
              format=None,
              solver_capability=None,
              io_options=None):
        """
        Write the optimization block to a file, with a given format.
        """
        # Guess the format if none is specified
        if format is None:
            format = guess_format(filename)
        with pyomo.opt.WriterFactory(format) as writer:
            if writer is None:
                raise ValueError(
                    "Cannot write model in format '%s': no model "
                    "writer registered for that format"
                    % str(format))
            if solver_capability is None:
                solver_capability = lambda x: True
            if io_options is None:
                io_options = {}
            filename_, symbol_map = writer(self,
                                           filename,
                                           solver_capability,
                                           io_options)
        assert filename_ == filename
        return filename, symbol_map

class block_list(ComponentList):
    """A list-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(block_list, self).__init__(*args, **kwds)

class block_dict(ComponentDict):
    """A dict-style container for blocks."""
    # To avoid a circular import, for the time being, this
    # property will be set in block.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(block_dict, self).__init__(*args, **kwds)

class StaticBlock(IBlockStorage):
    """
    A helper class for implementing blocks with a static
    set of components using __slots__. Derived classes
    should assign a static set of component to the instance
    in the __init__ method before calling this base class's
    __init__ method. The set of components should be
    identified using __slots__.
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

    def component_entry_key(self, component):
        for key in self.__slots__:
            if getattr(self, key) is component:
                return key
        raise ValueError("No component entry: %s"
                         % (component))

    #
    # Define the IBlockStorage abstract methods
    #

    def components(self,
                   ctype,
                   active=None,
                   sort=False,
                   descend_into=True,
                   descent_order=None):
        # TODO
#       assert descent_order is None
#       assert active is None
#       assert sort is False
#       assert descend_into is True
        for name in self.__slots__:
            obj = getattr(self, name)
            if hasattr(obj, "ctype") and \
               obj.ctype == ctype:
                yield component

    def blocks(self, *args, **kwds):
        return block.blocks(self, *args, **kwds)

    #
    # Interface
    #

    def write(self, *args, **kwds):
        return block.write(self, *args, **kwds)

if __name__ == "__main__":
    def _fmt(num, suffix='B'):
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1000.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1000.0
        return "%.1f %s%s" % (num, 'Yi', suffix)
    import pympler.asizeof
    from pyomo.core.base.block import Block
    from pyomo.core.base import RangeSet

    block_bytes = pympler.asizeof.asizeof(block())
    Block_bytes = pympler.asizeof.asizeof(Block())
    print("block: %s" % (_fmt(block_bytes)))
    print("Block: %s (%.2fx)"
          % (_fmt(Block_bytes), Block_bytes/float(block_bytes)))

    N = 1000
    block_list_bytes = pympler.asizeof.asizeof(
        block_list(block() for i in range(N)))
    block_dict_bytes = pympler.asizeof.asizeof(
        block_dict((i, block()) for i in range(N)))
    def _indexed_block_rule(b, i):
        return b
    index = RangeSet(N)
    index.construct()
    indexed_Block = Block(index, rule=_indexed_block_rule)
    indexed_Block.construct()
    indexed_Block_bytes = pympler.asizeof.asizeof(indexed_Block) 
    print("")
    print("block_list{1000}: %s" % (_fmt(block_list_bytes)))
    print("block_dict{1000}: %s" % (_fmt(block_dict_bytes)))
    print("Indexed Block{1000}: %s (%.2fx)"
          % (_fmt(indexed_Block_bytes),
             indexed_Block_bytes/float(block_list_bytes)))
