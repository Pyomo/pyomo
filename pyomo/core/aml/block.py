#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging

from pyomo.core.base.misc import apply_indexed_rule

from pyomo.core.base.component_interface import IComponent
from pyomo.core.base.component_block import (IBlockStorage,
                                             block,
                                             block_dict)

from pyomo.core.aml.constructed_object import \
    IConstructedObject
from pyomo.core.aml.indexed_component_container import \
    _IndexedComponentContainerMixin

logger = logging.getLogger('pyomo.core')

class _AddToConstructBlock(IConstructedObject, block):
    """A block that constructs objects when they are added
    to it, unless the block itself is not constructed."""

    def __init__(self):
        self._constructed = True
        block.__init__(self)

    def __setattr__(self, name, obj):
        # code that processes a temporary virtual set would
        # go here (probably)
        # ...

        # call the base class to add the object
        super(_AddToConstructBlock, self).__setattr__(name, obj)

        # now construct the object
        if self.constructed:
            if isinstance(obj, IConstructedObject) and \
               (not obj.constructed):
                obj.construct()

class Block(IConstructedObject):

    def __new__(cls, *args, **kwds):
        if cls != Block:
            return super(Block, cls).__new__(cls)
        if args == () or ((type(args[0]) == set) and \
                          (len(args) == 1)):
            return SimpleBlock.__new__(SimpleBlock)
        else:
            return IndexedBlock.__new__(IndexedBlock)

    def __init__(self, **kwargs):
        self._constructed = False
        self._rule = kwargs.pop('rule', None )
        self._options = kwargs.pop('options', None )
        _concrete = kwargs.pop('concrete',False)
        if _concrete:
            # Call self.construct() as opposed to just
            # setting the _constructed flag so that the base
            # class construction procedure fires (this picks
            # up any construction rule that the user may
            # provide)
            self.construct()

    def construct(self, data=None):
        """
        Initialize the block
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing %s '%s', from data=%s",
                         self.__class__.__name__,
                         self.name,
                         str(data))
        if self._constructed:
            return

        for child in self.children():
            if isinstance(child, IConstructedObject) and \
               (not child.constructed):
                child.construct(data=data)

        self._construct_impl()
        self._constructed = True

    def _construct_impl(self):
        raise NotImplementedError      #pragma:nocover

class SimpleBlock(_AddToConstructBlock,
                  Block):
    """A single block."""

    def __init__(self, *args, **kwds):
        assert len(args) == 0
        self._constructed = False
        _AddToConstructBlock.__init__(self)
        Block.__init__(self, **kwds)

    def _construct_impl(self):

        if self._rule is None:
            return

        obj = apply_indexed_rule(
            None, self._rule, self, idx, self._options)
        assert obj is None

class IndexedBlock(_IndexedComponentContainerMixin,
                   block_dict,
                   Block):
    """An dictionary of blocks."""

    def __init__(self, *args, **kwds):
        _IndexedComponentContainerMixin.__init__(self, *args)
        block_dict.__init__(self)
        Block.__init__(self, **kwds)

    def _construct_impl(self):

        if self._rule is None:
            return

        for idx in self.index:
            self[idx] = _obj = _AddToConstructBlock()
            obj = apply_indexed_rule(
                None, self._rule, _obj, idx, self._options)
            assert obj is None
