#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.global_set import (
    UnindexedComponent_index, UnindexedComponent_set
)

class _TransformedDisjunctData(_BlockData):
    __slots__ = ('_src_disjunct',)
    __autoslot_mappers__ = {'_src_disjunct': AutoSlots.weakref_mapper}

    @property
    def src_disjunct(self):
        return self._src_disjunct

    def __init__(self, component):
        _BlockData.__init__(self, component)
        # pointer to the Disjunct whose transformation block this is.
        self._src_disjunct = None

class _TransformedDisjunct(Block):
    _ComponentDataClass = _TransformedDisjunctData

    def __new__(cls, *args, **kwds):
        if cls != _TransformedDisjunct:
            return super(_TransformedDisjunct, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return _ScalarTransformedDisjunct.__new__(
                _ScalarTransformedDisjunct)
        else:
            return _IndexedTransformedDisjunct.__new__(
                _IndexedTransformedDisjunct)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('ctype', Block)
        Block.__init__(self, *args, **kwargs)

class _ScalarTransformedDisjunct(_TransformedDisjunctData,
                                 _TransformedDisjunct):
    def __init__(self, *args, **kwds):
        _TransformedDisjunctData.__init__(self, self)
        _TransformedDisjunct.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index

class _IndexedTransformedDisjunct(_TransformedDisjunct):
    pass
