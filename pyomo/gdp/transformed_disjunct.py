#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.core.base.block import BlockData, IndexedBlock
from pyomo.core.base.global_set import UnindexedComponent_index, UnindexedComponent_set


class _TransformedDisjunctData(BlockData):
    __slots__ = ('_src_disjunct',)
    __autoslot_mappers__ = {'_src_disjunct': AutoSlots.weakref_mapper}

    @property
    def src_disjunct(self):
        return None if self._src_disjunct is None else self._src_disjunct()

    def __init__(self, component):
        BlockData.__init__(self, component)
        # pointer to the Disjunct whose transformation block this is.
        self._src_disjunct = None


class _TransformedDisjunct(IndexedBlock):
    _ComponentDataClass = _TransformedDisjunctData
