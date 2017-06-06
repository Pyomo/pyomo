#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.set_types import *
from pyomo.core.kernel.set_types import (_VirtualSet,
                                         _virtual_sets)
from pyomo.core.base.sets import SimpleSet

# we probably do not need _VirtualSet as a base class in this case
class _AMLVirtualSet(SimpleSet, _VirtualSet):
    def __init__(self, *args, **kwds):
        SimpleSet.__init__(self, *args, **kwds)
        self.virtual=True
        self.concrete=False

    def data(self):
        raise TypeError("Cannot access data for a virtual set")

class _AnySet(_AMLVirtualSet):
    """A virtual set that allows any value"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        _AMLVirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        return True

class _EmptySet(_AMLVirtualSet):
    """A virtual set that allows no values"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        _AMLVirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        return False

class _AnySetWithNone(_AnySet):
    """A virtual set that allows any value (including None)"""

    def __contains__(self, element):
        logger.warning("DEPRECATION WARNING: Use the Any set instead of AnyWithNone")
        return True

#
# Concrete instances of the standard sets
#
Any=_AnySet(name="Any", doc="A set of any data")
EmptySet=_EmptySet(name="EmptySet", doc="A set of no data")
AnyWithNone=_AnySetWithNone(name="AnyWithNone", doc="A set of any data (including None)")
