#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# The definition of __all__ is a bit funky here, because we want to
# expose symbols in pyomo.core.expr.current that are not included in
# pyomo.core.expr.  The idea is that pyomo.core.expr provides symbols
# that are used by general users, but pyomo.core.expr.current provides
# symbols that are used by developers.
# 
__all__ = []

from pyomo.core.expr import current
__all__.extend(current.__public__)
for obj in current.__public__:
    globals()[obj] = getattr(current, obj)

from pyomo.core.expr import numvalue
__all__.extend(numvalue.__all__)
for obj in numvalue.__all__:
    globals()[obj] = getattr(numvalue, obj)

