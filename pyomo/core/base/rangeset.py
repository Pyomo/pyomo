#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['RangeSet']

from .set import RangeSet

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.core.base.rangeset module is deprecated.  '
    'Import RangeSet objects from pyomo.core.base.set or pyomo.core.',
    version='5.7')
