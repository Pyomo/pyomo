#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.core.kernel.component_set module is deprecated.  '
    'Import ComponentSet from pyomo.common.collections.',
    version='5.7.1')
