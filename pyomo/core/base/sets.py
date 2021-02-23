#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO
# . rename 'filter' to something else
# . confirm that filtering is efficient

__all__ = ['Set', 'set_options', 'simple_set_rule', 'SetOf']

from .set import (
    process_setarg, set_options, simple_set_rule,
    _SetDataBase, _SetData, Set, SetOf, IndexedSet,
)

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.core.base.sets module is deprecated.  '
    'Import Set objects from pyomo.core.base.set or pyomo.core.',
    version='5.7')
