#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO: this import is for historical backwards compatibility and should
# probably be removed
from pyomo.common.collections import ComponentMap

from pyomo.core.expr.numvalue import *
from pyomo.core.expr.boolean_value import (
    as_boolean, BooleanConstant, BooleanValue,
    native_logical_values)
from pyomo.core.kernel.objective import (minimize,
                                         maximize)
from pyomo.core.base.config import PyomoOptions

from pyomo.core.base.expression import *
from pyomo.core.base.label import *

#
# Components
#
from pyomo.core.base.component import *
import pyomo.core.base.indexed_component
from pyomo.core.base.action import *
from pyomo.core.base.check import *
from pyomo.core.base.set import (
    Set, SetOf, simple_set_rule, RangeSet,
)
from pyomo.core.base.param import *
from pyomo.core.base.var import *
from pyomo.core.base.boolean_var import (
    BooleanVar, _BooleanVarData, _GeneralBooleanVarData,
    BooleanVarList, SimpleBooleanVar)
from pyomo.core.base.constraint import *
from pyomo.core.base.logical_constraint import (
    LogicalConstraint, LogicalConstraintList, _LogicalConstraintData)
from pyomo.core.base.objective import *
from pyomo.core.base.connector import *
from pyomo.core.base.sos import *
from pyomo.core.base.piecewise import *
from pyomo.core.base.suffix import *
from pyomo.core.base.external import *
from pyomo.core.base.symbol_map import *
from pyomo.core.base.reference import Reference
#
from pyomo.core.base.set_types import *
from pyomo.core.base.misc import *
from pyomo.core.base.block import *
from pyomo.core.base.PyomoModel import *
from pyomo.core.base.plugin import *
#
import pyomo.core.base._pyomo
#
import pyomo.core.base.util

from pyomo.core.base.instance2dat import *

# These APIs are deprecated and should be removed in the near future
from pyomo.core.base.set import (
    set_options, RealSet, IntegerSet, BooleanSet,
)

#
# This is a hack to strip out modules, which shouldn't have been included in these imports
#
import types
_locals = locals()
__all__ = [__name for __name in _locals.keys() if (not __name.startswith('_') and not isinstance(_locals[__name],types.ModuleType)) or __name == '_' ]
__all__.append('pyomo')
