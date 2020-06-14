#logical version for numvalue

import sys
from pyomo.core.expr import boolean_value
sys.modules[__name__] = boolean_value
