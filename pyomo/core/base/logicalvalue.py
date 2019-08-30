#logical version for numvalie

import sys
from pyomo.core.expr import logicalvalue
sys.modules[__name__] = logicalvalue
