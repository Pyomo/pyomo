# Example 5.1.1 from
#
# Practical Bilevel Optimization: Algorithms and Applications
#   Jonathan Bard

import pyomo.modeling
from pyomo.core import *
from pyomo.bilevel import *

M = ConcreteModel()
M.x = Var(bounds=(0,None))
M.y = Var(bounds=(0,None))
M.o = Objective(expr=M.x - 4*M.y)

M.sub = SubModel(fixed=M.x)
M.sub.o = Objective(expr=M.y)
M.sub.c1 = Constraint(expr=-M.x - M.y <= -3)
M.sub.c2 = Constraint(expr=-2*M.x + M.y <= 0)
M.sub.c3 = Constraint(expr=2*M.x + M.y <= 12)
M.sub.c4 = Constraint(expr=-3*M.x + 27 <= -4)

M.pprint()

model=M
