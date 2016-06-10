# @pyomo:
from pyomo.environ import *
from pyomo.bilevel import *

M = ConcreteModel()
M.x = Var()
M.y = Var()
M.z = Var()

M.o = Objective(expr=     M.x - 4*M.y + 2*M.z)
M.c1 = Constraint(expr=-  M.x -   M.y <=  -3)
M.c2 = Constraint(expr=-3*M.x + 2*M.y >= -10)

M.s = SubModel(fixed=M.x)
M.s.o = Objective(expr=     M.x + M.y -   M.z)
M.s.c1 = Constraint(expr=-2*M.x + M.y - 2*M.z <= -1)
M.s.c2 = Constraint(expr= 2*M.x + M.y + 4*M.z <= 14)

M.s.s = SubModel(fixed=M.y)
M.s.s.o = Objective(expr=   M.x - 2*M.y - 2*M.z)
M.s.s.c = Constraint(expr=2*M.x -   M.y -   M.z <= 2)

model = M
# @:pyomo

M.pprint()
