# @pyomo:
from pyomo.environ import *
from pyomo.bilevel import *

M = ConcreteModel()
M.x = Var()
M.s = SubModel()
M.s.y = Var()
M.s.s = SubModel()
M.s.s.z = Var()

M.o = Objective(expr=       M.x - 4*M.s.y + 2*M.s.s.z)
M.c1 = Constraint(expr=   - M.x -   M.s.y             <=  -3)
M.c2 = Constraint(expr=  -3*M.x + 2*M.s.y             >= -10)
M.s.o = Objective(expr=     M.x +   M.s.y -   M.s.s.z)
M.s.c1 = Constraint(expr=-2*M.x +   M.s.y - 2*M.s.s.z <=  -1)
M.s.c2 = Constraint(expr= 2*M.x +   M.s.y + 4*M.s.s.z <=  14)
M.s.s.o = Objective(expr=   M.x - 2*M.s.y - 2*M.s.s.z)
M.s.s.c = Constraint(expr=2*M.x -   M.s.y -   M.s.s.z <=   2)

model = M
# @:pyomo

M.pprint()
