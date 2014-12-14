from pyomo.environ import *

M = ConcreteModel()
M.x = Var([1,2,3], within=Boolean)

M.o = Objective(expr=summation(M.x))
M.c1 = Constraint(expr=4*M.x[1]+M.x[2] >= 1)
M.c2 = Constraint(expr=M.x[2]+4*M.x[3] >= 1)

model=M
