# bard511.py

from pyomo.environ import *
from pyomo.bilevel import *

M = ConcreteModel()
M.x = Var(bounds=(0,None))
M.y = Var(bounds=(0,None))
M.o = Objective(expr=M.x - 4*M.y, sense=minimize)

M.sub = SubModel(fixed=M.x)
M.sub.o = Objective(expr=M.y, sense=minimize)
M.sub.c1 = Constraint(expr=-  M.x -   M.y <= -3)
M.sub.c2 = Constraint(expr=-2*M.x +   M.y <=  0)
M.sub.c3 = Constraint(expr= 2*M.x +   M.y <= 12)
M.sub.c4 = Constraint(expr= 3*M.x - 2*M.y <=  4)

model = M
