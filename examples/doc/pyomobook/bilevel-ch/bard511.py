# bard511.py
from pyomo.environ import *
from pyomo.bilevel import *

M = ConcreteModel()
M.x = Var(bounds=(0,None))
M.sub = SubModel()
M.sub.y = Var(bounds=(0,None))

M.o = Objective(expr=M.x - 4*M.sub.y, sense=minimize)

M.sub.o = Objective(expr=M.sub.y, sense=minimize)
M.sub.c1 = Constraint(expr=-  M.x -   M.sub.y <= -3)
M.sub.c2 = Constraint(expr=-2*M.x +   M.sub.y <=  0)
M.sub.c3 = Constraint(expr= 2*M.x +   M.sub.y <= 12)
M.sub.c4 = Constraint(expr= 3*M.x - 2*M.sub.y <=  4)
