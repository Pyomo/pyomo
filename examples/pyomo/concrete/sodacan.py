# sodacan.py
from pyomo.environ import *
from math import pi

M = ConcreteModel()
M.r = Var(bounds=(0,None))
M.h = Var(bounds=(0,None))
M.o = Objective(expr=\
        2*pi*M.r*(M.r + M.h))
M.c = Constraint(expr=\
        pi*M.h*M.r**2 == 355)
