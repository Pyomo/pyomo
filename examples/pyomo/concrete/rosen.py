# rosen.py
from pyomo.environ import *

M = ConcreteModel()
M.x = Var()
M.y = Var()
M.o  = Objective(
          expr=(M.x-1)**2 + \
           100*(M.y-M.x**2)**2)

model = M
