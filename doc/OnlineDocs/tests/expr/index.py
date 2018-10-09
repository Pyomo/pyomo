from pyomo.environ import *

#---------------------------------------------
# @simple
M = ConcreteModel()
M.v = Var()

e = M.v*2
# @simple
print(e)

