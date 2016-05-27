import pyomo.environ
from pyomo.core import *

# @data:
N = [1,2] # Python list
c = {1:1, 2:2} # Python dictionary
a = {(1,1):3, (2,1):4, (1,2):2, (2,2):5} # Python dictionary
b = {1:1, 2:2} # Python dictionary
# @:data

model = ConcreteModel()
model.x = Var(N, within=NonNegativeReals)
# @obj:
model.obj = Objective(expr=sum(c[i]*model.x[i] for i in N))
# @:obj
model.con1 = Constraint(expr=sum(a[i,1]*model.x[i] for i in N) >= b[1])
model.con2 = Constraint(expr=sum(a[i,2]*model.x[i] for i in N) >= b[2])
