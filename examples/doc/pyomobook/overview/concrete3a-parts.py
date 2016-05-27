import pyomo.environ
from pyomo.core import *

# @Xdata:
N = [1,2]
c = {1:1, 2:2}
a = {(1,1):3, (2,1):4, (1,2):2, (2,2):5}
b = {1:1, 2:2}
# @:Xdata

model = ConcreteModel()
model.x = Var(N, within=NonNegativeReals)
# @obj:
model.obj = Objective(expr=c[1]*model.x[1] + c[2]*model.x[2])
# @:obj
model.con1 = Constraint(expr=sum(a[i,1]*model.x[i] for i in [1,2]) >= b[1])
model.con2 = Constraint(expr=sum(a[i,2]*model.x[i] for i in [1,2]) >= b[2])
