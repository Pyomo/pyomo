import pyomo.environ
from pyomo.core import *

N = [1,2]
c = {1:1, 2:2}
a = {(1,1):3, (2,1):4, (1,2):2, (2,2):5}
b = {1:1, 2:2}

model = ConcreteModel()
model.x = Var(N, within=NonNegativeReals)
model.obj = Objective(expr=sum(c[i]*model.x[i] for i in N))

# @all:
@simple_constraintlist_rule
def con_rule(model, m):
    if m == 3:
        return None
    return sum(a[i,m]*model.x[i] for i in N) >= b[m]
model.con = ConstraintList(rule=con_rule)
# @:all
