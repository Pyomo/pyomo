from pyomo.environ import *

N = [1,2]
M = [1,2]
c = {1:1, 2:2}
a = {(1,1):3, (2,1):4, (1,2):2, (2,2):5}
b = {1:1, 2:2}

model = ConcreteModel()
model.x = Var(N, within=NonNegativeReals)
model.obj = Objective(expr=sum(c[i]*model.x[i] for i in N))

def con_rule(model, m):
    return sum(a[i,m]*model.x[i] for i in N) >= b[m]
model.con = Constraint(M, rule=con_rule)
