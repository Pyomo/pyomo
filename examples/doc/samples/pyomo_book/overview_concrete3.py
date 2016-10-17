from pyomo.environ import *

N = [1,2]
c = {1:1, 2:2}
a = {(1,1):3, (2,1):4, (1,2):2, (2,2):5}
b = {1:1, 2:2}

model = ConcreteModel()
model.x = Var(N, within=NonNegativeReals)
model.obj = Objective(expr=
                sum(c[i]*model.x[i] for i in N))
model.con1 = Constraint(expr=
                sum(a[i,1]*model.x[i] for i in N) >= b[1])
model.con2 = Constraint(expr=
                sum(a[i,2]*model.x[i] for i in N) >= b[2])
