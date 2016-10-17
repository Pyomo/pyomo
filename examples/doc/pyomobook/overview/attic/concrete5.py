import pyomo.environ
from pyomo.core import *


model = ConcreteModel()

def N_rule(model):
    return [1,2]
model.N = Set(initialize=N_rule)

model.M = Set(initialize=[1,2])
model.c = Param(model.N, initialize={1:1, 2:2})
model.a = Param(model.N, model.M,
            initialize={(1,1):3, (2,1):4, (1,2):2, (2,2):5})
model.b = Param(model.M, initialize={1:1, 2:2})

model.x = Var(model.N, within=NonNegativeReals)

def obj_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.N)
model.obj = Objective(rule=obj_rule)

def con_rule(model, m):
    return sum(model.a[i,m]*model.x[i] for i in model.N) \
                    >= model.b[m]
model.con = Constraint(model.M, rule=con_rule)
