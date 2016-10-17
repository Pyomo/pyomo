# bad2.py
from pyomo.environ import *

model = AbstractModel()
model.q = Param(initialize=0, mutable=True)
model.A = Set(initialize=[1,2,3])
model.x = Var(model.A)

def x_rule(model):
    if model.q > 0:
        return sum(model.x[i] for i in model.A) >= 1
    else:
        return sum(model.x[i] for i in model.A) >= 0
model.c = Constraint(rule=x_rule)

instance = model.create_instance()
instance.pprint()
