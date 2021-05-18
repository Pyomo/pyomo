# bad2.py
import pyomo.environ as pyo

model = pyo.AbstractModel()
model.q = pyo.Param(initialize=0, mutable=True)
model.A = pyo.Set(initialize=[1,2,3])
model.x = pyo.Var(model.A)

def x_rule(model):
    if model.q > 0:
        return sum(model.x[i] for i in model.A) >= 1
    else:
        return sum(model.x[i] for i in model.A) >= 0
model.c = pyo.Constraint(rule=x_rule)

instance = model.create_instance()
instance.pprint()
