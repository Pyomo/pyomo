# bad1.py
import pyomo.environ as pyo

model = pyo.AbstractModel()
model.A = pyo.Set(initialize=[1,2,3])
model.x = pyo.Var(model.A)

def x_rule(M):
    return sum(M.x[i] for i in model.A) >= 0
model.c = pyo.Constraint(rule=x_rule)

instance = model.create_instance()
instance.pprint()
