# bad1.py
from pyomo.environ import *

model = AbstractModel()
model.A = Set(initialize=[1,2,3])
model.x = Var(model.A)

def x_rule(M):
    return sum(M.x[i] for i in model.A) >= 0
model.c = Constraint(rule=x_rule)

instance = model.create_instance()
instance.pprint()
