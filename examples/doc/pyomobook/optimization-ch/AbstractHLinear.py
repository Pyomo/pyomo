# AbstractHLinear.py - A simple linear version of (H)
from pyomo.environ import *

model = AbstractModel(name="Simple Linear (H)")

model.A = Set()

model.h = Param(model.A)
model.d = Param(model.A)
model.c = Param(model.A)
model.b = Param()
model.u = Param(model.A)

def xbounds_rule(model, i):
    return (0, model.u[i])
model.x = Var(model.A, bounds=xbounds_rule)

def obj_rule(model):
    return sum(model.h[i] * \
               (1 - model.u[i]/model.d[i]**2) * model.x[i] \
               for i in model.A)
model.z = Objective(rule=obj_rule, sense=maximize)

def budget_rule(model):
    return summation(model.c, model.x) <= model.b
model.budgetconstr = Constraint(rule=budget_rule)
