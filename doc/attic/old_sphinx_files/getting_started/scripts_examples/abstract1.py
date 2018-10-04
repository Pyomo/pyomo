from __future__ import division
from pyomo.environ import *

model = AbstractModel()

model.m = Param(within=NonNegativeIntegers)
model.n = Param(within=NonNegativeIntegers)

model.I = RangeSet(1, model.m)
model.J = RangeSet(1, model.n)

model.a = Param(model.I, model.J)
model.b = Param(model.I)
model.c = Param(model.J)

# the next line declares a variable indexed by the set J
model.x = Var(model.J, domain=NonNegativeReals)

def obj_expression(model):
    return summation(model.c, model.x)

model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]

# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = Constraint(model.I, rule=ax_constraint_rule)
