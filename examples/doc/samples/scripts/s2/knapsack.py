from pyomo.core import *


model = AbstractModel()

model.ITEMS = Set()

model.v = Param(model.ITEMS, within=PositiveReals)

model.w = Param(model.ITEMS, within=PositiveReals)

model.limit = Param(within=PositiveReals)

model.x = Var(model.ITEMS, within=PercentFraction)


def value_rule(model):
    return sum(model.v[i] * model.x[i] for i in model.ITEMS)


model.value = Objective(sense=maximize, rule=value_rule)


def weight_rule(model):
    return sum(model.w[i] * model.x[i] for i in model.ITEMS) <= model.limit


model.weight = Constraint(rule=weight_rule)


# This constraint is not active, to illustrate how zero dual values are
# handled by the pyomo command.
def W_rule(model):
    return sum(model.w[i] * model.x[i] for i in model.ITEMS) <= 2 * model.limit


model.W = Constraint(rule=W_rule)
