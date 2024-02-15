#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.ConcreteModel()

# @decl1:
model.x = pyo.Var([1, 2], initialize=1.0)
model.diff = pyo.Constraint(expr=model.x[2] - model.x[1] <= 7.5)
# @:decl1

model.pprint()
model = None
model = pyo.ConcreteModel()

# @decl2:
model.x = pyo.Var([1, 2], initialize=1.0)


def diff_rule(model):
    return model.x[2] - model.x[1] <= 7.5


model.diff = pyo.Constraint(rule=diff_rule)
# @:decl2

model.pprint()
model = None
model = pyo.ConcreteModel()

# @decl3:
N = [1, 2, 3]

a = {1: 1, 2: 3.1, 3: 4.5}
b = {1: 1, 2: 2.9, 3: 3.1}

model.y = pyo.Var(N, within=pyo.NonNegativeReals, initialize=0.0)


def CoverConstr_rule(model, i):
    return a[i] * model.y[i] >= b[i]


model.CoverConstr = pyo.Constraint(N, rule=CoverConstr_rule)
# @:decl3

model.pprint()
model = None
model = pyo.ConcreteModel()

# @decl6:
TimePeriods = [1, 2, 3, 4, 5]
LastTimePeriod = 5

model.StartTime = pyo.Var(TimePeriods, initialize=1.0)


def Pred_rule(model, t):
    if t == LastTimePeriod:
        return pyo.Constraint.Skip
    else:
        return model.StartTime[t] <= model.StartTime[t + 1]


model.Pred = pyo.Constraint(TimePeriods, rule=Pred_rule)
# @:decl6

model.pprint()
model = None

# @slack:
model = pyo.ConcreteModel()
model.x = pyo.Var(initialize=1.0)
model.y = pyo.Var(initialize=1.0)

model.c1 = pyo.Constraint(expr=model.y - model.x <= 7.5)
model.c2 = pyo.Constraint(expr=-2.5 <= model.y - model.x)
model.c3 = pyo.Constraint(expr=pyo.inequality(-3.0, model.y - model.x, 7.0))

print(pyo.value(model.c1.body))  # 0.0

print(model.c1.lslack())  # inf
print(model.c1.uslack())  # 7.5
print(model.c2.lslack())  # 2.5
print(model.c2.uslack())  # inf
print(model.c3.lslack())  # 3.0
print(model.c3.uslack())  # 7.0
# @:slack

model.display()
