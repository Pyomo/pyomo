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
model.e = pyo.Expression()
# @:decl1

model.pprint()
model = None
model = pyo.ConcreteModel()

# @decl2:
model.x = pyo.Var()
model.e1 = pyo.Expression(expr=model.x + 1)


def e2_rule(model):
    return model.x + 2


model.e2 = pyo.Expression(rule=e2_rule)
# @:decl2

model.pprint()
del e2_rule
model = None
model = pyo.ConcreteModel()

# @decl3:
N = [1, 2, 3]
model.x = pyo.Var(N)


def e_rule(model, i):
    if i == 1:
        return pyo.Expression.Skip
    else:
        return model.x[i] ** 2


model.e = pyo.Expression(N, rule=e_rule)
# @:decl3

model.pprint()
del e_rule
model = None
model = pyo.ConcreteModel()

# @decl4:
model.x = pyo.Var()
model.e = pyo.Expression(expr=(model.x - 1.0) ** 2)
model.o = pyo.Objective(expr=0.1 * model.e + model.x)
model.c = pyo.Constraint(expr=model.e <= 1.0)
# @:decl4

model.pprint()

# @decl5:
model.x.set_value(2.0)
print(pyo.value(model.e))  # 1.0
print(pyo.value(model.o))  # 2.1
print(pyo.value(model.c.body))  # 1.0

model.e.set_value((model.x - 2.0) ** 2)
print(pyo.value(model.e))  # 0.0
print(pyo.value(model.o))  # 2.0
print(pyo.value(model.c.body))  # 0.0
# @:decl5

model.pprint()
