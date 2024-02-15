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

print('declscalar')
# @declscalar:
model.a = pyo.Objective()
# @:declscalar
model.display()

model = None
model = pyo.ConcreteModel()

print('declexprrule')
# @declexprrule:
model.x = pyo.Var([1, 2], initialize=1.0)

model.b = pyo.Objective(expr=model.x[1] + 2 * model.x[2])


def m_rule(model):
    expr = model.x[1]
    expr += 2 * model.x[2]
    return expr


model.c = pyo.Objective(rule=m_rule)
# @:declexprrule
model.display()

model = None
model = pyo.ConcreteModel()

print('declmulti')
# @declmulti:
A = ['Q', 'R', 'S']
model.x = pyo.Var(A, initialize=1.0)


def d_rule(model, i):
    return model.x[i] ** 2


model.d = pyo.Objective(A, rule=d_rule)
# @:declmulti

print('declskip')


# @declskip:
def e_rule(model, i):
    if i == 'R':
        return pyo.Objective.Skip
    return model.x[i] ** 2


model.e = pyo.Objective(A, rule=e_rule)
# @:declskip
model.display()

model = None
model = pyo.ConcreteModel()

print('value')
# @value:
A = ['Q', 'R']
model.x = pyo.Var(A, initialize={'Q': 1.5, 'R': 2.5})
model.o = pyo.Objective(expr=model.x['Q'] + 2 * model.x['R'])
print(model.o.expr)  # x[Q] + 2*x[R]
print(model.o.sense)  # minimize
print(pyo.value(model.o))  # 6.5
# @:value

model.display()
