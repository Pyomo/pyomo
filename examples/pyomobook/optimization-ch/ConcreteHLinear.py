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

# ConcreteHLinear.py - Linear (H)
import pyomo.environ as pyo

model = pyo.ConcreteModel(name="Linear (H)")

A = ['I_C_Scoops', 'Peanuts']
h = {'I_C_Scoops': 1, 'Peanuts': 0.1}
d = {'I_C_Scoops': 5, 'Peanuts': 27}
c = {'I_C_Scoops': 3.14, 'Peanuts': 0.2718}
b = 12
u = {'I_C_Scoops': 100, 'Peanuts': 40.6}


def x_bounds(m, i):
    return (0, u[i])


model.x = pyo.Var(A, bounds=x_bounds)


# @obj:
def obj_rule(model):
    return sum(h[i] * (1 - u[i] / d[i] ** 2) * model.x[i] for i in A)


# @:obj
model.z = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

model.budgetconstr = pyo.Constraint(expr=sum(c[i] * model.x[i] for i in A) <= b)

model.pprint()
