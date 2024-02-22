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

print("indexed1")
# --------------------------------------------------
# @indexed1:
model = pyo.ConcreteModel()
model.A = pyo.Set(initialize=[1, 2, 3])
model.B = pyo.Set(initialize=['Q', 'R'])
model.x = pyo.Var()
model.y = pyo.Var(model.A, model.B)
model.o = pyo.Objective(expr=model.x)
model.c = pyo.Constraint(expr=model.x >= 0)


def d_rule(model, a):
    return a * model.x <= 0


model.d = pyo.Constraint(model.A, rule=d_rule)
# @:indexed1

model.pprint()
