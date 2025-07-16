#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# abstract2.py


import pyomo.environ as pyo

model = pyo.AbstractModel()

model.I = pyo.Set()
model.J = pyo.Set()

model.a = pyo.Param(model.I, model.J)
model.b = pyo.Param(model.I)
model.c = pyo.Param(model.J)

# the next line declares a variable indexed by the set J
model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)


def obj_expression(model):
    return pyo.summation(model.c, model.x)


model.OBJ = pyo.Objective(rule=obj_expression)


def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i, j] * model.x[j] for j in model.J) >= model.b[i]


# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)
