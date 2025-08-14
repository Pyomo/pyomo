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

# abstract2piece.py
# Similar to abstract2.py, but the objective is now c times x to the fourth power

import pyomo.environ as pyo

model = pyo.AbstractModel()

model.I = pyo.Set()
model.J = pyo.Set()

Topx = 6.1  # range of x variables

model.a = pyo.Param(model.I, model.J)
model.b = pyo.Param(model.I)
model.c = pyo.Param(model.J)

# the next line declares a variable indexed by the set J
model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals, bounds=(0, Topx))
model.y = pyo.Var(model.J, domain=pyo.NonNegativeReals)

# to avoid warnings, we set breakpoints at or beyond the bounds
PieceCnt = 100
bpts = []
for i in range(PieceCnt + 2):
    bpts.append(float((i * Topx) / PieceCnt))


def f4(model, j, xp):
    # we not need j, but it is passed as the index for the constraint
    return xp**4


model.ComputeObj = pyo.Piecewise(
    model.J, model.y, model.x, pw_pts=bpts, pw_constr_type='EQ', f_rule=f4
)


def obj_expression(model):
    return pyo.summation(model.c, model.y)


model.OBJ = pyo.Objective(rule=obj_expression)


def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i, j] * model.x[j] for j in model.J) >= model.b[i]


# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)
