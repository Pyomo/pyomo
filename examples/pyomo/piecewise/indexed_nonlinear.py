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

# Nonlinear version of example4.
# Must have a nonlinear solver
# to run this example.
import pyomo.environ as pyo
from indexed import model, f


# Reuse the rule from example4 to define the
# nonlinear constraint
def nonlinear_con_rule(model, i, j):
    return model.Z[i, j] == f(model, i, j, model.X[i, j])


model.nonlinear_constraint = pyo.Constraint(model.INDEX1, rule=nonlinear_con_rule)

# deactivate all constraints on the Piecewise component
model.linearized_constraint.deactivate()

# initialize the nonlinear variables to 'good' starting points
for idx in model.X.index_set():
    model.X[idx] = 1.7
    model.Z[idx] = 1.25
