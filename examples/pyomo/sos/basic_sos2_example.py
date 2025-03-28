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

import pyomo.environ as pyo

# Problem borrowed from the lpsolve project documentation
# http://lpsolve.sourceforge.net/5.0/SOS.htm

# Don't forget that you can see that actual input to the solver if you use
# the -k command line flag:

# $ pyomo basic_sos2_example.py --solver=cplex -k

# A standalone SOS2 example, so the data comes from these functions instead of
# a .dat file.  Thus, define all the "cruft" up here, so it's clearer how the
# model is tied together down below.


def c_param_init(model, v):
    return (-1, -1, -3, -2, -2)[v - 1]  # -1 because Python is 0-based


def b_param_init(model, c):
    return (30, 30)[c - 1]  # -1 because Python is 0-based


def A_param_init(model, c, v):
    data = ((-1, -1, 1, 1, 0), (1, 0, 1, -3, 0))

    return data[c - 1][v - 1]  # -1 because Python is 0-based


def obj_rule(model):
    objective_expression = sum(model.c[i] * model.x[i] for i in model.variable_set)

    return objective_expression


def constraint_rule(model, c):
    constraint_equation = model.b[c] >= sum(
        model.A[c, i] * model.x[i] for i in model.variable_set
    )

    return constraint_equation


x1_constraint_rule = lambda M: M.x[1] <= 40
x2_constraint_rule = lambda M: M.x[2] <= 1
x5_constraint_rule = lambda M: M.x[5] <= 1


###############################################################################
# Cruft above complete, now we actually create and tie the model together

model = pyo.AbstractModel()
M = model

M.variable_set = pyo.RangeSet(1, 5)
M.constraint_set = pyo.RangeSet(1, 2)

M.c = pyo.Param(M.variable_set, initialize=c_param_init)

M.A = pyo.Param(M.constraint_set, M.variable_set, initialize=A_param_init)
M.b = pyo.Param(M.constraint_set, initialize=b_param_init)

M.x = pyo.Var(M.variable_set, within=pyo.PositiveReals)

M.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)  # min "c transpose" X

# At first, this is little more than a standard form Ax=b ...
M.constraints = pyo.Constraint(M.constraint_set, rule=constraint_rule)

# ... with a couple of extra constraints ...
M.x1_constraint = pyo.Constraint(rule=x1_constraint_rule)
M.x2_constraint = pyo.Constraint(rule=x2_constraint_rule)
M.x5_constraint = pyo.Constraint(rule=x5_constraint_rule)

# ... and finally, add the constraint for which this example was created
M.x_sos_vars = pyo.SOSConstraint(var=M.x, sos=2)
