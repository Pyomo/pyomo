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

# a slightly less brain-dead separable parabolic function of three variables.
# optimal objective function value is obviously 5, with the optimal solution
# being x[1]=x[2]=x[3]=3.

import pyomo.environ as pyo

model = pyo.AbstractModel()


def indices_rule(model):
    return range(1, 4)


model.indices = pyo.Set(initialize=indices_rule, within=pyo.PositiveIntegers)

model.x = pyo.Var(model.indices, within=pyo.Reals)


def bound_x_rule(model, i):
    return (-10, model.x[i], 10)


model.bound_x = pyo.Constraint(model.indices, rule=bound_x_rule)


def objective_rule(model):
    return 5 + sum([(model.x[i] - 3) * (model.x[i] - 3) for i in model.indices])


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
