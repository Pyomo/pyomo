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

"""
This example shows how to represent a piecewise function using
Pyomo's built SOSConstraint component. The function is defined as:

       / 3x-2 , 1 <= x <= 2
f(x) = |
       \ 5x-6 , 2 <= x <= 3
"""

import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.index_set = pyo.Set(initialize=[1, 2])
DOMAIN_PTS = {1: [1, 2, 3], 2: [1, 2, 3]}
F = {1: [1, 4, 9], 2: [1, 4, 9]}
# Note we can also implement this like below
# F = lambda x: x**2
# Update the return value for constraint2_rule if
# F is defined using the function above


# Indexing set required for the SOSConstraint declaration
def SOS_indices_init(model, t):
    return [(t, i) for i in range(len(DOMAIN_PTS[t]))]


model.SOS_indices = pyo.Set(
    model.index_set, dimen=2, ordered=True, initialize=SOS_indices_init
)


def sos_var_indices_init(model):
    return [(t, i) for t in model.index_set for i in range(len(DOMAIN_PTS[t]))]


model.sos_var_indices = pyo.Set(ordered=True, dimen=2, initialize=sos_var_indices_init)

model.x = pyo.Var(model.index_set)  # domain variable
model.Fx = pyo.Var(model.index_set)  # range variable
model.y = pyo.Var(model.sos_var_indices, within=pyo.NonNegativeReals)  # SOS2 variable

model.obj = pyo.Objective(expr=pyo.sum_product(model.Fx), sense=pyo.maximize)


def constraint1_rule(model, t):
    return model.x[t] == sum(
        model.y[t, i] * DOMAIN_PTS[t][i] for i in range(len(DOMAIN_PTS[t]))
    )


def constraint2_rule(model, t):
    # Uncomment below for F defined as dictionary
    return model.Fx[t] == sum(
        model.y[t, i] * F[t][i] for i in range(len(DOMAIN_PTS[t]))
    )
    # Uncomment below for F defined as lambda function
    # return model.Fx[t] == sum(model.y[t,i]*F(DOMAIN_PTS[t][i]) for i in range(len(DOMAIN_PTS[t])) )


def constraint3_rule(model, t):
    return sum(model.y[t, j] for j in range(len(DOMAIN_PTS[t]))) == 1


model.constraint1 = pyo.Constraint(model.index_set, rule=constraint1_rule)
model.constraint2 = pyo.Constraint(model.index_set, rule=constraint2_rule)
model.constraint3 = pyo.Constraint(model.index_set, rule=constraint3_rule)
model.SOS_set_constraint = pyo.SOSConstraint(
    model.index_set, var=model.y, index=model.SOS_indices, sos=2
)

# Fix the answer for testing purposes
model.set_answer_constraint1 = pyo.Constraint(expr=model.x[1] == 2.5)
model.set_answer_constraint2 = pyo.Constraint(expr=model.x[2] == 2.0)
