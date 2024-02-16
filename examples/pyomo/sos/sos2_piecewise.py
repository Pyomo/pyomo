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

"""
This example shows how to represent a piecewise function using 
Pyomo's built SOSConstraint component. The function is defined as:

       / 3x-2 , 1 <= x <= 2
f(x) = | 
       \ 5x-6 , 2 <= x <= 3
"""

from pyomo.core import *

model = ConcreteModel()

model.index_set = Set(initialize=[1, 2])
DOMAIN_PTS = {1: [1, 2, 3], 2: [1, 2, 3]}
F = {1: [1, 4, 9], 2: [1, 4, 9]}
# Note we can also implement this like below
# F = lambda x: x**2
# Update the return value for constraint2_rule if
# F is defined using the function above


# Indexing set required for the SOSConstraint declaration
def SOS_indices_init(model, t):
    return [(t, i) for i in range(len(DOMAIN_PTS[t]))]


model.SOS_indices = Set(
    model.index_set, dimen=2, ordered=True, initialize=SOS_indices_init
)


def sos_var_indices_init(model):
    return [(t, i) for t in model.index_set for i in range(len(DOMAIN_PTS[t]))]


model.sos_var_indices = Set(ordered=True, dimen=2, initialize=sos_var_indices_init)

model.x = Var(model.index_set)  # domain variable
model.Fx = Var(model.index_set)  # range variable
model.y = Var(model.sos_var_indices, within=NonNegativeReals)  # SOS2 variable

model.obj = Objective(expr=sum_product(model.Fx), sense=maximize)


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


model.constraint1 = Constraint(model.index_set, rule=constraint1_rule)
model.constraint2 = Constraint(model.index_set, rule=constraint2_rule)
model.constraint3 = Constraint(model.index_set, rule=constraint3_rule)
model.SOS_set_constraint = SOSConstraint(
    model.index_set, var=model.y, index=model.SOS_indices, sos=2
)

# Fix the answer for testing purposes
model.set_answer_constraint1 = Constraint(expr=model.x[1] == 2.5)
model.set_answer_constraint2 = Constraint(expr=model.x[2] == 2.0)
