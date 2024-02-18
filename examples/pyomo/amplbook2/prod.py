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

# Imports
from pyomo.core import *

# Create the model object
model = AbstractModel()

# Sets
model.P = Set()

# Parameters
model.a = Param(model.P)
model.b = Param()
model.c = Param(model.P)
model.u = Param(model.P)

# Variables
model.X = Var(model.P)


# Objective
def Objective_rule(model):
    return sum([model.c[j] * model.X[j] for j in model.P])


model.Total_Profit = Objective(rule=Objective_rule, sense=maximize)


# Time Constraint
def Time_rule(model):
    return sum_product(model.X, denom=model.a) <= model.b


model.Time = Constraint(rule=Time_rule)


# Limit Constraint
def Limit_rule(model, j):
    return (0, model.X[j], model.u[j])


model.Limit = Constraint(model.P, rule=Limit_rule)
