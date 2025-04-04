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

# Imports
import pyomo.environ as pyo

# Create the model object
model = pyo.AbstractModel()

# Sets
model.P = pyo.Set()

# Parameters
model.a = pyo.Param(model.P)
model.b = pyo.Param()
model.c = pyo.Param(model.P)
model.u = pyo.Param(model.P)

# Variables
model.X = pyo.Var(model.P)


# Objective
def Objective_rule(model):
    return sum([model.c[j] * model.X[j] for j in model.P])


model.Total_Profit = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)


# Time Constraint
def Time_rule(model):
    return pyo.sum_product(model.X, denom=model.a) <= model.b


model.Time = pyo.Constraint(rule=Time_rule)


# Limit Constraint
def Limit_rule(model, j):
    return (0, model.X[j], model.u[j])


model.Limit = pyo.Constraint(model.P, rule=Limit_rule)
