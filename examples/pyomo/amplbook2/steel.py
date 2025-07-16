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

#
# Imports
#
import pyomo.environ as pyo

#
# Setup
#

model = pyo.AbstractModel()

model.PROD = pyo.Set()

model.rate = pyo.Param(model.PROD, within=pyo.PositiveReals)

model.avail = pyo.Param(within=pyo.NonNegativeReals)

model.profit = pyo.Param(model.PROD)

model.market = pyo.Param(model.PROD, within=pyo.NonNegativeReals)


def Make_bounds(model, i):
    return (0, model.market[i])


model.Make = pyo.Var(model.PROD, bounds=Make_bounds)


def Objective_rule(model):
    return pyo.sum_product(model.profit, model.Make)


model.Total_Profit = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)


def Time_rule(model):
    ans = 0
    for p in model.PROD:
        ans = ans + (1.0 / model.rate[p]) * model.Make[p]
    return ans < model.avail


def XTime_rule(model):
    return pyo.sum_product(model.Make, denom=(model.rate,)) < model.avail


# model.Time = pyo.Constraint(rule=Time_rule)
