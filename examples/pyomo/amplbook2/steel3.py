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

model.commit = pyo.Param(model.PROD, within=pyo.NonNegativeReals)

model.market = pyo.Param(model.PROD, within=pyo.NonNegativeReals)


def Make_bounds(model, i):
    return (model.commit[i], model.market[i])


model.Make = pyo.Var(model.PROD, bounds=Make_bounds)


def Objective_rule(model):
    return pyo.sum_product(model.profit, model.Make)


model.totalprofit = pyo.Objective(rule=Objective_rule, sense=pyo.maximize)


def Time_rule(model):
    return pyo.sum_product(model.Make, denom=(model.rate)) < model.avail


model.Time = pyo.Constraint(rule=Time_rule)
