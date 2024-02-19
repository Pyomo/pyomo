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

#
# Imports
#
from pyomo.core import *

#
# Setup
#

model = AbstractModel()

model.PROD = Set()

model.rate = Param(model.PROD, within=PositiveReals)

model.avail = Param(within=NonNegativeReals)

model.profit = Param(model.PROD)

model.market = Param(model.PROD, within=NonNegativeReals)


def Make_bounds(model, i):
    return (0, model.market[i])


model.Make = Var(model.PROD, bounds=Make_bounds)


def Objective_rule(model):
    return sum_product(model.profit, model.Make)


model.Total_Profit = Objective(rule=Objective_rule, sense=maximize)


def Time_rule(model):
    ans = 0
    for p in model.PROD:
        ans = ans + (1.0 / model.rate[p]) * model.Make[p]
    return ans < model.avail


def XTime_rule(model):
    return sum_product(model.Make, denom=(model.rate,)) < model.avail


# model.Time = Constraint(rule=Time_rule)
