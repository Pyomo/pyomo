#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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

model.commit = Param(model.PROD, within=NonNegativeReals)

model.market = Param(model.PROD, within=NonNegativeReals)

def Make_bounds(model, i):
    return (model.commit[i],model.market[i])
model.Make = Var(model.PROD, bounds=Make_bounds)

def Objective_rule(model):
    return sum_product(model.profit, model.Make)
model.totalprofit = Objective(rule=Objective_rule, sense=maximize)

def Time_rule(model):
    return sum_product(model.Make, denom=(model.rate)) < model.avail
model.Time = Constraint(rule=Time_rule)
