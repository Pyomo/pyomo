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

from pyomo.core import *

model = AbstractModel()

model.places = Set()
model.routes = Set(within=model.places * model.places)
model.supply = Param(model.places)
model.demand = Param(model.places)
model.cost = Param(model.routes)
model.minimum = Param(model.routes)
model.maximum = Param(model.routes)
model.amount = Var(model.routes, within=NonNegativeReals)
model.excess = Var(model.places, within=NonNegativeReals)


def costRule(model):
    return sum(model.cost[n] * model.amount[n] for n in model.routes)


model.costTotal = Objective(rule=costRule)


def loadRule(model, i, j):
    return (model.minimum[i, j], model.amount[i, j], model.maximum[i, j])


model.loadOnRoad = Constraint(model.routes, rule=loadRule)


def supplyDemandRule(model, nn):
    amountIn = sum(model.amount[i, j] for (i, j) in model.routes if j == nn)
    amountOut = sum(model.amount[i, j] for (i, j) in model.routes if i == nn)

    input = amountIn + model.supply[nn]
    output = amountOut + model.demand[nn] + model.excess[nn]

    return input == output


model.supplyDemand = Constraint(model.places, rule=supplyDemandRule)
