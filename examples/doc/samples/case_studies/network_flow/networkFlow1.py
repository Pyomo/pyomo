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

import pyomo.environ as pyo

model = pyo.AbstractModel()

model.places = pyo.Set()
model.routes = pyo.Set(within=model.places * model.places)
model.supply = pyo.Param(model.places)
model.demand = pyo.Param(model.places)
model.cost = pyo.Param(model.routes)
model.minimum = pyo.Param(model.routes)
model.maximum = pyo.Param(model.routes)
model.amount = pyo.Var(model.routes, within=pyo.NonNegativeReals)
model.excess = pyo.Var(model.places, within=pyo.NonNegativeReals)


def costRule(model):
    return sum(model.cost[n] * model.amount[n] for n in model.routes)


model.costTotal = pyo.Objective(rule=costRule)


def loadRule(model, i, j):
    return (model.minimum[i, j], model.amount[i, j], model.maximum[i, j])


model.loadOnRoad = pyo.Constraint(model.routes, rule=loadRule)


def supplyDemandRule(model, nn):
    amountIn = sum(model.amount[i, j] for (i, j) in model.routes if j == nn)
    amountOut = sum(model.amount[i, j] for (i, j) in model.routes if i == nn)

    input = amountIn + model.supply[nn]
    output = amountOut + model.demand[nn] + model.excess[nn]

    return input == output


model.supplyDemand = pyo.Constraint(model.places, rule=supplyDemandRule)
