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

model.warehouses = Set()
model.stores = Set()
model.supply = Param(model.warehouses)
model.demand = Param(model.stores)
model.costs = Param(model.warehouses, model.stores)
model.amounts = Var(model.warehouses, model.stores, within=NonNegativeReals)


def costRule(model):
    return sum(
        model.costs[n, i] * model.amounts[n, i]
        for n in model.warehouses
        for i in model.stores
    )


model.cost = Objective(rule=costRule)


def minDemandRule(model, store):
    return sum(model.amounts[i, store] for i in model.warehouses) >= model.demand[store]


model.demandConstraint = Constraint(model.stores, rule=minDemandRule)


def maxSupplyRule(model, warehouse):
    return (
        sum(model.amounts[warehouse, j] for j in model.stores)
        <= model.supply[warehouse]
    )


model.supplyConstraint = Constraint(model.warehouses, rule=maxSupplyRule)
