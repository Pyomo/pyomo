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

model.warehouses = pyo.Set()
model.stores = pyo.Set()
model.supply = pyo.Param(model.warehouses)
model.demand = pyo.Param(model.stores)
model.costs = pyo.Param(model.warehouses, model.stores)
model.amounts = pyo.Var(model.warehouses, model.stores, within=pyo.NonNegativeReals)


def costRule(model):
    return sum(
        model.costs[n, i] * model.amounts[n, i]
        for n in model.warehouses
        for i in model.stores
    )


model.cost = pyo.Objective(rule=costRule)


def minDemandRule(model, store):
    return sum(model.amounts[i, store] for i in model.warehouses) >= model.demand[store]


model.demandConstraint = pyo.Constraint(model.stores, rule=minDemandRule)


def maxSupplyRule(model, warehouse):
    return (
        sum(model.amounts[warehouse, j] for j in model.stores)
        <= model.supply[warehouse]
    )


model.supplyConstraint = pyo.Constraint(model.warehouses, rule=maxSupplyRule)
