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
# Example 2.5 - Allen Holder
#

import pyomo.environ as pyo

# Instantiate the model
model = pyo.AbstractModel()

# Sets
model.NumMonths = pyo.Param(within=pyo.NonNegativeIntegers)
model.EngineType = pyo.Set()
model.Month = pyo.RangeSet(1, model.NumMonths)

# Parameters
model.Demand = pyo.Param(model.EngineType, model.Month, within=pyo.NonNegativeIntegers)
model.InvCost = pyo.Param(within=pyo.NonNegativeReals)
model.InitInv = pyo.Param(model.EngineType, within=pyo.NonNegativeIntegers)
model.FinInv = pyo.Param(model.EngineType, within=pyo.NonNegativeIntegers)
model.Labor = pyo.Param(model.EngineType, within=pyo.NonNegativeReals)
model.LaborBound = pyo.Param(within=pyo.NonNegativeReals)
model.ProdCost = pyo.Param(model.EngineType, within=pyo.NonNegativeReals)
model.ProdBound = pyo.Param(within=pyo.NonNegativeIntegers)

# Variables
model.Produce = pyo.Var(model.EngineType, model.Month, within=pyo.NonNegativeIntegers)
model.Inventory = pyo.Var(model.EngineType, model.Month, within=pyo.NonNegativeIntegers)


# Objective
def CalcCost(M):
    return sum(
        M.Produce[e, t] * M.ProdCost[e] for e in M.EngineType for t in M.Month
    ) + sum(M.Inventory[e, t] * M.InvCost for e in M.EngineType for t in M.Month)


model.TotalCost = pyo.Objective(rule=CalcCost, sense=pyo.minimize)


# Constraints
def EnsureBalance(M, e, t):
    if t != 1:
        return (
            M.Inventory[e, t]
            == M.Inventory[e, t - 1] + M.Produce[e, t] - M.Demand[e, t]
        )
    else:
        return M.Inventory[e, t] == M.InitInv[e] + M.Produce[e, t] - M.Demand[e, t]


model.InventoryBalance = pyo.Constraint(
    model.EngineType, model.Month, rule=EnsureBalance
)


def EnsureLaborLimit(M, t):
    return sum(M.Produce[e, t] * M.Labor[e] for e in M.EngineType) <= M.LaborBound


model.LimitLabor = pyo.Constraint(model.Month, rule=EnsureLaborLimit)


def EnsureProdLimit(M, t):
    return sum(M.Produce[e, t] for e in M.EngineType) <= M.ProdBound


model.ProdLimit = pyo.Constraint(model.Month, rule=EnsureProdLimit)


def LeaveEnough(M, e, t):
    if t == len(M.Month):
        return M.Inventory[e, t] >= M.FinInv[e]


model.FinalInventory = pyo.Constraint(model.EngineType, model.Month, rule=LeaveEnough)
