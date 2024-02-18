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
# Example 2.5 - Allen Holder
#

from pyomo.core import *
from pyomo.opt import *

# Instantiate the model
model = AbstractModel()

# Sets
model.NumMonths = Param(within=NonNegativeIntegers)
model.EngineType = Set()
model.Month = RangeSet(1, model.NumMonths)

# Parameters
model.Demand = Param(model.EngineType, model.Month, within=NonNegativeIntegers)
model.InvCost = Param(within=NonNegativeReals)
model.InitInv = Param(model.EngineType, within=NonNegativeIntegers)
model.FinInv = Param(model.EngineType, within=NonNegativeIntegers)
model.Labor = Param(model.EngineType, within=NonNegativeReals)
model.LaborBound = Param(within=NonNegativeReals)
model.ProdCost = Param(model.EngineType, within=NonNegativeReals)
model.ProdBound = Param(within=NonNegativeIntegers)

# Variables
model.Produce = Var(model.EngineType, model.Month, within=NonNegativeIntegers)
model.Inventory = Var(model.EngineType, model.Month, within=NonNegativeIntegers)


# Objective
def CalcCost(M):
    return sum(
        M.Produce[e, t] * M.ProdCost[e] for e in M.EngineType for t in M.Month
    ) + sum(M.Inventory[e, t] * M.InvCost for e in M.EngineType for t in M.Month)


model.TotalCost = Objective(rule=CalcCost, sense=minimize)


# Constraints
def EnsureBalance(M, e, t):
    if t != 1:
        return (
            M.Inventory[e, t]
            == M.Inventory[e, t - 1] + M.Produce[e, t] - M.Demand[e, t]
        )
    else:
        return M.Inventory[e, t] == M.InitInv[e] + M.Produce[e, t] - M.Demand[e, t]


model.InventoryBalance = Constraint(model.EngineType, model.Month, rule=EnsureBalance)


def EnsureLaborLimit(M, t):
    return sum(M.Produce[e, t] * M.Labor[e] for e in M.EngineType) <= M.LaborBound


model.LimitLabor = Constraint(model.Month, rule=EnsureLaborLimit)


def EnsureProdLimit(M, t):
    return sum(M.Produce[e, t] for e in M.EngineType) <= M.ProdBound


model.ProdLimit = Constraint(model.Month, rule=EnsureProdLimit)


def LeaveEnough(M, e, t):
    if t == len(M.Month):
        return M.Inventory[e, t] >= M.FinInv[e]


model.FinalInventory = Constraint(model.EngineType, model.Month, rule=LeaveEnough)
