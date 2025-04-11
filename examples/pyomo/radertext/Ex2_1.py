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
# Example 2.1 - Allen Holder
#

import pyomo.environ as pyo

# Instantiate the model
model = pyo.AbstractModel()

# Sets
model.DoorType = pyo.Set()
model.MachineType = pyo.Set()
model.MarketDoorType1 = pyo.Set(within=model.DoorType)
model.MarketDoorType2 = pyo.Set(within=model.DoorType)

# Parameters
model.Hours = pyo.Param(model.DoorType, model.MachineType, within=pyo.NonNegativeReals)
model.Labor = pyo.Param(model.DoorType, model.MachineType, within=pyo.NonNegativeReals)
model.Profit = pyo.Param(model.DoorType, within=pyo.NonNegativeReals)
model.MachineLimit = pyo.Param(model.MachineType, within=pyo.NonNegativeReals)
model.LaborLimit = pyo.Param(within=pyo.NonNegativeReals)

# Variables
model.NumDoors = pyo.Var(model.DoorType, within=pyo.NonNegativeIntegers)


# Objective
def CalcProfit(M):
    return sum(M.NumDoors[d] * M.Profit[d] for d in M.DoorType)


model.TotProf = pyo.Objective(rule=CalcProfit, sense=pyo.maximize)


# Constraints
def EnsureMachineLimit(M, m):
    return sum(M.NumDoors[d] * M.Labor[d, m] for d in M.DoorType) <= M.MachineLimit[m]


model.MachineUpBound = pyo.Constraint(model.MachineType, rule=EnsureMachineLimit)


def EnsureLaborLimit(M):
    return (
        sum(M.NumDoors[d] * M.Labor[d, m] for d in M.DoorType for m in M.MachineType)
        <= M.LaborLimit
    )


model.MachineUpBound = pyo.Constraint(rule=EnsureLaborLimit)


def EnsureMarketRatio(M):
    return sum(M.NumDoors[d] for d in M.MarketDoorType1) <= sum(
        M.NumDoors[d] for d in M.MarketDoorType2
    )


model.MarketRatio = pyo.Constraint(rule=EnsureMarketRatio)
