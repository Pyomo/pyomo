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
# Example 2.1 - Allen Holder
#

from pyomo.core import *

# Instantiate the model
model = AbstractModel()

# Sets
model.DoorType = Set()
model.MachineType = Set()
model.MarketDoorType1 = Set(within=model.DoorType)
model.MarketDoorType2 = Set(within=model.DoorType)

# Parameters
model.Hours = Param(model.DoorType, model.MachineType, within=NonNegativeReals)
model.Labor = Param(model.DoorType, model.MachineType, within=NonNegativeReals)
model.Profit = Param(model.DoorType, within=NonNegativeReals)
model.MachineLimit = Param(model.MachineType, within=NonNegativeReals)
model.LaborLimit = Param(within=NonNegativeReals)

# Variables
model.NumDoors = Var(model.DoorType, within=NonNegativeIntegers)


# Objective
def CalcProfit(M):
    return sum(M.NumDoors[d] * M.Profit[d] for d in M.DoorType)


model.TotProf = Objective(rule=CalcProfit, sense=maximize)


# Constraints
def EnsureMachineLimit(M, m):
    return sum(M.NumDoors[d] * M.Labor[d, m] for d in M.DoorType) <= M.MachineLimit[m]


model.MachineUpBound = Constraint(model.MachineType, rule=EnsureMachineLimit)


def EnsureLaborLimit(M):
    return (
        sum(M.NumDoors[d] * M.Labor[d, m] for d in M.DoorType for m in M.MachineType)
        <= M.LaborLimit
    )


model.MachineUpBound = Constraint(rule=EnsureLaborLimit)


def EnsureMarketRatio(M):
    return sum(M.NumDoors[d] for d in M.MarketDoorType1) <= sum(
        M.NumDoors[d] for d in M.MarketDoorType2
    )


model.MarketRatio = Constraint(rule=EnsureMarketRatio)
