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
# Problem 2.17 - courtesy Allen Holder
#

from pyomo.core import *
from pyomo.opt import *

# Instantiate the model
model = AbstractModel()

# Parameters for Set Definitions
model.NumCrudeTypes = Param(within=PositiveIntegers)
model.NumGasTypes = Param(within=PositiveIntegers)

# Sets
model.CrudeType = RangeSet(1, model.NumCrudeTypes)
model.GasType = RangeSet(1, model.NumGasTypes)

# Parameters
model.Cost = Param(model.CrudeType, within=NonNegativeReals)
model.CrudeOctane = Param(model.CrudeType, within=NonNegativeReals)
model.CrudeMax = Param(model.CrudeType, within=NonNegativeReals)
model.MinGasOctane = Param(model.GasType, within=NonNegativeReals)
model.GasPrice = Param(model.GasType, within=NonNegativeReals)
model.GasDemand = Param(model.GasType, within=NonNegativeReals)
model.MixtureUpBounds = Param(
    model.CrudeType, model.GasType, within=NonNegativeReals, default=10**8
)
model.MixtureLowBounds = Param(
    model.CrudeType, model.GasType, within=NonNegativeReals, default=0
)

# Variabls
model.x = Var(model.CrudeType, model.GasType, within=NonNegativeReals)
model.q = Var(model.CrudeType, within=NonNegativeReals)
model.z = Var(model.GasType, within=NonNegativeReals)


# Objective
def CalcProfit(M):
    return sum(M.GasPrice[j] * M.z[j] for j in M.GasType) - sum(
        M.Cost[i] * M.q[i] for i in M.CrudeType
    )


model.Profit = Objective(rule=CalcProfit, sense=maximize)

# Constraints


def BalanceCrude(M, i):
    return sum(M.x[i, j] for j in M.GasType) == M.q[i]


model.BalanceCrudeProduction = Constraint(model.CrudeType, rule=BalanceCrude)


def BalanceGas(M, j):
    return sum(M.x[i, j] for i in M.CrudeType) == M.z[j]


model.BalanceGasProduction = Constraint(model.GasType, rule=BalanceGas)


def EnsureCrudeLimit(M, i):
    return M.q[i] <= M.CrudeMax[i]


model.LimitCrude = Constraint(model.CrudeType, rule=EnsureCrudeLimit)


def EnsureGasDemand(M, j):
    return M.z[j] >= M.GasDemand[j]


model.DemandGas = Constraint(model.GasType, rule=EnsureGasDemand)


def EnsureOctane(M, j):
    return (
        sum(M.x[i, j] * M.CrudeOctane[i] for i in M.CrudeType)
        >= M.MinGasOctane[j] * M.z[j]
    )


model.OctaneLimit = Constraint(model.GasType, rule=EnsureOctane)


def EnsureLowMixture(M, i, j):
    return sum(M.x[k, j] for k in M.CrudeType) * M.MixtureLowBounds[i, j] <= M.x[i, j]


model.LowCrudeBound = Constraint(model.CrudeType, model.GasType, rule=EnsureLowMixture)


def EnsureUpMixture(M, i, j):
    return sum(M.x[k, j] for k in M.CrudeType) * M.MixtureUpBounds[i, j] >= M.x[i, j]


model.UpCrudeBound = Constraint(model.CrudeType, model.GasType, rule=EnsureUpMixture)
