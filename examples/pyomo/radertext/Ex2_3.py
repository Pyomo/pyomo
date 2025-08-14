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
# Problem 2.17 - courtesy Allen Holder
#

import pyomo.environ as pyo

# Instantiate the model
model = pyo.AbstractModel()

# Parameters for Set Definitions
model.NumCrudeTypes = pyo.Param(within=pyo.PositiveIntegers)
model.NumGasTypes = pyo.Param(within=pyo.PositiveIntegers)

# Sets
model.CrudeType = pyo.RangeSet(1, model.NumCrudeTypes)
model.GasType = pyo.RangeSet(1, model.NumGasTypes)

# Parameters
model.Cost = pyo.Param(model.CrudeType, within=pyo.NonNegativeReals)
model.CrudeOctane = pyo.Param(model.CrudeType, within=pyo.NonNegativeReals)
model.CrudeMax = pyo.Param(model.CrudeType, within=pyo.NonNegativeReals)
model.MinGasOctane = pyo.Param(model.GasType, within=pyo.NonNegativeReals)
model.GasPrice = pyo.Param(model.GasType, within=pyo.NonNegativeReals)
model.GasDemand = pyo.Param(model.GasType, within=pyo.NonNegativeReals)
model.MixtureUpBounds = pyo.Param(
    model.CrudeType, model.GasType, within=pyo.NonNegativeReals, default=10**8
)
model.MixtureLowBounds = pyo.Param(
    model.CrudeType, model.GasType, within=pyo.NonNegativeReals, default=0
)

# Variabls
model.x = pyo.Var(model.CrudeType, model.GasType, within=pyo.NonNegativeReals)
model.q = pyo.Var(model.CrudeType, within=pyo.NonNegativeReals)
model.z = pyo.Var(model.GasType, within=pyo.NonNegativeReals)


# Objective
def CalcProfit(M):
    return sum(M.GasPrice[j] * M.z[j] for j in M.GasType) - sum(
        M.Cost[i] * M.q[i] for i in M.CrudeType
    )


model.Profit = pyo.Objective(rule=CalcProfit, sense=pyo.maximize)

# Constraints


def BalanceCrude(M, i):
    return sum(M.x[i, j] for j in M.GasType) == M.q[i]


model.BalanceCrudeProduction = pyo.Constraint(model.CrudeType, rule=BalanceCrude)


def BalanceGas(M, j):
    return sum(M.x[i, j] for i in M.CrudeType) == M.z[j]


model.BalanceGasProduction = pyo.Constraint(model.GasType, rule=BalanceGas)


def EnsureCrudeLimit(M, i):
    return M.q[i] <= M.CrudeMax[i]


model.LimitCrude = pyo.Constraint(model.CrudeType, rule=EnsureCrudeLimit)


def EnsureGasDemand(M, j):
    return M.z[j] >= M.GasDemand[j]


model.DemandGas = pyo.Constraint(model.GasType, rule=EnsureGasDemand)


def EnsureOctane(M, j):
    return (
        sum(M.x[i, j] * M.CrudeOctane[i] for i in M.CrudeType)
        >= M.MinGasOctane[j] * M.z[j]
    )


model.OctaneLimit = pyo.Constraint(model.GasType, rule=EnsureOctane)


def EnsureLowMixture(M, i, j):
    return sum(M.x[k, j] for k in M.CrudeType) * M.MixtureLowBounds[i, j] <= M.x[i, j]


model.LowCrudeBound = pyo.Constraint(
    model.CrudeType, model.GasType, rule=EnsureLowMixture
)


def EnsureUpMixture(M, i, j):
    return sum(M.x[k, j] for k in M.CrudeType) * M.MixtureUpBounds[i, j] >= M.x[i, j]


model.UpCrudeBound = pyo.Constraint(
    model.CrudeType, model.GasType, rule=EnsureUpMixture
)
