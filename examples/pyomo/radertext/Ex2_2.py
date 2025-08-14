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
# Example 2.2 - Allen Holder
#

import pyomo.environ as pyo

# Instantiate the model
model = pyo.AbstractModel()

# Parameters for Set Definitions
model.NumTimePeriods = pyo.Param(within=pyo.NonNegativeIntegers)

# Sets
model.StartTime = pyo.RangeSet(1, model.NumTimePeriods)

# Parameters
model.RequiredWorkers = pyo.Param(model.StartTime, within=pyo.NonNegativeIntegers)

# Variables
model.NumWorkers = pyo.Var(model.StartTime, within=pyo.NonNegativeIntegers)


# Objective
def CalcTotalWorkers(M):
    return sum(M.NumWorkers[i] for i in M.StartTime)


model.TotalWorkers = pyo.Objective(rule=CalcTotalWorkers, sense=pyo.minimize)


# Constraints
def EnsureWorkforce(M, i):
    if i != M.NumTimePeriods.value:
        return M.NumWorkers[i] + M.NumWorkers[i + 1] >= M.RequiredWorkers[i + 1]
    else:
        return (
            M.NumWorkers[1] + M.NumWorkers[M.NumTimePeriods.value]
            >= M.RequiredWorkers[1]
        )


model.WorkforceDemand = pyo.Constraint(model.StartTime, rule=EnsureWorkforce)
