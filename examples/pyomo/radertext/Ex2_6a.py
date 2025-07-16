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
# Example 2.6a - Allen Holder
#

import pyomo.environ as pyo

# Instantiate the model
model = pyo.AbstractModel()

# Sets and Set Parameters
model.NumSensors = pyo.Param(within=pyo.NonNegativeIntegers)
model.Sensor = pyo.RangeSet(1, model.NumSensors)

# Parameters
model.xPos = pyo.Param(model.Sensor, within=pyo.NonNegativeIntegers)
model.yPos = pyo.Param(model.Sensor, within=pyo.NonNegativeIntegers)

# Variables
model.xCentralSensor = pyo.Var(within=pyo.NonNegativeIntegers)
model.yCentralSensor = pyo.Var(within=pyo.NonNegativeIntegers)
model.xMax = pyo.Var(model.Sensor, within=pyo.NonNegativeReals)
model.yMax = pyo.Var(model.Sensor, within=pyo.NonNegativeReals)


# Objective
def CalcDist(M):
    return sum(M.xMax[s] + M.yMax[s] for s in M.Sensor)


model.Dist = pyo.Objective(rule=CalcDist, sense=pyo.minimize)

# Constraints


def xEnsureUp(s, M):
    return M.xCentralSensor - M.xPos[s] <= M.xMax[s]


model.xUpBound = pyo.Constraint(model.Sensor, rule=xEnsureUp)


def xEnsureLow(s, M):
    return M.xCentralSensor - M.xPos[s] >= -M.xMax[s]


model.xLowBound = pyo.Constraint(model.Sensor, rule=xEnsureLow)


def yEnsureUp(s, M):
    return M.yCentralSensor - M.yPos[s] <= M.yMax[s]


model.yUpBound = pyo.Constraint(model.Sensor, rule=yEnsureUp)


def yEnsureLow(s, M):
    return M.yCentralSensor - M.yPos[s] >= -M.yMax[s]


model.yLowBound = pyo.Constraint(model.Sensor, rule=yEnsureLow)
