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
# Example 2.6b - Allen Holder
#

from pyomo.core import *
from pyomo.opt import *

# Instantiate the model
model = AbstractModel()

# Sets and Set Parameters
model.NumSensors = Param(within=NonNegativeIntegers)
model.Sensor = RangeSet(1, model.NumSensors)

# Parameters
model.xPos = Param(model.Sensor, within=NonNegativeIntegers)
model.yPos = Param(model.Sensor, within=NonNegativeIntegers)

# Variables
model.xCentralSensor = Var(within=NonNegativeIntegers)
model.yCentralSensor = Var(within=NonNegativeIntegers)
model.xMax = Var(model.Sensor, within=NonNegativeReals)
model.yMax = Var(model.Sensor, within=NonNegativeReals)
model.Max = Var(within=NonNegativeReals)


# Objective
def CalcDist(M):
    return M.Max


model.Dist = Objective(rule=CalcDist, sense=minimize)

# Constraints


def xEnsureUp(M, s):
    return M.xCentralSensor - M.xPos[s] <= M.xMax[s]


model.xUpBound = Constraint(model.Sensor, rule=xEnsureUp)


def xEnsureLow(M, s):
    return M.xCentralSensor - M.xPos[s] >= -M.xMax[s]


model.xLowBound = Constraint(model.Sensor, rule=xEnsureLow)


def yEnsureUp(M, s):
    return M.yCentralSensor - M.yPos[s] <= M.yMax[s]


model.yUpBound = Constraint(model.Sensor, rule=yEnsureUp)


def yEnsureLow(M, s):
    return M.yCentralSensor - M.yPos[s] >= -M.yMax[s]


model.yLowBound = Constraint(model.Sensor, rule=yEnsureLow)


def EnsureSensorBound(M, s):
    return M.xMax[s] + M.yMax[s] <= M.Max


model.MaxDist = Constraint(model.Sensor, rule=EnsureSensorBound)
