# nonlin with indexes
import sys
import math
import pyomo.environ
from pyomo.core import*

SizesIn = {"xAxis":5, "yAxis":4}
pIn = 1
qIn = 1

model = ConcreteModel()

# Sets
model.AxisSet = Set(ordered=True, initialize=["xAxis", "yAxis"])

# Parameters
def sizesInit(model, i):
    return SizesIn[i]
model.sizes = Param(model.AxisSet, within=Reals, initialize=sizesInit)
model.pq = Param(model.AxisSet, within=Reals, initialize={"xAxis":pIn, "yAxis":qIn})

# Variables
def xbnd(model, i):
    return (-SizesIn[i], SizesIn[i])
model.x = Var(model.AxisSet, bounds=xbnd, initialize=0)

# Objective
model.obj = Objective(expr = sum( (model.pq[i] - model.x[i])**2 for i in model.AxisSet)**0.5)

# Constraints
model.KeineAhnung = Constraint (expr = sum( (model.x[i] / model.sizes[i])**2 for i in model.AxisSet) - 1 >= 0)
