import sys
import pyomo.environ
from pyomo.core import *

LenIn = 5
WidthIn = 4
pIn = 1
qIn = 1

model = ConcreteModel()

# Parameters
model.length = Param(within=Reals, initialize=LenIn)
model.width = Param(within=Reals, initialize=WidthIn)
model.p = Param(within=Reals, initialize=pIn)
model.q = Param(within=Reals, initialize=qIn)

# Variables
model.x = Var(bounds=(-LenIn, LenIn), initialize=0)
model.y = Var(bounds=(-WidthIn, WidthIn), initialize=0)

# Objective
model.obj = Objective(expr = (((model.p - model.x)**2) + ((model.q - model.y)**2))**0.5)

# Constraints
model.KeineAhnung = Constraint(expr = ((model.x / model.length)**2) + ((model.y / model.width)**2) - 1 >= 0)
