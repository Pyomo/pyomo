# scholtes3.py	QOR2-MN-2-0
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An QPEC from S. Scholtes

# Number of variables:   2 slack
# Number of constraints: 0

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

# start point close to (0,0)
model.x = Var([1,2], within=NonNegativeReals, initialize=0.0001)

model.objf = Objective(expr=0.5*( (model.x[1] - 1)**2 + (model.x[2] - 1)**2 ))

model.LCP = Complementarity(expr=complements(0 <= model.x[1], model.x[2] >= 0))

