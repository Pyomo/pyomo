# nash1.py QQR2-MN-8-5
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 9

# Number of variables:  6 + 2 slacks
# Number of constraints: 4

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var([1,2], bounds=(0,10))
model.y = Var([1,2])
model.l = Var([1,2], within=NonNegativeReals)   # Multipliers

model.f = Objective(expr=( (model.x[1] - model.y[1])**2 + (model.x[2] - model.y[2])**2 )/2)

model.F1 = Constraint(expr=0 == -34 + 2*model.y[1] + (8/3)*model.y[2]   - ( -model.l[1] ))
model.F2 = Constraint(expr=0 == -24.25 + 1.25*model.y[1] + 2*model.y[2] - ( -model.l[2] ))

model.g1 = Complementarity(expr=complements(0 <= - model.x[2] - model.y[1] + 15, model.l[1] >= 0))
model.g2 = Complementarity(expr=complements(0 <= - model.x[1] - model.y[2] + 15, model.l[2] >= 0))

