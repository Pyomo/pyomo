# bilevel1.py  QQR2-MN-16-10
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 7

# Number of variables:   10 
# Number of constraints: 9

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

model.x = Var([1,2], bounds=(0,50))
model.y = Var([1,2])
model.l = Var(range(1,7), within=NonNegativeReals)      # Multipliers

model.f = Objective(expr=2*model.x[1] + 2*model.x[2] - 3*model.y[1] - 3*model.y[2] - 60)

model.c1 = Constraint(expr=model.x[1] + model.x[2] + model.y[1] - 2*model.y[2] - 40 <= 0)
model.F1 = Constraint(expr=0 == 2*model.y[1] - 2*model.x[1] + 40 - (model.l[1] - model.l[2] - 2*model.l[5]))
model.F2 = Constraint(expr=0 == 2*model.y[2] - 2*model.x[2] + 40 - (model.l[3] - model.l[4] - 2*model.l[6]))

model.g1 = Complementarity(expr=complements(0 <= model.y[1] + 10, model.l[1] >= 0))
model.g2 = Complementarity(expr=complements(0 <= -model.y[1] + 20, model.l[2] >= 0))
model.g3 = Complementarity(expr=complements(0 <= model.y[2] + 10, model.l[3] >= 0))
model.g4 = Complementarity(expr=complements(0 <= -model.y[2] + 20, model.l[4] >= 0))
model.g5 = Complementarity(expr=complements(0 <= model.x[1] - 2*model.y[1] - 10, model.l[5] >= 0))
model.g6 = Complementarity(expr=complements(0 <= model.x[2] - 2*model.y[2] - 10, model.l[6] >= 0))

