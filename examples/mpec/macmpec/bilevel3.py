# bilevel3.py  QQR2-MN-16-12
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 11

# Number of variables:   12 
# Number of constraints: 11
# Nonlinear complementarity constraint

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

model.x = Var([1,2], within=NonNegativeReals, initialize={1:0, 2:2})
model.y = Var(range(1,7))
model.l = Var(range(1,5))                   # Multipliers

model.f = Objective(expr=- model.x[1]**2 - 3*model.x[2] - 4*model.y[1] + model.y[2]**2)

model.c1 = Constraint(expr=model.x[1]**2 + 2*model.x[2] <= 4)
model.F1 = Constraint(expr=0 == 2*model.y[1] + 2*model.y[3] - 3*model.y[4] - model.y[5])
model.F2 = Constraint(expr=0 == - 5 - model.y[3] + 4*model.y[4] - model.y[6])
model.F3 = Constraint(expr=0 == model.x[1]**2 - 2*model.x[1] + model.x[2]**2 - 2*model.y[1] + model.y[2] + 3 - ( model.l[1] ))
model.F4 = Constraint(expr=0 == model.x[2] + 3*model.y[1] - 4*model.y[2] - 4      - ( model.l[2] ))
model.F5 = Constraint(expr=0 == model.y[1]                            - ( model.l[3] ))
model.F6 = Constraint(expr=0 == model.y[2]                            - ( model.l[4] ))

model.g1 = Complementarity(expr=complements(0 <= model.l[1], model.y[3] >= 0))
model.g2 = Complementarity(expr=complements(0 <= model.l[2], model.y[4] >= 0))
model.g3 = Complementarity(expr=complements(0 <= model.l[3], model.y[5] >= 0))
model.g4 = Complementarity(expr=complements(0 <= model.l[4], model.y[6] >= 0))

