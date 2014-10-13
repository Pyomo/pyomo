# bilevel1m.py QQR2-MN-8-7
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 7

# Number of variables:   8
# Number of constraints: 7

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var([1,2], bounds=(0,50))
model.y = Var([1,2])
model.l = Var(range(1,5))

model.f = Objective(expr=2*model.x[1] + 2*model.x[2] - 3*model.y[1] - 3*model.y[2] - 60)

model.c1 = Constraint(expr=model.x[1] + model.x[2] + model.y[1] - 2*model.y[2] - 40 <= 0)
model.F1 = Constraint(expr=0 == 2*model.y[1] - 2*model.x[1] + 40 - (model.l[1] - 2*model.l[3]))
model.F2 = Constraint(expr=0 == 2*model.y[2] - 2*model.x[2] + 40 - (model.l[2] - 2*model.l[4]))

model.m1 = Complementarity(expr=complements(-10 <= model.y[1] <= 20, model.l[1]))
model.m2 = Complementarity(expr=complements(-10 <= model.y[2] <= 20, model.l[2]))
model.g5 = Complementarity(expr=complements(0 <= model.x[1] - 2*model.y[1] - 10, model.l[3] >= 0))
model.g6 = Complementarity(expr=complements(0 <= model.x[2] - 2*model.y[2] - 10, model.l[4] >= 0))

