# bilevel2.py  QQR2-MN-32-18
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 10

# Number of variables:   20 + 12 slacks
# Number of constraints: 17
# Nonlinear complementarity constraint

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

ubx = {1:10, 2:5, 3:15, 4:20}
x_init = {1:5, 2:5, 3:15, 4:15}

model.I = RangeSet(1,4)

def x_bounds(model, i):
    return (0, ubx[i])
model.x = Var(model.I, bounds=x_bounds)
model.y = Var(model.I)
model.l = Var(range(1,13), within=NonNegativeReals)     # Multipliers

model.f = Objective(expr=- (200 - model.y[1] - model.y[3])*(model.y[1] + model.y[3]) - (160 - model.y[2] - model.y[4])*(model.y[2] + model.y[4]))

model.l1 = Constraint(expr=model.x[1] + model.x[2] + model.x[3] + model.x[4] <= 40)
model.F1 = Constraint(expr=0 == model.y[1] - 4  - ( - 0.4*model.l[1] - 0.6*model.l[2] + model.l[3] - model.l[4]))
model.F2 = Constraint(expr=0 == model.y[2] - 13 - ( - 0.7*model.l[1] - 0.3*model.l[2] + model.l[5] - model.l[6]))
model.F3 = Constraint(expr=0 == model.y[3] - 35 - ( - 0.4*model.l[7] - 0.6*model.l[8] + model.l[9] - model.l[10]))
model.F4 = Constraint(expr=0 == model.y[4] - 2  - ( - 0.7*model.l[7] - 0.3*model.l[8] + model.l[11] - model.l[12]))

model.g1 = Complementarity(expr=complements(  0 <= model.x[1] - 0.4*model.y[1] - 0.7*model.y[2], model.l[1]  >= 0))
model.g2 = Complementarity(expr=complements(  0 <= model.x[2] - 0.6*model.y[1] - 0.3*model.y[2], model.l[2]  >= 0))
model.g3 = Complementarity(expr=complements(  0 <= model.y[1]                                  , model.l[3]  >= 0))
model.g4 = Complementarity(expr=complements(  0 <= - model.y[1] + 20                           , model.l[4]  >= 0))
model.g5 = Complementarity(expr=complements(  0 <= model.y[2]                                  , model.l[5]  >= 0))
model.g6 = Complementarity(expr=complements(  0 <= - model.y[2] + 20                           , model.l[6]  >= 0))
model.g7 = Complementarity(expr=complements(  0 <= model.x[3] - 0.4*model.y[3] - 0.7*model.y[4], model.l[7]  >= 0))
model.g8 = Complementarity(expr=complements(  0 <= model.x[4] - 0.6*model.y[3] - 0.3*model.y[4], model.l[8]  >= 0))
model.g9 = Complementarity(exprcomplements=(  0 <= model.y[3]                                  , model.l[9]  >= 0))
model.g10 = Complementarity(expr=complements( 0 <= - model.y[3] + 40                           , model.l[10] >= 0))
model.g11 = Complementarity(expr=complements( 0 <= model.y[4]                                  , model.l[11] >= 0))
model.g12 = Complementarity(expr=complements( 0 <= -model.y[4] + 40                            , model.l[12] >= 0))

