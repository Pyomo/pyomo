# bard2.py QQR2-MN-
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An MPEC from J.F. Bard, Convex two-level optimization,
# Mathematical Programming 40(1), 15-27, 1988.
# From Aiyoshi & Shimizu, IEEE Trans Syst Man & Cyb SMC-11 (1981),
# 444-449.

# Corrected index error in constraint lin_12 (S Leyffer)

# Number of variables:   8 + 4 slack + 4 multipliers
# Number of constraints: 9
# Nonlinear complementarity constraints

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

N = [1,2]
u_x = {(1,1): 10, (1,2): 5, (2,1): 15, (2,2):20}
u_y = {(1,1): 20, (1,2): 20, (2,1): 40, (2,2):40}

def x_bounds(model, i, j):
    return (0, u_x[i,j])
model.x = Var(N,N, bounds=x_bounds)

def y_bounds(model, i, j):
    return (0, u_y[i,j])
model.y = Var(N,N, bounds=y_bounds)

# ... multipliers
model.l = Var(N,N)

model.f = Objective(expr=(200 - model.y[1,1] - model.y[2,1])*(model.y[1,1] + model.y[2,1]) + (160 - model.y[1,2] - model.y[2,2])*(model.y[1,2] + model.y[2,2]))

model.lincs = Constraint(expr=model.x[1,1] + model.x[1,2] + model.x[2,1] + model.x[2,2] <= 40)

model.KKT1_1 = Constraint(expr=2*(model.y[1,1] - 4)  + model.l[1,1]*0.4 + model.l[1,2]*0.6 == 0)
model.KKT1_2 = Constraint(expr=2*(model.y[1,2] - 13) + model.l[1,1]*0.7 + model.l[1,2]*0.3 == 0)

model.lin_11 = Complementarity(expr=complements(0 <= model.x[1,1] - 0.4*model.y[1,1] - 0.7*model.y[1,2], model.l[1,1] >= 0))
model.lin_12 = Complementarity(expr=complements(0 <= model.x[1,2] - 0.6*model.y[1,1] - 0.3*model.y[1,2], model.l[1,2] >= 0))

model.KKT2_1 = Constraint(expr=2*(model.y[2,1] - 35)  + model.l[2,1]*0.4 + model.l[2,2]*0.6 == 0)
model.KKT2_2 = Constraint(expr=2*(model.y[2,2] - 2) + model.l[2,1]*0.7 + model.l[2,2]*0.3 == 0)

model.lin_21 = Complementarity(expr=complements(0 <= model.x[2,1] - 0.4*model.y[2,1] - 0.7*model.y[2,2], model.l[2,1] >= 0))
model.lin_22 = Complementarity(expr=complements(0 <= model.x[2,2] - 0.6*model.y[2,1] - 0.3*model.y[2,2], model.l[2,2] >= 0))

