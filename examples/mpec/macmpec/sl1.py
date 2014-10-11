# sl1.py   QQR2-MN-11-6
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# A QPEC obtained by varying the rhs of HS21 (a QP)
# from an idea communicated by S. Scholtes

# Number of variables:   5 + 3 multipliers
# Number of constraints: 5

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import Complementarity

zl = {1:10, 2:0.01, 3:0}
zu = {1:1e10, 2:10, 3:1}


model = ConcreteModel()

model.x = Var([1,2])

def z_bounds(model, i):
    return (zl[i], zu[i])
model.z = Var([1,2,3],bounds=z_bounds)

# ... slack variables & multipliers
model.l = Var([1,2,3], within=NonNegativeReals)

model.f = Objective(expr=(model.x[1] - 2)**2 + model.x[2]**2)   # min. deviation from soln

model.KKT1 = Constraint(expr=0.02*model.x[1] - 10*model.l[1] - model.l[2] == 0);  # ... KKT in x[1]
model.KKT2 = Constraint(expr=2*model.x[2] - model.l[1] - model.l[3] == 0)  # ... KKT in x[2]

model.lin_1 = Complementarity(expr=complements(0 <= 10*model.x[1] + model.x[2] - (10 + model.z[1]), model.l[1] >= 0))
model.lin_2 = Complementarity(expr=complements(0 <= model.x[1] - (2 + model.z[2]), model.l[2] >= 0))
model.lin_3 = Complementarity(expr=complements(0 <= model.x[2] - 50*model.z[3], model.l[3] >= 0))

