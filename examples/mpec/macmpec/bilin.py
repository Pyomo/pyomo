# bilin.py QQR2-MN-8-5
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An bilevel linear program due to Hansen, Jaumard and Savard,
# "New branch-and-bound rules for linear bilevel programming,
# SIAM J. Sci. Stat. Comp. 13, 1194-1217, 1992. See also book
# Mathematical Programs with Equilibrium Constraints,
# by Luo, Pang & Ralph, CUP, 1997, p. 357.

# Number of variables:   2 + 3 multipliers
# Number of constraints: 4

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var([1,2], within=NonNegativeReals, initialize=1.0)

# ...multipliers
model.y = Var(range(1,7), within=NonNegativeReals, initialize=1.0)

model.f = Objective(expr=8*model.x[1] + 4*model.x[2] - 4*model.y[1] + 40*model.y[2] + 4*model.y[3])

model.lin = Constraint(expr=model.x[1] + 2*model.x[2] - model.y[3] <= 1.3)

model.KKT1 = Complementarity(expr=complements(   0 <= 2 - model.y[4] - 2*model.y[5] + 4*model.y[6]                 ,  model.y[1] >= 0))
model.KKT2 = Complementarity(expr=complements(   0 <= 1 + model.y[4] + 4*model.y[5] - 2*model.y[6]                 ,  model.y[2] >= 0))
model.KKT3 = Complementarity(expr=complements(   0 <= 2 + model.y[4] - model.y[5] - model.y[6]                     ,  model.y[3] >= 0))

model.slack1 = Complementarity(expr=complements( 0 <= 1 + model.y[1] - model.y[2] - model.y[3]                     ,  model.y[4] >= 0))
model.slack2 = Complementarity(expr=complements( 0 <= 2 - 4*model.x[1] + 2*model.y[1] - 4*model.y[2] + model.y[3]  ,  model.y[5] >= 0))
model.slack3 = Complementarity(expr=complements( 0 <= 2 - 4*model.x[2] - 4*model.y[1] + 2*model.y[2] + model.y[3]  ,  model.y[6] >= 0))

