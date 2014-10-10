# bard1m.py    QQR2-MN-9-5
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# From GAMS model in mpeclib of Steven Dirkse, see
# http://www1.gams.com/mpec/mpeclib.htm

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

model.sy = Var(within=NonNegativeReals)
model.l = Var([1,2,3], within=NonNegativeReals)

model.cost = Objective(expr=(model.x-5)**2 + (2*model.y + 1)**2)

model.cons1 = Complementarity(expr=complements(0 <= 3*model.x - model.y - 3, model.l[1] >= 0))
model.cons2 = Complementarity(expr=complements(0 <= -model.x + 0.5*model.y + 4, model.l[2] >= 0))
model.cons3 = Complementarity(expr=complements(0 <= -model.x - model.y + 7, model.l[3] >= 0))

model.d_y = Constraint(expr=model.sy == (((2*(model.y-1)-1.5*model.x)-model.l[1]*(-1)*1)-model.l[2]*0.5)-model.l[3]*(-1)*1)

