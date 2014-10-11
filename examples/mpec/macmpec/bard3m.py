# bard3m.py    QQR2-MN-
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# From GAMS model in mpeclib of Steven Dirkse, see
# http://www1.gams.com/mpec/mpeclib.htm

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

# ... upper level variables x
model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
 
# .. lower level variables
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.m_cons1 = Var(within=NonNegativeReals)
model.m_cons2 = Var(within=NonNegativeReals)
 
# .. upper level problem objective
model.cost = Objective(expr=-model.x1**2 - 3*model.x2 + model.y2**2 - 4*model.y1)

model.side = Constraint(expr=model.x1**2 + 2*model.x2 <= 4)
 
   # ... optimality conditions of lower level problem
model.cons1 = Complementarity(expr=complements(0 <= model.x1**2 - 2*model.x1 + model.x2**2 - 2*model.y1 + model.y2 + 3, model.m_cons1 >= 0))
model.cons2 = Complementarity(expr=complements(0 <= model.x2 + 3*model.y1 - 4*model.y2 - 4, model.m_cons2 >= 0))
 
model.d_y1 = Complementarity(expr=complements(0 <= (2*model.y1+2*model.m_cons1)-3*model.m_cons2, model.y1 >= 0))
model.d_y2 = Complementarity(expr=complements(0 <= (-5-model.m_cons1)+4*model.m_cons2, model.y2 >= 0))

