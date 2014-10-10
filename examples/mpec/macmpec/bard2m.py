# bard2m.py    QQR2-MN-
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# From GAMS model in mpeclib of Steven Dirkse, see
# http://www1.gams.com/mpec/mpeclib.htm

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

model.x11 = Var(bounds=(0,10))
model.x12 = Var(bounds=(0,5))
model.x21 = Var(bounds=(0,15))
model.x22 = Var(bounds=(0,20))

# ... variables for optimality conds of 2nd level problem a
model.y11 = Var(bounds=(0,20))
model.y12 = Var(bounds=(0,20))
model.m_c11 = Var(within=NonPositiveReals)
model.m_c12 = Var(within=NonPositiveReals)

# ... variables for optimality conds of 2nd level problem b
model.y21 = Var(bounds=(0,40))
model.y22 = Var(bounds=(0,40))
model.m_c21 = Var(within=NonPositiveReals)
model.m_c22 = Var(within=NonPositiveReals)

model.cost = Objective(expr=-(200-model.y11-model.y21)*(model.y11+model.y21) - (160 - model.y12 - model.y22)*(model.y12 + model.y22))

model.side = Constraint(expr=model.x11 + model.x12 + model.x21 + model.x22 <= 40)

   # ... optimality conds of second level problem a
model.c11 = Complementarity(expr=complements(0 <= - ( 0.4*model.y11 + 0.7*model.y12 - model.x11 ), model.m_c11 <= 0))
model.c12 = Complementarity(expr=complements(0 <= - ( 0.6*model.y11 + 0.3*model.y12 - model.x12 ), model.m_c12 <= 0))

model.d_y11 = Complementarity(expr=complements(0 == 2*(model.y11-4)-model.m_c11*0.4-model.m_c12*0.6, model.y11))
model.d_y12 = Complementarity(expr=complements(0 == 2*(model.y12-13)-model.m_c11*0.7-model.m_c12*0.3, model.y12))

   # ... optimality conds of second level problem b
model.c21 = Complementarity(expr=complements(0 <= - ( 0.4*model.y21 + 0.7*model.y22 - model.x21 ), model.m_c21 <= 0))
model.c22 = Complementarity(expr=complements(0 <= - ( 0.6*model.y21 + 0.3*model.y22 - model.x22 ), model.m_c22 <= 0))

model.d_y21 = Complementarity(expr=complements(0 == 2*(model.y21-35)-model.m_c21*0.4-model.m_c22*0.6, model.y21))
model.d_y22 = Complementarity(expr=complements(0 == 2*(model.y22-2)-model.m_c21*0.7-model.m_c22*0.3, model.y22))

