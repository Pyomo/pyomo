# ex9.2.6.py QLR-AY-NLP-16-12-6
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.7 in the Test Collection Book
# Test problem 9.2.6 in the web page
# Test problem from Falk and Liu 1995
# Original example from A.D. De Silva's dissertation 78
# Ouadratic Outer and Inner Objective - Nonlinear

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,7)

model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=model.x1*model.x1 - 2*model.x1 + model.x2*model.x2 - 2*model.x2 + model.y1*model.y1 + model.y2*model.y2)

# ... Inner Problem Constraints
model.c1 = Constraint(expr=  0.5 - model.y1 + model.s[1] == 0)
model.c2 = Constraint(expr=  0.5 - model.y2 + model.s[2] == 0)
model.c3 = Constraint(expr=  model.y1 - 1.5 + model.s[3] == 0)
model.c4 = Constraint(expr=  model.y2 - 1.5 + model.s[4] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= 2*(model.y1 - model.x1) - model.l[1] + model.l[3] == 0)
model.kt2 = Constraint(expr= 2*(model.y2 - model.x2) - model.l[2] + model.l[4] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

