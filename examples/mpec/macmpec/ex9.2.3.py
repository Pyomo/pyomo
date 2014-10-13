# ex9.2.3.py LLR-AY-NLP-16-16-6
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.4 in the Test Collection Book
# Test problem 9.2.3 in the web page
# Test problem from Visweswaran etal 1996
# Originally from Shimizu Aiyoshi 81
# Ouadratic Outer and Inner Objective - Nonlinear

#********************************************
# This program locates the LOCAL minimum
#********************************************

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,7)

model.y1 = Var(bounds=(-8, None))
model.y2 = Var(bounds=(-8, None))
model.x1 = Var(bounds=(1, 50))
model.x2 = Var(bounds=(1, 50))
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=2*model.x1 + 2*model.x2 - 3*model.y1 - 3*model.y2 - 60)

# ... Outer Problem Constraints
model.o1 = Constraint(expr=   model.x1 + model.x2 +   model.y1 - 2*model.y2  <= 40)

# ... Inner Problem Constraints
model.c1 = Constraint(expr= - model.x1 + 2*model.y1 + model.s[1] == -10)
model.c2 = Constraint(expr= - model.x2 + 2*model.y2 + model.s[2] == -10)
model.c3 = Constraint(expr=            -   model.y1 + model.s[3] == 10)
model.c4 = Constraint(expr=                model.y1 + model.s[4] == 20)
model.c5 = Constraint(expr=            -   model.y2 + model.s[5] == 10)
model.c6 = Constraint(expr=                model.y2 + model.s[6] == 20)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= 2*(model.y1 - model.x1 + 20) + 2*model.l[1] - model.l[3] + model.l[4] == 0)
model.kt2 = Constraint(expr= 2*(model.y2 - model.x2 + 20) + 2*model.l[2] - model.l[5] + model.l[6] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)
