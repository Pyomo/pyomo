# ex9.1.1.py LLR-AY-NLP-13-12-5
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.2 in the Test Collection Book
# Test problem 9.1.1 in the web page
# Test problem from Clark and Westerberg 1990
#
# Is there a mistake in constraint kt2? l[2] appears twice!

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,6)

model.y1 = Var()
model.y2 = Var()
model.x  = Var(within=NonNegativeReals)
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.objf = Objective(expr=- model.x - 3*model.y1 + 2*model.y2)

# ... Inner Problem Constraints
model.c1 = Constraint(expr= -2*model.x +   model.y1 + 4*model.y2 + model.s[1] == 16)
model.c2 = Constraint(expr=  8*model.x + 3*model.y1 - 2*model.y2 + model.s[2] == 48)
model.c3 = Constraint(expr= -2*model.x +   model.y1 - 3*model.y2 + model.s[3] == -12)
model.c4 = Constraint(expr=            -   model.y1              + model.s[4] == 0)
model.c5 = Constraint(expr=                model.y1              + model.s[5] == 4)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=-1 + model.l[1] + 3*model.l[2] + model.l[3] - model.l[4] + model.l[5] == 0)
model.kt2 = Constraint(expr=                  4*model.l[2] - 2*model.l[2] - 3*model.l[3] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

