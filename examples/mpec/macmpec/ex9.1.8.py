# ex9.1.8.py LLR-AY-NLP-14-12-5
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.8 in the Test Collection Book
# Test problem 9.1.8 in the web page
# Test problem from Bard and Falk 82
# Note the correct results are given here

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,6)

model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=- 2*model.x1 + model.x2 + 0.5*model.y1)

# ... Outer constraint
model.c0 = Constraint(expr=model.x1 +   model.x2  <= 2)

# ... Inner Problem Constraints
model.c1 = Constraint(expr= -2*model.x1 +   model.y1 - model.y2 + model.s[1] == -2.5)
model.c2 = Constraint(expr=    model.x1 - 3*model.x2 + model.y2 + model.s[2] == 2)
model.c3 = Constraint(expr=  - model.y1 + model.s[3] == 0)
model.c4 = Constraint(expr=  - model.y2 + model.s[4] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= model.l[1] - model.l[3] == 4)
model.kt2 = Constraint(expr= model.l[1] + model.l[2] - model.l[4] == -1)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

