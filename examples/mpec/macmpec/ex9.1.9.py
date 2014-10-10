# ex9.1.9.py LLR-AY-NLP-12-11-5
# Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.10 in the Test Collection Book
# Test problem 9.1.9 in the web page
# Test Problem from visweswaran-etal 1996
# Taken from Bard 1983

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

I = range(1,6)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.c1 = Objective(expr=model.x + model.y)

# ... Inner Problem Constraints
model.c2 = Constraint(expr=       -model.x - 0.5*model.y + model.s[1] == -2)
model.c3 = Constraint(expr=  -0.25*model.x +     model.y + model.s[2] ==  2)
model.c4 = Constraint(expr=        model.x + 0.5*model.y + model.s[3] ==  8)
model.c5 = Constraint(expr=        model.x -   2*model.y + model.s[4] ==  2)
model.c6 = Constraint(expr=                    - model.y + model.s[5] ==  0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= -0.5*model.l[1] + model.l[2] + 0.5*model.l[3] - 2*model.l[4] - model.l[5] == 1)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

