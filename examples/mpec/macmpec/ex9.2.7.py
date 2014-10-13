# ex9.2.7.py QLR-AY-NLP-10-9-4
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.8 in the Test Collection Book
# Test problem 9.2.7 in the web page
# Test problem from Visweswaran etal 1996

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,5)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=(model.x - 5)*(model.x-5) + (2*model.y + 1)*(2*model.y + 1))

# ... Inner Problem Constraints
model.c1 = Constraint(expr= -3*model.x +     model.y + model.s[1] == -3)
model.c2 = Constraint(expr=    model.x - 0.5*model.y + model.s[2] ==  4)
model.c3 = Constraint(expr=    model.x +     model.y + model.s[3] ==  7)
model.c4 = Constraint(expr=            -     model.y + model.s[4] ==  0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= 2*(model.y-1) - 1.5*model.x + model.l[1] - 0.5*model.l[2] + model.l[3] - model.l[4] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

