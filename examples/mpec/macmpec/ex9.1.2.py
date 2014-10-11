# ex9.1.2.py LLR-AY-NLP-10-9-4 
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.3 in the Test Collection Book
# Test problem 9.1.2 in the web page
# Test problem from Liu and Hart 1994

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,5)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=Boolean)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function

model.objf = Objective(expr=- model.x -3*model.y)

# ... Inner Problem    Constraints
model.c2 = Constraint(expr=  - model.x +   model.y + model.s[1] == 3)
model.c3 = Constraint(expr=    model.x + 2*model.y + model.s[2] == 12)
model.c4 = Constraint(expr=  4*model.x -   model.y + model.s[3] == 12)
model.c5 = Constraint(expr=            -   model.y + model.s[4] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=model.l[1] + 2*model.l[2] - model.l[3] - model.l[4] == -1)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)
