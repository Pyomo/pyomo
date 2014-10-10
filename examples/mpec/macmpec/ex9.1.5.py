# ex9.1.5.py LLR-AY-NLP-13-12-5 
# Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.5 in the Test Collection Book
# Test problem 9.1.5 in the web page
# Test problem from Bard 91
# Note there is a typo in the Book corrected here

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

I = range(1,6)

model.x = Var(within=NonNegativeReals)
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=- model.x + 10*model.y1 - model.y2)

# ... Inner Problem Constraints
model.c1 = Constraint(expr=  model.x  + model.y1 + model.s[1] == 1)
model.c2 = Constraint(expr=  model.x  + model.y2 + model.s[2] == 1)
model.c3 = Constraint(expr=  model.y1 + model.y2 + model.s[3] == 1)
model.c4 = Constraint(expr=           - model.y1 + model.s[4] == 0)
model.c5 = Constraint(expr=           - model.y2 + model.s[5] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=model.l[1] + model.l[3] - model.l[4] == 1)
model.kt2 = Constraint(expr=model.l[2] + model.l[3] - model.l[5] == 1)
   
def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)
