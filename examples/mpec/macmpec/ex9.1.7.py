# ex9.1.7.py LLR-AY-NLP-17-15-6
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.8 in the Test Collection Book
# Test problem 9.1.7 in the web page
# Test Problem from Bard and Falk 1982
# Originally from Candler-Townsley 78

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,7)

model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.y3 = Var(within=NonNegativeReals)
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.c1 = Objective(expr=-8*model.x1 - 4*model.x2 + 4*model.y1 - 40*model.y2 + 4*model.y3)

# ... Inner Problem Constraints
model.c2 = Constraint(expr= -model.y1 + model.y2 + model.y3 + model.s[1] == 1)
model.c3 = Constraint(expr= 2*model.x1 - model.y1 + 2*model.y2 - 0.5*model.y3 + model.s[2] == 1)
model.c4 = Constraint(expr= 2*model.x2 + 2*model.y1 - model.y2 - 0.5*model.y3 + model.s[3] == 1)
model.c5 = Constraint(expr= - model.y1 + model.s[4] == 0)
model.c6 = Constraint(expr= - model.y2 + model.s[5] == 0) 
model.c7 = Constraint(expr= - model.y3 + model.s[6] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=  -model.l[1] - model.l[2] + 2*model.l[3] - model.l[4] == -1)
model.kt2 = Constraint(expr=   model.l[1] + 2*model.l[2] - model.l[3] - model.l[5] == -1)
model.kt3 = Constraint(expr=   model.l[1] - 0.5*model.l[2] - 0.5*model.l[3] - model.l[6] == -2)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

