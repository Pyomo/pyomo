# ex9.2.5.py QLR-AY-NLP-8-7-3
# Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.6 in the Test Collection Book
# Test problem 9.2.5 in the web page
# Test problem from Clark and Westerberg 1990a

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

I = [1,2,3]

model.y = Var()

model.x = Var(bounds=(0,8))
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=(model.x-3)*(model.x-3) + (model.y-2)*(model.y-2))

# ... Inner Problem Constraints
model.c1 = Constraint(expr= -2*model.x +   model.y + model.s[1] == 1)
model.c2 = Constraint(expr=    model.x - 2*model.y + model.s[2] == 2)
model.c3 = Constraint(expr=    model.x + 2*model.y + model.s[3] == 14)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=  2*(model.y-5) + model.l[1] - 2*model.l[2] + 2*model.l[3] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

