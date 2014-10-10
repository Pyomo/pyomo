# ex9.1.4.py  LLR-AY-10-9-4
# Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.5 in the Test Collection Book
# Test problem 9.1.4 in the web page
# Test problem from Clark and Westerberg 1988

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

I = range(1,5)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.c1 = Objective(expr=model.x - 4*model.y)

# ... Inner Problem Constraints
model.c2 = Constraint(expr=  -2*model.x +   model.y + model.s[1] == 0)
model.c3 = Constraint(expr=   2*model.x + 5*model.y + model.s[2] == 108)
model.c4 = Constraint(expr=   2*model.x - 3*model.y + model.s[3] == -4)
model.c5 = Constraint(expr=             -   model.y + model.s[4] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=model.l[1] + 5*model.l[2] - 3*model.l[3] - model.l[4] == -1)

# ... Complementarity Constraints
def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

