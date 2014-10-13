# ex9.2.4.py QLR-AY-NLP-8-7-2
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.5 in the Test Collection Book
# Test problem 9.2.4 in the web page
# Test problem from Yezza 1996

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = [1,2]

model.l1 = Var()
model.x  = Var(within=NonNegativeReals)
model.y1 = Var(within=NonNegativeReals)
model.y2 = Var(within=NonNegativeReals)
model.s  = Var(I, within=NonNegativeReals)
model.l  = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=0.5*( model.y1- 2)*(model.y1 - 2) + 0.5*(model.y2-2)*(model.y2 - 2))

# ... Inner Problem Constraints
model.c1 = Constraint(expr=   model.y1 + model.y2 == model.x)
model.c2 = Constraint(expr= - model.y1            + model.s[1] == 0) 
model.c3 = Constraint(expr=            - model.y2 + model.s[2] == 0)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= model.y1 + model.l1 - model.l[1] == 0)
model.kt2 = Constraint(expr=        1 + model.l1 - model.l[2] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)
