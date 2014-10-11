# ex9.2.2.py QLR-AY-NLP-10-11-4
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.3 in the Test Collection Book
# Test problem 9.2.2 in the web page
# Test problem from Visweswaran etal 1996
# Originally from Shimizu Aiyoshi 81

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,5)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=model.x*model.x + (model.y-10)*(model.y-10))

# ... Outer Problem Constraints
model.o1 = Constraint(expr=   model.x           <= 15)
model.o2 = Constraint(expr= - model.x + model.y <= 0)
model.o3 = Constraint(expr= - model.x           <= 0)

# ... Inner Problem Constraints
model.c1 = Constraint(expr= model.x + model.y + model.s[1] == 20)
model.c2 = Constraint(expr=         - model.y + model.s[2] == 0)
model.c3 = Constraint(expr=           model.y + model.s[3] == 20)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr= 2*(model.x + 2*model.y - 30) + model.l[1] - model.l[2] + model.l[3] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

