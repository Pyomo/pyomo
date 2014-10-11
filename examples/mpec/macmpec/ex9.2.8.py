# ex9.2.8.py QLR-AY-NLP-6-5-2
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.9 in the Test Collection Book
# Test problem 9.2.8 in the web page
# Test problem from Yezza 96
# Bilinear Inner Objective

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = [1,2]

model.x = Var(bounds=(0,1))
model.y = Var(within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.ob = Objective(expr=-4*model.x*model.y + 3*model.y + 2*model.x + 1)

# ... Inner Problem Constraints
model.c1 = Constraint(expr=  -model.y + model.s[1] == 0)
model.c2 = Constraint(expr=   model.y + model.s[2] == 1)

# ... KKT conditions for the inner problem optimum
model.kt1 = Constraint(expr=  -(1 - 4*model.x) - model.l[1] + model.l[2] == 0)

def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)

