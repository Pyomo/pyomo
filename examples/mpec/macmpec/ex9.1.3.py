# ex9.1.3.py LLR-AY-NLP-23-21-6
# Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.3 in the Test Collection Book
# Test problem 9.1.3 in the web page
# Test problem from Candler-Townsley 82
#
# Removed MI-model of complementarity.

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

I = range(1,7)
J = range(1,4)

model.y = Var(I, within=NonNegativeReals)
model.mu = Var(J)
model.x = Var(J, within=NonNegativeReals)
model.s = Var(I, within=NonNegativeReals)
model.l = Var(I, within=NonNegativeReals)

# ... Outer Objective function
model.c1 = Objective(expr=4*model.y[1] - 40*model.y[2] - 4*model.y[3] - 8*model.x[1] - 4*model.x[2])

# ... Inner Problem Constraints
c2 = Constraint(expr=   -model.y[1] +   model.y[2] +     model.y[3] + model.y[4]                         == 1)
c3 = Constraint(expr=   -model.y[1] + 2*model.y[2] - 0.5*model.y[3]      + model.y[5]      + 2*model.x[1] == 1)
c4 = Constraint(expr=  2*model.y[1] -   model.y[2] - 0.5*model.y[3]           + model.y[6] + 2*model.x[2] == 1)
c5 = Constraint(expr=  - model.y[1] + model.s[1] == 0)
c6 = Constraint(expr=  - model.y[2] + model.s[2] == 0) 
c7 = Constraint(expr=  - model.y[3] + model.s[3] == 0)
c8 = Constraint(expr=  - model.y[4] + model.s[4] == 0)
c9 = Constraint(expr=  - model.y[5] + model.s[5] == 0)
c10 = Constraint(expr= - model.y[6] + model.s[6] == 0) 

# ... KKT conditions for the inner problem optimum
kt1 = Constraint(expr=  1 - model.mu[1] -     model.mu[2] +   2*model.mu[3] - model.l[1] == 0)
kt2 = Constraint(expr=  1 + model.mu[1] +   2*model.mu[2] -     model.mu[3] - model.l[2] == 0)
kt3 = Constraint(expr=  2 + model.mu[1] - 0.5*model.mu[2] - 0.5*model.mu[3] - model.l[3] == 0)
kt4 = Constraint(expr=      model.mu[1]                                     - model.l[4] == 0)
kt5 = Constraint(expr=                        model.mu[2]                   - model.l[5] == 0)
kt6 = Constraint(expr=                                          model.mu[3] - model.l[6] == 0)


# ... Complementarity Constraints
def compl_(model, i):
    return complements(0 <= model.l[i], model.s[i] >= 0)
model.compl = Complementarity(I, rule=compl_)
