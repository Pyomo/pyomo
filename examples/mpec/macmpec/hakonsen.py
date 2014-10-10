# hakonsen.py  OOR2-MY-9-8-4
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Feb. 2001.

# MPEC of taxation model taken from (6)-(10) of
# Light, M., "Optimal taxation: An application of mathematical 
# programming with equilibrium constraints in economics", 
# Department of Economics, University of Colorado, Boulder, 1999.
# attributed to Hakonsen, L. "Essays on Taxation, Efficiency and the 
# Environment", PhD thesis, Norwegian School of Economics & Business 
# Administration, April 1998.

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

I = [1,2]

# ... constants
L  = 100    # ... units of time
G  = 25     # ... collected revenue (tax)
pL = 1      # ... wage chosen as numerairs

# ... model variables
model.x = Var(I, bounds=(0,1))                          # ... consumption (=1 to avoid error with obj)
model.l = Var(within=NonNegativeReals, initialize=1)    # ... leisure
model.p = Var(I, within=NonNegativeReals)               # ... prices
model.t = Var(I, within=NonNegativeReals)               # ... tax rates

# ... maximize utility function
model.utility = Objective(expr=( model.x[1]*model.x[2]*model.l )**(1/3), sense=maximize)

def prices_(model, i):
    return complements(pL >= model.p[i], model.x[i] >= 0)
model.prices = Complementarity(I, rule=prices_)

# ... multiply lhs by denominator avoids div. by 0
def consum_(model, i):
    return complements(model.x[i] * (3*model.p[i]*(1+model.t[i])) >= 100*pL, model.p[i] >= 0)
model.consum = Complementarity(I, rule=consum_)

# ... dropped in reference (?)
model.equatn = Constraint(expr=L*pL == sum(model.x[i]*model.p[i] for i in I) + model.l*pL + G)

model.revenue = Constraint(expr=sum(model.p[i]*model.t[i]*model.x[i] for i in I) >= G)

