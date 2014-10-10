# design-cent-4.py        QOR-AY-NLP-22-9-12
# 
# Design centering problem cast as an MPEC, from an idea by
# O. Stein and G. Still, "Solving semi-infinite optimization 
# problems with Interior Point techniques", Lehrstuhl C fuer 
# Mathematik, Rheinisch Westfaelische Technische Hochschule,
# Preprint No. 96, November 2001.
#
# Maximize the volume of the parameterized body B(x) contained 
# in a second body G, described by a set of convex inequalities.
# 
#   maximize volume( B(x) )
#   subj. to B(x) \subset G
#
# where B(x) = { y | x_3 <= y_1 <= x_1 and x_4 <= y_2 <= x_2 }
# is a box.
#
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee, Jan. 2002

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

# ... sets
model.I = RangeSet(1,4)     # ... design variables
model.J = RangeSet(1,2)     # ... lower level variables
model.K = RangeSet(1,3)     # ... lower level constraints
model.L = RangeSet(1,4)     # ... number of constraints in B(x)

# ... parameters
model.pi = Param(initialize=3.141592654)

# ... initial points from solving design-init-1.mod
model.x0 = Param(model.I, default={1:1, 2:1, 3:-1, 4:-1})
model.y0 = Param(model.J, model.K, default=0)
model.ll0 = Param(model.L, model.K, default=1)

# ... variables
model.x = Var(model.I, initialize=model.x0)                                         # ... description of B(x)
model.y = Var(model.J, model.K, initialize=model.y0)                                # ... contact points of B(x), G
model.ll = Var(model.L, model.K, within=NonNegativeReals, initialize=model.ll0)     # ... multipliers gamma

# ... maximize the volume of the inscribed body
model.volume = Objective(expr=(model.x[1]-model.x[3]) * (model.x[2] - model.x[4]), sense=maximize)

# ... lower level solutions lie in body G
model.g1 = Constraint(expr= - model.y[1,1] - model.y[2,1]**2     <= 0)
model.g2 = Constraint(expr=  model.y[1,2]/4 + model.y[2,2] - 3/4 <= 0)
model.g3 = Constraint(expr=                 - model.y[2,3] - 1   <= 0)

# ... first order conditions for 3 lower level problem
model.KKT_11 = Constraint(expr=  1              + model.ll[1,1] - model.ll[3,1] == 0)
model.KKT_21 = Constraint(expr=  2*model.y[2,1] + model.ll[2,1] - model.ll[4,1] == 0)

model.KKT_12 = Constraint(expr= -1/4            + model.ll[1,2] - model.ll[3,2] == 0)
model.KKT_22 = Constraint(expr=  -1             + model.ll[2,2] - model.ll[4,2] == 0)
model.KKT_13 = Constraint(expr= 0               + model.ll[1,3] - model.ll[3,3] == 0)
model.KKT_23 = Constraint(expr= 1               + model.ll[2,3] - model.ll[4,3] == 0)

# ... complementarity & dual feasibility for lower level problem
def compl_1_(model, k):
    return complements(0 <= model.ll[1,k], model.y[1,k] - model.x[1] <= 0)
model.compl_1 = Complementarity(model.K, rule=compl_1_)

def compl_2_(model, k):
    return complements(0 <= model.ll[2,k], model.y[2,k] - model.x[2] <= 0)
model.compl_2 = Complementarity(model.K, rule=compl_2_)

def compl_3_(model, k):
    return complements(0 <= model.ll[3,k], - model.y[1,k] + model.x[3] <= 0)
model.compl_3 = Complementarity(model.K, rule=compl_3_)

def compl_4_(model, k):
    return complements(0 <= model.ll[4,k], - model.y[2,k] + model.x[4] <= 0)
model.compl_4 = Complementarity(model.K, rule=compl_4_)

