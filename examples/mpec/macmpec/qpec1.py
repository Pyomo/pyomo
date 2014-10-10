# qpec1.py	QQR2-MN-v-v
# Original Pyomo coding by William Hart
# Adapted from original AMPL coding by Sven Leyffer, University of Dundee

# A QPEC from H. Jiang and D. Ralph, Smooth SQP methods for mathematical
# programs with nonlinear complementarity constraints, University of
# Melbourne, December 1997.

# Number of variables:  v 
# Number of constraints: v

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

# ... parameters enlarge the problem 
model.n = Param(initialize=10)      # number of controls (upper level vars)
model.m = Param(initialize=20)      # number of states (complementary vars)
model.N = RangeSet(1, model.n)
model.M = RangeSet(1, model.m)      # NOTE: Assume m >= n
model.NM = RangeSet(model.n+1, model.m)

# ... constants
model.rr = Param(model.N, initialize=1.0)
model.ss = Param(model.M, initialize=2.0)

# ... variables
model.x = Var(model.N, initialize=1.0)
model.y = Var(model.M, within=NonNegativeReals, initialize=1.0)

# ... problem statement
def f_(model):
    return  sum( (model.x[i] + model.rr[i])**2 for i in model.N) + \
            sum( (model.y[j] + model.ss[j])**2 for j in model.M)
model.f = Objective(rule=f_)

def lin1_(model, i):
    return complements(0 <= model.y[i] - model.x[i], model.y[i] >= 0)
model.lin1 = Complementarity(model.N, rule=lin1_)

def lin2_(model, i):
    return complements(0 <= model.y[i], model.y[i] >= 0)
model.lin2 = Complementarity(model.NM, rule=lin2_)

