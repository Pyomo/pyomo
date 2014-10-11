# flp4.py  QLR-AN-LCP-v-v-v
# 
# Problem 4 from Fukushima, M. Luo, Z.-Q.Pang, J.-S.,
# "A globally convergent Sequential Quadratic Programming
# Algorithm for Mathematical Programs with Linear Complementarity 
# Constraints", Computational Optimization and Applications, 10(1),
# pp. 5-34, 1998.
#
# This is a QPEC with random data of the form
#
#   minimize    0.5*x^T x + e^T y
#
#   subject to  A x <= b
#           0 <= y  _|_  N x + M y + q >= 0
#
# The data files are:
#
#    p     m     n  data-file       dense/sparse
# ----------------------------------------------------------
#   30    30    50  flp4-1.dat      dense
#   50    60    50  flp4-2.dat      dense
#  100    70    70  flp4-3.dat      dense
#  150   100   100  flp4-4.dat      dense
#  300   300   500  flp4-s-1.dat        0.02 %
#  500   600   500  flp4-s-2.dat        0.02 %
# 1000   700   700  flp4-s-3.dat        0.02 % 
# 1500  1500  1500  flp4-s-4.dat        0.02 % 
#
# ... generated in matlab with genflp4.m/genflp4s.m for dense/sparse.

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

# ... dimensions of the problem
model.m = Param(within=Integers)
model.n = Param(within=Integers)
model.p = Param(within=Integers)

# ... sets of indices
model.MM = RangeSet(1, model.m)
model.NN = RangeSet(1, model.n)
model.PP = RangeSet(1, model.p)

# ... random data (initialized to zero for sparse problems)
model.A = Param(model.PP, model.NN, default=0)      # ... A x <= b
model.b = Param(model.PP, default=0)
model.N = Param(model.MM, model.NN, default=0)       # ... N x + M y + q >= 0
model.M = Param(model.MM, model.MM, default=0)
model.q = Param(model.MM, default=0)

# ... variables
model.x = Var(model.NN, initialize=1)
model.y = Var(model.MM, initialize=0)

def objf_(model):
    return 0.5*sum(model.x[j]**2 for j in model.NN) + summation(model.y)
model.objf = Objective(rule=objf_)

def lincs_(model, k):
    return sum(model.A[k,j]*model.x[j] for j in model.NN) <= model.b[k] 
model.lincs = Constraint(model.PP, rule=lincs_)

def compl_(model, i):
    return complements(0 <= model.y[i],
                       0 <= sum(model.N[i,j]*model.x[j] for j in model.NN) \
                         + sum(model.M[i,l]*model.y[l] for l in model.MM) \
                         + model.q[i])
model.compl = Complementarity(model.MM, rule=compl_)

