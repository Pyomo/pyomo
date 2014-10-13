# liswet1-inv.py  QLR-AN-KKT-v-v-v 
# MPEC Pyomo by William Hart
# Adapted from MPEC AMPL by S. Leyffer, University of Dundee, May 2002.
#
# A QPEC from an idea by Stefan Scholtes:
# Find minimum l_2 norm distance to original QP soln 
# for perturbed right-hand-side
# 
# Original NLP-AMPL Model by Hande Y. Benson
#
# Copyright (C) 2001 Princeton University
# All Rights Reserved
#
# ... from ...
#
#   Source:
#   W. Li and J. Swetits,
#   "A Newton method for convex regression, data smoothing and
#   quadratic programming with bounded constraints",
#   SIAM J. Optimization 3 (3) pp 466-488, 1993.
#
#   SIF input: Nick Gould, August 1994.

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

# ... parameters
model.N = Param()
model.K = Param(initialize=2)

def B_(model, i):
    return 1 if i == 0 else model.B[i-1]*i
model.B = Param(RangeSet(0, model.K), initialize=B_)

def C_(model, i):
    return 1 if i == 0 else (-1)**i * model.B[model.K]/(model.B[i]*model.B[model.K-i])
model.C = Param(RangeSet(0, model.K), initialize=C_)

def T_(model, i):
    return (i-1)/(model.N+model.K-1)
model.T = Param(RangeSet(1, model.N+model.K), initialize=T_)

model.x_star = Param(RangeSet(1, model.N+model.K+1))    # ... solution to forward QP

# ... variables
model.z = Var(RangeSet(1, model.N), within=NonNegativeReals)    # ... control variables (pert. to rhs)
model.x = Var(RangeSet(1, model.N+model.K), initialize=0)       # ... state variables
model.l = Var(RangeSet(1, model.N), within=NonNegativeReals)    # ... multipliers

# ... minimize l_2 norm distance to original solution
def l_2_dist_(model):
    return sum( (model.x[i] - model.x_star[i])**2 for i in range(1, model.N+model.K+1))
model.l_2_dist = Objective(rule=l_2_dist_)

# ... first order conditions
def KKT_(model, i):
    return - (sqrt(model.T[i])+0.1*sin(i)) + model.x[i] \
           - sum(model.C[j+K-i]*model.l[j] for j in sequence(max(i-model.K,1), min(i, model.N))) == 0
model.KKT = Constraint(RangeSet(1, model.N+model.K), rule=KKT_)

# ... constraints on controls
def controls_(model):
    return summation(model.z) >= 0.2
model.controls = Constraint(rule=controls_)

# ... complementarity condition
def compl_(model, j):
    return complements(0 <= model.l[j],
                        sum( model.C[i]*model.x[j+model.K-i] for i in range(0, model.K+1)) >= model.z[j])
model.compl = Complementarity(RangeSet(1, model.N), rule=compl_)
