# hs044-i.py   QLR2-MY-20-14-10
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# A QPEC obtained by varying the rhs of HS44 (a QP)
# from an idea communicated by S. Scholtes.

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = AbstractModel()

I = range(1,5)          # ... variables of HS44
J = range(1,7)          # ... constraints of HS44
K = range(1,7)          # ... perturbations to HS44

# ... data of HS44
model.sol = Param(I)                # ... solution to HS44
model.A = Param(J, I, default=0)    # ... constraint matrix
model.b = Param(J, default=0)       # ... rhs of constraints
model.H = Param(I, I, default=0)    # ... Hessian matrix of HS44
model.g = Param(I, default=0)       # ... linear part of objective of HS44

# ... bounds on perturbations
model.zl = Param(K, default=0)
model.zu = Param(K, default=10)
model.u = Param(K)
model.v = Param(K)

# ... variables (states, i.e. primal/dual pair of HS44)
model.x = Var(I, within=NonNegativeReals)   # ... original primal variables
model.l = Var(J, within=NonNegativeReals)   # ... multipliers of general constraints
model.m = Var(I, within=NonNegativeReals)   # ... multipliers of simple bounds

# ... variables (controls, i.e. perturbations to rhs & g)
def z_bounds(model, k):
    return (zl[k], zu[k])
model.z = Var(K, bounds=z_bounds)

# ... minimize l_2 norm of derivation from optimal solution
def norm_(model):
    return sum( (model.sol[i] - model.x[i])**2 for i in I)
model.norm = Objective(rule=norm_)

# ... perturbed KKT conditions (1st order)
def KKT_(model, i):
    return sum(model.H[i,ii]*model.x[ii] for ii in I) + (model.g[i] + model.u[i]*model.z[i]) \
                - sum(model.A[j,i]*model.l[i] for j in J) - model.m[i] == 0
model.KKT = Constraint(I, rule=KKT_)

# ... perturbed slackness conditions (general c/s)
def slackness_g_(model, j):
    return complements(0 <= model.l[j],
                       model.b[j] - model.v[j]*model.z[j] + sum(model.A[j,i]*model.x[i] for i in I) >= 0)
model.slackness_g = Complementarity(J, rule=slackness_g_)
             
# ... perturbed slackness conditions (simple bounds)
def slackness_x_(model, i):
    return complements(0 <= model.x[i], model.m[i] >= 0)
model.slackness_x = Complementarity(I, rule=slackness_x_)
