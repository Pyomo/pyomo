# portfl-i.py  QLR2-AY-NLP-87-25-12
# Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Feb. 2001,
# from portfl models of Bob Vanderbei.

# A QPEC obtained by varying the grad of portfl1-6
# from an idea communicated by S. Scholtes.
#
# This is almost a sensible problem. Here, ask what 
# vector r of minimum norm perturbations to returns R, 
# gives a solution that is as close as possible to the 
# given solution (obtained by rounding soln to portfl1-6).
#
# Problem has data files portfl1.dat - portfl6.dat with
# different parameters F & R.

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = AbstractModel()

model.NS = Param(initialize= 12)        # ... number of securites in portfolio
model.NR = Param(initialize= 62)        # ... total number of stocks

model.F = Param(RangeSet(1,model.NS), RangeSet(1,model.NR))     # ... covariance (?)
model.R = Param(RangeSet(1,model.NR))                           # ... expected returns

model.sol = Param(RangeSet(1,model.NS)) # ... primal solution of portfl - QP

# ... primal/dual variables of original QP
def s_bound(model, i):
    return (0.0, 1/model.NS)
model.s = Var(RangeSet(1,model.NS), bounds=s_bound)  # ... size of stock i in portfolio
model.m = Var(RangeSet(1,model.NS), bounds=(0.0,None))   # ... multipliers of s[i] >= 0
model.l = Var()                                     # ... multipliers of e^T s = 1

# ... perturbations to stock returns
model.r = Var(RangeSet(1,model.NR), bounds=(0,None)) # ... perturbation to R

# ... minimize deviation from primal solution PLUS perturbation norm
def diff_(model):
    return sum(( model.s[i] - model.sol[i] )**2 for i in sequence(model.NS)) + sum( model.r[i]**2 for i in sequence(model.NR))
model.diff = Objective(rule=diff_)

# ... 1st order condition for s
def KKT_(model, k):
      0 == sum(2*(sum(model.s[j]*model.F[j,i] for j in sequence(model.NS)) - (model.R[i]+model.r[i]))*model.F[k,i] for i in sequence(model.NR)) - model.l - model.m[k]
model.KKT = Constraint(RangeSet(1, model.NS), rule=KKT_)

# ... primal feasibility (stock sum to 1)
def cons1_(model):
    return sum(s[i] for i in sequence(model.NS)) == 1
model.cons1 = Constraint(rule=cons1_)
   
# ... complementary slackness condition
def compl_s_(model, i):
    return complements(0 <= model.s[i], model.m[i] >= 0)
model.compl_s = Complementarity(RangeSet(1, model.NS), rule=compl_s_)

