# -------------------------------------------------------
#
#  TRAFFIC EQUILIBRIUM - MPEC FORMULATION
#
#  From a GAMS model by S.P. Dirkse & M.C. Ferris (MPECLIB),
#  (see http://www.gams.com/mpec/).
#  See also "Traffic Modeling and Variational Inequalities using 
#  GAMS", by Dirkse & Ferris, University of Wisconsin, CS, 1997.
#
#  Pyomo coding William Hart
#  Adapted from AMPL coding Sven Leyffer, University of Dundee, Jan. 2000
#  (removing redundant complementarity, i.e. to force equations)
#
#  data file is siouxfls.dat
#  -------------------------------------------------------

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

model.N = RangeSet(1,24)    # set of nodes
model.DEST = RangeSet(1,24) # destination nodes

model.ARCS = Set(within=model.N*model.N)    # arcs
model.TOLL = Set(within=model.ARCS)		    # tolled arcs

model.clo = Param(model.TOLL)       # lower bnd on cost
model.cup = Param(model.TOLL)       # upper bnd on cost
model.d = Param(model.N,model.N)    # trip matrix / table
model.A = Param(model.ARCS)         # cost coeff. for separable cost functn
model.B = Param(model.ARCS)         # cost coeff. for separable cost functn
model.K = Param(model.ARCS)         # cost coeff. for separable cost functn

# ... flow along arc i-j to k 
def x_index(model):
    for (i,j) in model.ARCS:
        for k in model.DEST:
            if i != k:
                yield(i,j,k)
model.x = Var(x_index, within=NonNegativeReals)

# ... aggregate flow on arc i-j
model.F = Var(model.ARCS)

# ... tarriff on arc i-j
def trffcost_bounds(model, i, j):
    return (model.clo[i,j], model.cup[i,j])
model.trffcost = Var(model.TOLL, bounds=trffcost_bounds)

# ... time to get from node i to node j
model.T = Var(model.N, model.N, within=NonNegativeReals)

# ... minimize system cost (or congestion)
def congestion_(model):
    return sum( model.A[i,j] + model.B[i,j] * ( model.F[i,j]/model.K[i,j] )**4 for (i,j) in model.ARCS)
model.congestion = Objective(rule=congestion_)

# The following constraint imposes individual rationality:
# The time to reach node k from node i is no greater than
# the time required to travel from node i to node j and then
# from node j to node k. (2nd Wardrop principle)

def rational_(model, i, j, k):
    return complements(
	    0 <= model.A[i,j] + model.B[i,j] * ( model.F[i,j]/model.K[i,j] )**4 + model.T[j,k] \
             + ( 100*model.trffcost[i,j] if (i,j) in model.TOLL else 0.0 ) - model.T[i,k],
        model.x[i,j,k] >= 0)
model.rational = Complementarity(x_index, rule=rational_)

# The flow into a node equals demand plus flow out:

def balance_index(model):
    for i in model.N:
        for k in model.DEST:
            if i != k:
                yield (i,k)
def balance_(model, i, k):
    return \
        sum(model.x[i,j,k] for j in model.N if (i,j) in model.ARCS and i != k) - \
        sum(model.x[j,i,k] for j in model.N if (j,i) in model.ARCS and j != k) \
        == model.d[i,k]
model.balance = Constraint(x_index, rule=balance_)

# Flow on a given arc constitutes flows to all destinations K:

def fdef_(model, i, j):
    return model.F[i,j] == sum(model.x[i,j,l] for l in model.DEST if l != i)
model.fdef = Constraint(model.ARCS, rule=fdef_)

