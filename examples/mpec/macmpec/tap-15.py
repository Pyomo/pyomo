# tap-15.py  OOR-AY-NCP-86-68-32
#
# Traffic equilibrium & toll pricing model derived from 
# tollmpec (Dirkse & Ferris) with invented data for 15 node
# model.
#
# Pyomo coding by William Hart
# Adapted from ampl by S. Leyffer, University of Dundee, Feb. 2001.

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.N = RangeSet(1, 15)                       # set of nodes
model.DEST = Set(initialize=[3,4,11])           # destination nodes

model.ARCS = Set(within=model.N*model.N)        # arcs

model.d = Param(model.N, model.N, default=0)    # demand (only /=0 for O-D pairs)
model.T = Param(model.ARCS)                     # cost coeff. for separable cost functn
model.b = Param(model.ARCS)                     # cost coeff. for separable cost functn

# ... flow along arc i-j to k 
def x_index(model):
    for (i,j) in model.ARCS:
        for k in model.DEST:
            if i != k:
                yield (i,j,k)
model.x = Var(x_index, within=NonNegativeReals)

# ... aggregate flow on arc i-j
model.F = Var(model.ARCS)

# ... toll on arc i-j (all roads are tolled ?)
model.toll =Var(model.ARCS, within=NonNegativeReals)

# ... time to get from node i to node j
model.time = Var(model.N, model.N, within=NonNegativeReals)

### ... maximize system revenue (unbounded !!!)
##maximize nprofit: sum{(i,j) in ARCS, k in DEST: i != k} toll[i,j] * F[i,j];

# ... minimize system cost (or congestion)
model.congestion = Objective(expr=sum(model.T[i,j]*(1 + 0.15 * (model.F[i,j]/model.b[i,j])**4) for (i,j) in model.ARCS))

# ... 2nd Wardrop principle
def rational_(model, i, j, k):
    return complements(0 <= model.T[i,j]*(1 + 0.15 * (model.F[i,j]/model.b[i,j])**4) + \
                            model.time[j,k] + 100 * model.toll[i,j] - model.time[i,k],
                       model.x[i,j,k] >= 0)
def rational_index(model):
    for (i,j) in model.ARCS:
        for k in model.DEST:
            if i != k:
                yield (i,j,k)
model.rational = Complementarity(rational_index, rule=rational_)

# ... the flow into a node equals demand plus flow out:
def balance_(model, i, j_, k):
    return sum(model.x[i,j,k] for j in model.N if (i,j) in model.ARCS and i != k) - \
           sum(model.x[j,i,k] for j in model.N if (j,i) in model.ARCS and j != k) \
           == model.d[i,k]
def balance_index(model):
    for (i,j) in model.ARCS:
        for k in DEST:
            if i != k:
                yield (i,j,k)
model.balance = Constraint(balance_index, rule=balance_)

# ... flow on a given arc constitutes flows to all destinations K:
def fdef_(model, i, j):
    return model.F[i,j] == sum(model.x[i,j,l] for l in model.DEST if l != 1)
model.fdef = Constraint(model.ARCS, rule=fdef_)
