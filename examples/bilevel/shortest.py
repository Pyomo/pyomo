#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *
from pyomo.bilevel import *

model = AbstractModel()

# Largest node ID
model.N = Param(within=PositiveIntegers)
# The set of nodes
model.NODES = RangeSet(1, model.N)
# Start node for the shortest path
model.s = Param(initialize=1, within=model.NODES)
# End node for the shortest path
model.t = Param(initialize=model.N, within=model.NODES)

# Arc start
model.i = Param(model.ARCS)
# Arc end
model.j = Param(model.ARCS)
# Arc tuples
model.ARCS = Set(within=model.NODES * model.NODES)

# Nominal integer length of arc a
model.ca = Param(model.ARCS)
# Added integer length of arc a if interdicted
model.da = Param(model.ARCS)
# Resource required to interdict arc a
model.ra = Param(model.ARCS)
# Total interdiction resource available
model.Gamma = Param(within=PositiveIntegers)

# Variable that is 1 if arc a is interdicted
model.x=Var(model.ARCS, within=Binary)
# Variable that is nonzero if arc a is traversed
model.y=Var(model.ARCS, within=NonNegativeReals)

# Define the objective function
def obj_rule(model):
    M = model.model()
    return sum((M.ca[a] + M.da[a]*M.x[a])*M.y[a] for a in M.ARCS)
model.obj = Objective(rule=obj_rule, sense=maximize)

# Interdiction budget constraint
def budget_rule(model):
    return sum(model.ra[a] * model.x[a] for a in model.ARCS) <= model.Gamma
model.budget = Constraint(rule=budget_rule)


# Define the submodel
model.sub = SubModel(fixed=model.x)
# Define objective function with the same expression but opposite sense
model.sub.obj = Objective(rule=obj_rule, sense=minimize)

# define flow constraints
def _pi(model, i):
    M = model.model()
    if i == value(M.s):
        rhs = 1
    elif i == value(M.t):
        rhs = -1
    else:
        rhs = 0
    return sum(M.y[a] for a in M.ARCS if a[0] == i) - \
           sum(M.y[a] for a in M.ARCS if a[1] == i) == rhs
model.sub.pi = Constraint(model.NODES, rule=_pi)

