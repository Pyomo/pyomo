import pyomo.environ
from pyomo.core import *
from pyomo.bilevel import *

#
# A bilevel program for shortest path
# network interdiction.  The objective
# is to maximize the length of the shortest
# path in the network from s to t.
#

model = AbstractModel()

#
# NODE DATA
#

# The set of nodes
model.nodes = Set()
# Start node for the shortest path
model.s = Param(within=model.nodes)
# End node for the shortest path
model.t = Param(within=model.nodes)

#
# ARC DATA
#

# Arcs
model.A = Set(within=model.nodes*model.nodes)
# Arcs going into of a node
def arc_in_rule(model):
    ans = {}
    for (i,j) in model.A:
        ans.setdefault(j,[]).append(i)
    return ans
model.arc_in = Set(model.nodes, initialize=arc_in_rule)
# Arcs going out of a node
def arc_out_rule(model):
    ans = {}
    for (i,j) in model.A:
        ans.setdefault(i,[]).append(j)
    return ans
model.arc_out = Set(model.nodes, initialize=arc_out_rule)

# Nominal integer length of arc a
model.c = Param(model.A, within=PositiveIntegers)
# Added integer length of arc a if interdicted
model.d = Param(model.A, within=PositiveIntegers)
# Resource required to interdict arc a
model.r = Param(model.A, within=NonNegativeReals)
# Total interdiction resource available
model.Gamma = Param(within=NonNegativeReals)

# Variable that is 1 if arc a is interdicted and 0 else
model.x = Var(model.A, within=Binary)
# Variable that is 1 if arc a is traversed and 0 else
model.y = Var(model.A, within=Binary)

# Minimize the path length
def o_rule(model):
    m = model.model()
    return sum(m.y[a] * (m.c[a] + m.d[a]*m.x[a]))
model.o = Objective(rule=o_rule, sense=maximize)
# Limit the total interdiction cost
def interdictions_rule(model):
    return summation(model.r, model.x) <= model.Gamma
model.interdictions = Constraint()

# Create a submodel.  The argument indicates the upper-level 
# decision variables, which are fixed.
model.sub = SubModel(fixed=model.x)
# Minimize the path length
model.sub.o = Objective(rule=o_rule)
# Flow balance constraint
def flow_rule(sub, i):
    model = sub.model()
    if i == model.s:
        rhs = 1
    elif i == model.t:
        rhs = -1
    else:
        rhs = 0
    return sum(model.y[i,j] for j in model.arc_out[i]) - sum(model.y[j,i] for j in model.arc_in[i]) == rhs
model.sub.flow = Constraint(model.nodes, rule=flow_rule)

