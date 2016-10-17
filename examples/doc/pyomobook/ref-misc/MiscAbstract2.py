import pyomo.environ
from pyomo.core import *
import random

random.seed(1000)

#
# Create a maxflow problem on a random graph
#
model = AbstractModel()

model.N = Param(within=Integers)

def edges_rule(m):
    return [(i,j) for i in sequence(m.N)
                  for j in sequence(m.N)
                  if i != j]
model.edges = Set(dimen=2, initialize=edges_rule, ordered=True)

def check_rule(m):
    # An error check
    return len(m.edges) == m.N*(m.N-1)
model.check = BuildCheck(rule=check_rule)

def w_rule(m, i, j):
    return random.randint(1,10)
model.w = Param(model.edges, initialize=w_rule)

def preaction_rule(m):
    print('"')
    print('action: "')
model.preaction = BuildAction(rule=preaction_rule)

# @all:
def action_rule(m, i, j):
    # A debugging statement
    print("%d %d %d" % (i,j, value(m.w[i,j])))
model.action = BuildAction(model.edges, rule=action_rule)
# @:all

def postaction_rule(m):
    print('"')
    print('log2: "')
model.postaction = BuildAction(rule=postaction_rule)

model.x = Var(model.edges, within=NonNegativeReals)

def obj_rule(m):
    N = value(m.N)
    return sum(m.x[i,N] for i in sequence(m.N)
                        if (i,N) in m.edges)
model.obj = Objective(sense=maximize, rule=obj_rule)

def flow_rule(m, i):
    return sum(m.x[j,i] for j in sequence(m.N-1)
                        if (j,i) in m.edges) == \
           sum(m.x[i,j] for j in sequence(2,m.N)
                        if (i,j) in m.edges)
model.flow = Constraint(RangeSet(2,model.N-1), rule=flow_rule)

def limit_rule(m, i, j):
    return m.x[i,j] <= m.w[i,j]
model.limit = Constraint(model.edges, rule=limit_rule)
