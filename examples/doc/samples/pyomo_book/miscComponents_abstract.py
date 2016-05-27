from pyomo.environ import *
import random

random.seed(1000)

# @all:
#
# Create a maxflow problem on a random graph
#
model = AbstractModel()

model.N = Param(within=Integers)

def edges_rule(m):
    return [(i,j) for i in sequence(m.N)
                  for j in sequence(m.N)
                  if i != j]
model.edges = Set(dimen=2, rule=edges_rule)

def check_rule(m):
    # An error check
    return len(m.edges) == m.N*(m.N-1)
model.check = BuildCheck(rule=check_rule)

def w_rule(m,i,j):
    return random.randint(1,10)
model.w = Param(model.edges, initialize=w_rule)

def action_rule(m):
    # A debugging statement
    print "Edge weights"
    for e in sorted(m.edges):
        print e, value(m.w[e])
model.action = BuildAction(rule=action_rule)

model.x = Var(model.edges, within=NonNegativeReals)

def obj_rule(m):
    N = value(m.N)
    return sum(m.x[i,N] for i in sequence(m.N)
                        if (i,N) in m.edges)
model.obj = Objective(sense=maximize)

def flow_rule(m, i):
    return sum(m.x[j,i] for j in sequence(m.N-1)
                        if (j,i) in m.edges) == \
           sum(m.x[i,j] for j in sequence(2,m.N)
                        if (i,j) in m.edges)
model.flow = Constraint(RangeSet(2,model.N-1))

def limit_rule(m, i, j):
    return m.x[i,j] <= m.w[i,j]
model.limit = Constraint(model.edges)
# @:all
