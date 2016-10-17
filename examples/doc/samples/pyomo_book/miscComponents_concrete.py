from pyomo.environ import *
from pyomo.opt import *
import random

random.seed(1000)

# @all:
def create(N):
    #
    # Create a maxflow problem on a random graph
    #
    model = ConcreteModel()

    def edges_rule(m):
        return [(i,j) for i in sequence(N)
                      for j in sequence(N)
                      if i != j]
    model.edges = Set(dimen=2, rule=edges_rule)

    if len(model.edges) != N*(N-1):
        raise RuntimeError, "Check failed"

    def w_rule(m,i,j):
        return random.randint(1,10)
    model.w = Param(model.edges, initialize=w_rule)

    print "Edge weights"
    for e in sorted(model.edges):
        print e, value(model.w[e])

    model.x = Var(model.edges, within=NonNegativeReals)

    def obj_rule(m):
        return sum(m.x[i,N] for i in sequence(N)
                            if (i,N) in m.edges)
    model.obj = Objective(sense=maximize)

    def flow_rule(m, i):
        return sum(m.x[j,i] for j in sequence(N-1)
                            if (j,i) in m.edges) == \
               sum(m.x[i,j] for j in sequence(2,N)
                            if (i,j) in m.edges)
    model.flow = Constraint(RangeSet(2,N-1))

    def limit_rule(m, i, j):
        return m.x[i,j] <= m.w[i,j]
    model.limit = Constraint(model.edges)

    return model

model = create(4)
# @:all

instance = model.create()
opt = SolverFactory('glpk')
results = opt.solve(instance)
print results
