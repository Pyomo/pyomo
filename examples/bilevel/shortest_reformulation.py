import coopr.environ
from coopr.pyomo import *

model = AbstractModel()

model.N = Param(within=PositiveIntegers)
#model.A = Param(within=PositiveIntegers)
model.Gamma = Param(within=PositiveIntegers)

model.s = 1
model.t = Param(initialize=model.N)
model.NODES = RangeSet(1, model.N)
model.ARCS = Set(within=model.NODES * model.NODES)
model.i = Param(model.ARCS)
model.j = Param(model.ARCS)
model.ca = Param(model.ARCS)
model.da = Param(model.ARCS)
model.ra = Param(model.ARCS)

# define variables
model.x=Var(model.ARCS, within=Binary)
model.pi=Var(model.NODES, within=Reals)

# define objective function

def obj_rule(model):
    return  model.pi[14] - model.pi[1]
model.obj = Objective(rule=obj_rule, sense=maximize)

def const_rule(model, i,j):
    return model.pi[model.j[i,j]] - model.pi[model.i[i,j]] - model.da[i,j] * model.x[i,j] <= model.ca[i,j]
model.main_const = Constraint(model.ARCS, rule=const_rule)

# interdiction budget constraint
def budget_rule(model):
    return sum(model.ra[a] * model.x[a] for a in model.ARCS) <= model.r0
model.budget = Constraint(rule=budget_rule)







