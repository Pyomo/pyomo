#
# A simple budget-constrained single-commodity network flow problem, taken from Ruszcynski.
#

from pyomo.core import *

#
# Model
#

model = AbstractModel()

#
# Parameters
#

model.Nodes = Set(ordered=True)

model.Arcs = Set(within=model.Nodes*model.Nodes)

# derived set
def aplus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.Arcs if j == v]
model.Aplus = Set(model.Nodes, within=model.Arcs, initialize=aplus_arc_set_rule)
#                                                 ########## changed from rule 1/31/14

# derived set
def aminus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.Arcs if i == v]
model.Aminus = Set(model.Nodes, within=model.Arcs, initialize=aminus_arc_set_rule)
#                                                 ########## changed from rule 1/31/14

# demands are assumed to be symmetric, although this is not explicitly checked.
model.Demand = Param(model.Nodes, model.Nodes, initialize=0.0)

model.CapCost = Param(model.Arcs)

model.b0Cost = Param(model.Arcs)

model.FCost = Param(model.Nodes, model.Nodes)

# the maximum sum of demands across the set of scenarios - necessarily computed externally.
model.M = Param(within=NonNegativeReals)

# Threshold for the UnmetDemand (percentage of Demand)
model.UnmetDemandThreshold = Param(within=NonNegativeReals)

# lambdaMult is the Lagrangian multiplier for the relaxed chance constraint
# (note: lambda is a reserved word in Python)
# note: a default makes no sense so watch out because zero is special (no delta)
model.lambdaMult = Param(initialize = 0, mutable=True)

# big M is needed to establish delta
# set it to total demand + 1
# note: without value() a bad lp file will be written
def BigM_Init(model):
   return sum(sum(value(model.Demand[i,j]) for i in model.Nodes) for j in model.Nodes) + 1
   #return 400000
model.BigM = Param(initialize=BigM_Init)

#
# Variables
#

# delta is the indicator that we meet demand (so 1-delta indicates that we shed load)
model.delta = Var(within=Binary)

# UnmetDemand
model.UnmetDemand = Var(model.Nodes, model.Nodes, within=NonNegativeReals)

# total arc capacity - a first stage variable.
model.x = Var(model.Arcs, within=NonNegativeReals)

# from between the nodes through the arc - a second stage variable.
model.y = Var(model.Nodes, model.Nodes, model.Arcs, within=NonNegativeReals)

# first stage budget allocation variable
model.b0 = Var(model.Arcs, within=Binary)

# second stage budget allocation variable
model.b = Var(model.Arcs, within=Binary)

# the cost variables for the stages, in isolation.
model.FirstStageCost = Var()
model.SecondStageCost = Var()
model.fofx = Var() 


#
# Constraints
#

def flow_balance_constraint_rule(model, v, k, l):
    if v == k:
        return (sum([model.y[k, l, i, j] for (i, j) in model.Aplus[v]]) - sum([model.y[k, l, i, j] for (i, j) in model.Aminus[v]]) + (model.Demand[k, l] - model.UnmetDemand[k, l])) == 0.0
    elif v == l:
        return (sum([model.y[k, l, i, j] for (i, j) in model.Aplus[v]]) - sum([model.y[k, l, i, j] for (i, j) in model.Aminus[v]]) - (model.Demand[k, l] - model.UnmetDemand[k, l])) == 0.0
    else:
        return (sum([model.y[k, l, i, j] for (i, j) in model.Aplus[v]]) - sum([model.y[k, l, i, j] for (i, j) in model.Aminus[v]])) == 0.0      
model.FlowBalanceConstraint = Constraint(model.Nodes, model.Nodes, model.Nodes, rule=flow_balance_constraint_rule)

def capacity_constraint_rule(model, i, j):
    return (None, sum([model.y[k, l, i, j] for k in model.Nodes for l in model.Nodes]) - model.x[i, j], 0.0)
model.CapacityConstraint = Constraint(model.Arcs, rule=capacity_constraint_rule)

def x_symmetry_constraint_rule(model, i, j):
    return (model.x[i, j] - model.x[j, i]) == 0.0
model.xSymmetryConstraint = Constraint(model.Arcs, rule=x_symmetry_constraint_rule)

def b0_symmetry_constraint_rule(model, i, j):
    return (model.b0[i, j] - model.b0[j, i]) == 0.0
model.b0SymmetryConstraint = Constraint(model.Arcs, rule=b0_symmetry_constraint_rule)
    
def b_symmetry_constraint_rule(model, i, j):
    return (model.b[i, j] - model.b[j, i]) == 0.0
model.bSymmetryConstraint = Constraint(model.Arcs, rule=b_symmetry_constraint_rule)
    
#def bought_arc0_constraint_rule(model, k, l, i, j):
#    return (0.0, model.M * model.b0[i, j] - model.y[k, l, i, j], None)
#model.BoughtArc0Constraint = Constraint(model.Nodes, model.Nodes, model.Arcs, rule=bought_arc0_constraint_rule)

#def bought_arc_constraint_rule(model, k, l, i, j):
#    return (0.0, model.M * model.b[i, j] - model.y[k, l, i, j], None)
#model.BoughtArcConstraint = Constraint(model.Nodes, model.Nodes, model.Arcs, rule=bought_arc_constraint_rule)

def no_arc_capacity_unless_bought_constraint_rule(model, i, j):
   return (0.0, model.M * model.b0[i, j] - model.x[i, j], None)
model.NoArcCapacityUnlessBoughtConstraint = Constraint(model.Arcs, rule=no_arc_capacity_unless_bought_constraint_rule)

# UnmetDemand Threshold: the UnmetDemand for each node-pair has to be smaller than the 
# threshold (percentage) times the corresponding Demand for this node-pair
def demand_service_threshold_rule(model, k, l):
   return model.UnmetDemand[k, l] <= model.Demand[k, l] * model.UnmetDemandThreshold

model.DemandServiceThreshold = Constraint(model.Nodes, model.Nodes, rule=demand_service_threshold_rule)

# establish the value for delta
def establish_delta_rule(model):
   return summation(model.UnmetDemand) <= model.BigM * (1-model.delta)
model.Establishdelta = Constraint(rule=establish_delta_rule)

#
# Stage-specific cost computations
#

def compute_first_stage_cost_rule(model):
    return (model.FirstStageCost - summation(model.CapCost, model.x) - summation(model.b0Cost, model.b0)) == 0.0
model.ComputeFirstStageCost = Constraint(rule=compute_first_stage_cost_rule)

def compute_second_stage_cost_rule(model):
    return model.SecondStageCost == summation(model.FCost, model.b) - (model.lambdaMult * model.delta)
model.ComputeSecondStageCost = Constraint(rule=compute_second_stage_cost_rule)

def fofx_rule(model):
  return ((model.fofx - model.FirstStageCost - model.SecondStageCost) == 0.0)
model.Compute_fofx = Constraint(rule=fofx_rule)
#
# Objective
#

def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost

model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

