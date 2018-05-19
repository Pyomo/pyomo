#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# A simple budget-constrained single-commodity network flow problem, 
# given to us by Ruszcynski and modified.
#
# This version first appeared in
# Scalable heuristics for a class of chance-constrained stochastic programs
# JP Watson, RJB Wets, DL Woodruff
# INFORMS Journal on Computing 22 (4), 543-554
#

# Annotated with location of stochastic matrix entries
# for use with pysp2smps conversion tool.

from pyomo.core import *
from pyomo.pysp.annotations import StochasticConstraintBodyAnnotation
from pyomo.pysp.annotations import StochasticConstraintBoundsAnnotation

model = AbstractModel()

#
# Parameters and sets
#

model.Nodes = Set(ordered=True)

model.Arcs = Set(within=model.Nodes*model.Nodes)

# derived set
def aplus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.Arcs if j == v]
model.Aplus = Set(model.Nodes, within=model.Arcs, initialize=aplus_arc_set_rule)

# derived set
def aminus_arc_set_rule(model, v):
    return  [(i, j) for (i, j) in model.Arcs if i == v]
model.Aminus = Set(model.Nodes, within=model.Arcs, initialize=aminus_arc_set_rule)

# demands are assumed to be symmetric, although this is not explicitly checked.
model.Demand = Param(model.Nodes, model.Nodes, default=0.0)

model.CapCost = Param(model.Arcs)

model.b0Cost = Param(model.Arcs)

model.FCost = Param(model.Nodes, model.Nodes)

# the maximum sum of demands across the set of scenarios - necessarily computed externally.
model.M = Param(within=NonNegativeReals)

#
# Variables
#

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

#
# Constraints
#

def flow_balance_constraint_rule(model, v, k, l):
    if v == k:
        return (sum([model.y[k, l, i, j] for (i, j) in model.Aplus[v]]) - sum([model.y[k, l, i, j] for (i, j) in model.Aminus[v]]) + model.Demand[k, l]) == 0.0
    elif v == l:
        return (sum([model.y[k, l, i, j] for (i, j) in model.Aplus[v]]) - sum([model.y[k, l, i, j] for (i, j) in model.Aminus[v]]) - model.Demand[k, l]) == 0.0
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
    
def bought_arc0_constraint_rule(model, k, l, i, j):
    return (0.0, model.M * model.b0[i, j] - model.y[k, l, i, j], None)
model.BoughtArc0Constraint = Constraint(model.Nodes, model.Nodes, model.Arcs, rule=bought_arc0_constraint_rule)

def bought_arc_constraint_rule(model, k, l, i, j):
    return (0.0, model.M * model.b[i, j] - model.y[k, l, i, j], None)
model.BoughtArcConstraint = Constraint(model.Nodes, model.Nodes, model.Arcs, rule=bought_arc_constraint_rule)

def no_arc_capacity_unless_bought_constraint_rule(model, i, j):
   return (0.0, model.M * model.b0[i, j] - model.x[i, j], None)
model.NoArcCapacityUnlessBoughtConstraint = Constraint(model.Arcs, rule=no_arc_capacity_unless_bought_constraint_rule)

#
# Stage-specific cost computations
#

def compute_first_stage_cost_rule(model):
    return (model.FirstStageCost - sum_product(model.CapCost, model.x) - sum_product(model.b0Cost, model.b0)) == 0.0
model.ComputeFirstStageCost = Constraint(rule=compute_first_stage_cost_rule)

# this is a Var and constraint to help DDSIP (dlw, Aug 2016)
def compute_second_stage_cost_rule(model):
    return (model.SecondStageCost - sum_product(model.FCost, model.b)) == 0.0
model.ComputeSecondStageCost = Constraint(rule=compute_second_stage_cost_rule)

#
# PySP Auto-generated Objective
#
# minimize: sum of StageCosts
#
# A active scenario objective equivalent to that generated by PySP is
# included here for informational purposes.
def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

def pysp_annotation_rule(model):
    # to enable SMPS and or DDSIP
    model.stoch_rhs = StochasticConstraintBoundsAnnotation()
    model.stoch_rhs.declare(model.FlowBalanceConstraint)
    model.stoch_matrix = StochasticConstraintBodyAnnotation()
    model.stoch_matrix.declare(model.ComputeSecondStageCost)
model.annotate = BuildAction(rule=pysp_annotation_rule)


