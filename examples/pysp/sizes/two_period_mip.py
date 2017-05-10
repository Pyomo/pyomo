#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# This is the two-period version of the SIZES optimization model.
# Ideally, the general multi-period formulation would suffice with
# sufficient data, but the sp component that specifies the variable->
# stage mapping can't deal with "partial" variables at the moment,
# i.e., subsets of the index set. Once it can, this file is obselete.
#

#
# Imports
#

from pyomo.core import *

#
# Model
#

model = AbstractModel()

#
# Parameters
#

# the number of product sizes.
model.NumSizes = Param(within=NonNegativeIntegers)

# the set of sizes, labeled 1 through NumSizes.
def product_sizes_rule(model):
    return set(range(1, model.NumSizes()+1))
model.ProductSizes = Set(initialize=product_sizes_rule)

# the deterministic demands for product at each size.
model.DemandsFirstStage = Param(model.ProductSizes, within=NonNegativeReals)
model.DemandsSecondStage = Param(model.ProductSizes, within=NonNegativeReals)

# the unit production cost at each size.
model.UnitProductionCosts = Param(model.ProductSizes, within=NonNegativeReals)

# the setup cost for producing any units of size i.
model.SetupCosts = Param(model.ProductSizes, within=NonNegativeReals)

# the unit penalty cost of meeting demand for size j with larger size i.
model.UnitPenaltyCosts = Param(model.ProductSizes, within=NonNegativeReals)

# the cost to reduce a unit i to a lower unit j.
model.UnitReductionCost = Param(within=NonNegativeReals)

# a cap on the overall production within any time stage.
model.Capacity = Param(within=PositiveReals)

# a derived set to constraint the NumUnitsCut variable domain.
# TBD: the (i,j) with i < j set should be a generic utility.
def num_units_cut_domain_rule(model):
    ans = set()
    for i in range(1,model.NumSizes+1):
        for j in range(1, i+1):
            ans.add((i,j))
    return ans

model.NumUnitsCutDomain = Set(initialize=num_units_cut_domain_rule,dimen=2)

# TBD - should be able to define the "M" here, after the total demands are defined.

#
# Variables
#

# are any products at size i produced?
model.ProduceSizeFirstStage = Var(model.ProductSizes, domain=Boolean)
model.ProduceSizeSecondStage = Var(model.ProductSizes, domain=Boolean)

# the number of units at each size produced.
# TBD - the bounds should link/propagate through the domain, but they aren't.
model.NumProducedFirstStage = Var(model.ProductSizes, domain=NonNegativeIntegers, bounds=(0.0, None))
model.NumProducedSecondStage = Var(model.ProductSizes, domain=NonNegativeIntegers, bounds=(0.0, None))

# the number of units of size i cut (down) to meet demand for units of size j.
# TBD - the bounds should link/propagate through the domain, but they aren't.
# TBD - can actually/should be implicitly integer.
model.NumUnitsCutFirstStage = Var(model.NumUnitsCutDomain, domain=NonNegativeIntegers, bounds=(0.0, None))
model.NumUnitsCutSecondStage = Var(model.NumUnitsCutDomain, domain=NonNegativeIntegers, bounds=(0.0, None))

model.FirstStageCost = Var()
model.SecondStageCost = Var()

#
# Constraints
#

# ensure that demand is satisfied in each time stage.
def demand_satisfied_first_stage_rule(i, model):
    expr = 0.0
    # TBD - simplify loop with range construct
    for j in model.ProductSizes:
        if j >= i:
            expr += model.NumUnitsCutFirstStage[j,i]
    expr -= model.DemandsFirstStage[i]
    return (0.0, expr, None)

def demand_satisfied_second_stage_rule(i, model):
    expr = 0.0
    for j in model.ProductSizes:
        if j >= i:
            expr += model.NumUnitsCutSecondStage[j,i]
    expr -= model.DemandsSecondStage[i]
    return (0.0, expr, None)

model.DemandSatisfiedFirstStage = Constraint(model.ProductSizes, rule=demand_satisfied_first_stage_rule)
model.DemandSatisfiedSecondStage = Constraint(model.ProductSizes, rule=demand_satisfied_second_stage_rule)

# ensure that you don't produce any units if the decision has been made to disable producion.
def enforce_production_first_stage_rule(i, model):
    # TBD - compute M as the maximal demand - really max across all scenarios, which complicates things.
    M = 10000000
    expr = model.NumProducedFirstStage[i] - M * model.ProduceSizeFirstStage[i]
    return (None, expr, 0.0)

def enforce_production_second_stage_rule(i, model):
    # TBD - compute M as the maximal demand
    M = 10000000
    expr = model.NumProducedSecondStage[i] - M * model.ProduceSizeSecondStage[i]
    return (None, expr, 0.0)

model.EnforceProductionBinaryFirstStage = Constraint(model.ProductSizes, rule=enforce_production_first_stage_rule)
model.EnforceProductionBinarySecondStage = Constraint(model.ProductSizes, rule=enforce_production_second_stage_rule)

# ensure that the production capacity is not exceeded for each time stage.
def enforce_capacity_first_stage_rule(model):
    expr = 0.0
    for i in model.ProductSizes:
        expr += model.NumProducedFirstStage[i]
    expr -= model.Capacity
    return (None, expr, 0.0)

def enforce_capacity_second_stage_rule(model):
    expr = 0.0
    for i in model.ProductSizes:
        expr += model.NumProducedSecondStage[i]
    expr -= model.Capacity
    return (None, expr, 0.0)

model.EnforceCapacityLimitFirstStage = Constraint(rule=enforce_capacity_first_stage_rule)
model.EnforceCapacityLimitSecondStage = Constraint(rule=enforce_capacity_second_stage_rule)

# ensure that you can't generate inventory out of thin air.
def enforce_inventory_first_stage_rule(i, model):
    expr = 0.0
    for j in model.ProductSizes:
        if j <= i:
            expr += model.NumUnitsCutFirstStage[i,j]
    expr -= model.NumProducedFirstStage[i]
    return (None, expr, 0.0)

def enforce_inventory_second_stage_rule(i, model):
    expr = 0.0
    for j in model.ProductSizes:
        if j <= i:
            expr += model.NumUnitsCutFirstStage[i,j]
    expr -= model.NumProducedFirstStage[i]
    for j in model.ProductSizes:
        if j <= i:
            expr += model.NumUnitsCutSecondStage[i,j]
    expr -= model.NumProducedSecondStage[i]
    return (None, expr, 0.0)

model.EnforceInventoryFirstStage = Constraint(model.ProductSizes, rule=enforce_inventory_first_stage_rule)
model.EnforceInventorySecondStage = Constraint(model.ProductSizes, rule=enforce_inventory_second_stage_rule)

#
# Stage-specific cost computations
#

def first_stage_cost_rule(model):
    # production costs followed by cut-down penalties.
    expr = 0.0
    for i in model.ProductSizes:
        expr += (model.SetupCosts[i] * model.ProduceSizeFirstStage[i] + model.UnitProductionCosts[i] * model.NumProducedFirstStage[i])
    for (i,j) in model.NumUnitsCutDomain:
        if i != j:
            expr += model.UnitReductionCost * model.NumUnitsCutFirstStage[i,j];
    return (model.FirstStageCost - expr) == 0.0

model.ComputeFirstStageCost = Constraint(rule=first_stage_cost_rule)

def second_stage_cost_rule(model):
    # production costs followed by cut-down penalties.
    expr = 0.0
    for i in model.ProductSizes:
        expr += (model.SetupCosts[i] * model.ProduceSizeSecondStage[i] + model.UnitProductionCosts[i] * model.NumProducedSecondStage[i])
    for (i,j) in model.NumUnitsCutDomain:
        if i != j:
            expr += model.UnitReductionCost * model.NumUnitsCutSecondStage[i,j];
    return (model.SecondStageCost - expr) == 0.0

model.ComputeSecondStageCost = Constraint(rule=second_stage_cost_rule)

#
# Objective
#

def total_cost_rule(model):
    return (model.FirstStageCost + model.SecondStageCost)

model.TotalCostObjective = Objective(rule = total_cost_rule, sense=minimize)
