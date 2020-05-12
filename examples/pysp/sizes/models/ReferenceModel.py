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
# This is the two-period version of the SIZES optimization model
# derived from the three stage model in:
#A. L{\o}kketangen and D. L. Woodruff,
#"Progressive Hedging and Tabu Search Applied to Mixed Integer (0,1) Multistage Stochastic Programming",
#Journal of Heuristics, 1996, Vol 2, Pages 111-128.


import pyomo.environ as pyo

#
# Model
#

model = pyo.AbstractModel()

#
# Parameters
#

# the number of product sizes.
model.NumSizes = pyo.Param(within=pyo.NonNegativeIntegers)

# the set of sizes, labeled 1 through NumSizes.
def product_sizes_rule(model):
    return list(range(1, model.NumSizes()+1))
model.ProductSizes = pyo.Set(initialize=product_sizes_rule)

# the deterministic demands for product at each size.
model.DemandsFirstStage = pyo.Param(model.ProductSizes, within=pyo.NonNegativeIntegers)
model.DemandsSecondStage = pyo.Param(model.ProductSizes, within=pyo.NonNegativeIntegers)

# the unit production cost at each size.
model.UnitProductionCosts = pyo.Param(model.ProductSizes, within=pyo.NonNegativeReals)

# the setup cost for producing any units of size i.
model.SetupCosts = pyo.Param(model.ProductSizes, within=pyo.NonNegativeReals)

# the cost to reduce a unit i to a lower unit j.
model.UnitReductionCost = pyo.Param(within=pyo.NonNegativeReals)

# a cap on the overall production within any time stage.
model.Capacity = pyo.Param(within=pyo.PositiveReals)

# a derived set to constrain the NumUnitsCut variable domain.
# TBD: the (i,j) with i >= j set should be a generic utility.
def num_units_cut_domain_rule(model):
    return ((i,j) for i in range(1,model.NumSizes()+1) for j in range(1,i+1))

model.NumUnitsCutDomain = pyo.Set(initialize=num_units_cut_domain_rule, dimen=2)

#
# Variables
#

# are any products at size i produced?
model.ProduceSizeFirstStage = pyo.Var(model.ProductSizes, domain=pyo.Boolean)
model.ProduceSizeSecondStage = pyo.Var(model.ProductSizes, domain=pyo.Boolean)

# NOTE: The following (num-produced and num-cut) variables are implicitly integer
#       under the normal cost objective, but with the PH cost objective, this isn't
#       the case.

# the number of units at each size produced.
model.NumProducedFirstStage = pyo.Var(model.ProductSizes, domain=pyo.NonNegativeIntegers, bounds=(0.0, model.Capacity))
model.NumProducedSecondStage = pyo.Var(model.ProductSizes, domain=pyo.NonNegativeIntegers, bounds=(0.0, model.Capacity))

# the number of units of size i cut (down) to meet demand for units of size j.
model.NumUnitsCutFirstStage = pyo.Var(model.NumUnitsCutDomain,
                                      domain=pyo.NonNegativeIntegers,
                                      bounds=(0.0, model.Capacity))
model.NumUnitsCutSecondStage = pyo.Var(model.NumUnitsCutDomain,
                                       domain=pyo.NonNegativeIntegers,
                                       bounds=(0.0, model.Capacity))

# stage-specific cost variables for use in the pysp scenario tree / analysis.
model.FirstStageCost = pyo.Var(domain=pyo.NonNegativeReals)
model.SecondStageCost = pyo.Var(domain=pyo.NonNegativeReals)

#
# Constraints
#

# ensure that demand is satisfied in each time stage, accounting for cut-downs.
def demand_satisfied_first_stage_rule(model, i):
    return (0.0, sum([model.NumUnitsCutFirstStage[j,i] for j in model.ProductSizes if j >= i]) - model.DemandsFirstStage[i], None)

def demand_satisfied_second_stage_rule(model, i):
    return (0.0, sum([model.NumUnitsCutSecondStage[j,i] for j in model.ProductSizes if j >= i]) - model.DemandsSecondStage[i], None)

model.DemandSatisfiedFirstStage = pyo.Constraint(model.ProductSizes, rule=demand_satisfied_first_stage_rule)
model.DemandSatisfiedSecondStage = pyo.Constraint(model.ProductSizes, rule=demand_satisfied_second_stage_rule)

# ensure that you don't produce any units if the decision has been made to disable producion.
def enforce_production_first_stage_rule(model, i):
    # The production capacity per time stage serves as a simple upper bound for "M".
    return (None, model.NumProducedFirstStage[i] - model.Capacity * model.ProduceSizeFirstStage[i], 0.0)

def enforce_production_second_stage_rule(model, i):
    # The production capacity per time stage serves as a simple upper bound for "M".
    return (None, model.NumProducedSecondStage[i] - model.Capacity * model.ProduceSizeSecondStage[i], 0.0)

model.EnforceProductionBinaryFirstStage = pyo.Constraint(model.ProductSizes, rule=enforce_production_first_stage_rule)
model.EnforceProductionBinarySecondStage = pyo.Constraint(model.ProductSizes, rule=enforce_production_second_stage_rule)

# ensure that the production capacity is not exceeded for each time stage.
def enforce_capacity_first_stage_rule(model):
    return (None, sum([model.NumProducedFirstStage[i] for i in model.ProductSizes]) - model.Capacity, 0.0)

def enforce_capacity_second_stage_rule(model):
    return (None, sum([model.NumProducedSecondStage[i] for i in model.ProductSizes]) - model.Capacity, 0.0)

model.EnforceCapacityLimitFirstStage = pyo.Constraint(rule=enforce_capacity_first_stage_rule)
model.EnforceCapacityLimitSecondStage = pyo.Constraint(rule=enforce_capacity_second_stage_rule)

# ensure that you can't generate inventory out of thin air.
def enforce_inventory_first_stage_rule(model, i):
    return (None, \
            sum([model.NumUnitsCutFirstStage[i,j] for j in model.ProductSizes if j <= i]) - \
            model.NumProducedFirstStage[i], \
            0.0)

def enforce_inventory_second_stage_rule(model, i):
    return (None, \
            sum([model.NumUnitsCutFirstStage[i,j] for j in model.ProductSizes if j <= i]) + \
            sum([model.NumUnitsCutSecondStage[i,j] for j in model.ProductSizes if j <= i]) \
            - model.NumProducedFirstStage[i] - model.NumProducedSecondStage[i], \
            0.0)

model.EnforceInventoryFirstStage = pyo.Constraint(model.ProductSizes, rule=enforce_inventory_first_stage_rule)
model.EnforceInventorySecondStage = pyo.Constraint(model.ProductSizes, rule=enforce_inventory_second_stage_rule)

# stage-specific cost computations.
def first_stage_cost_rule(model):
    production_costs = sum([model.SetupCosts[i] * model.ProduceSizeFirstStage[i] + \
                           model.UnitProductionCosts[i] * model.NumProducedFirstStage[i] \
                           for i in model.ProductSizes])
    cut_costs = sum([model.UnitReductionCost * model.NumUnitsCutFirstStage[i,j] \
                    for (i,j) in model.NumUnitsCutDomain if i != j])
    return (model.FirstStageCost - production_costs - cut_costs) == 0.0

model.ComputeFirstStageCost = pyo.Constraint(rule=first_stage_cost_rule)

def second_stage_cost_rule(model):
    production_costs = sum([model.SetupCosts[i] * model.ProduceSizeSecondStage[i] + \
                           model.UnitProductionCosts[i] * model.NumProducedSecondStage[i] \
                           for i in model.ProductSizes])
    cut_costs = sum([model.UnitReductionCost * model.NumUnitsCutSecondStage[i,j] \
                    for (i,j) in model.NumUnitsCutDomain if i != j])
    return (model.SecondStageCost - production_costs - cut_costs) == 0.0

model.ComputeSecondStageCost = pyo.Constraint(rule=second_stage_cost_rule)

#
# minimize: sum of StageCosts
#
# A active scenario objective equivalent to that generated by PySP is
# included here for informational purposes.
def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

