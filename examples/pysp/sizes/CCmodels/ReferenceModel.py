#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# revised to have a chance constraint for demand as needed by runef with the CC option
#
# This is the two-period version of the SIZES optimization model
# derived from the three stage model in:
#A. L{\o}kketangen and D. L. Woodruff,
#"Progressive Hedging and Tabu Search Applied to Mixed Integer (0,1) Multistage Stochastic Programming",
#Journal of Heuristics, 1996, Vol 2, Pages 111-128.

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
model.DemandsFirstStage = Param(model.ProductSizes, within=NonNegativeIntegers)
model.DemandsSecondStage = Param(model.ProductSizes, within=NonNegativeIntegers)

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

# a derived set to constrain the NumUnitsCut variable domain.
# TBD: the (i,j) with i >= j set should be a generic utility.
def num_units_cut_domain_rule(model):
   ans = set()
   for i in range(1,model.NumSizes()+1):
      for j in range(1, i+1):
         ans.add((i,j))    
   return ans

model.NumUnitsCutDomain = Set(initialize=num_units_cut_domain_rule, dimen=2)

#
# Variables
#

# are any products at size i produced?
model.ProduceSizeFirstStage = Var(model.ProductSizes, domain=Boolean)
model.ProduceSizeSecondStage = Var(model.ProductSizes, domain=Boolean)

# NOTE: The following (num-produced and num-cut) variables are implicitly integer
#       under the normal cost objective, but with the PH cost objective, this isn't
#       the case.

# the number of units at each size produced.
model.NumProducedFirstStage = Var(model.ProductSizes, domain=NonNegativeIntegers, bounds=(0.0, model.Capacity))
model.NumProducedSecondStage = Var(model.ProductSizes, domain=NonNegativeIntegers, bounds=(0.0, model.Capacity))

# the number of units of size i cut (down) to meet demands for units of size j.
model.NumUnitsCutFirstStage = Var(model.NumUnitsCutDomain, domain=NonNegativeIntegers, bounds=(0.0, model.Capacity))
model.NumUnitsCutSecondStage = Var(model.NumUnitsCutDomain, domain=NonNegativeIntegers, bounds=(0.0, model.Capacity))

# stage-specific cost variables for use in the pysp scenario tree / analysis.
model.FirstStageCost = Var(domain=NonNegativeReals)
model.SecondStageCost = Var(domain=NonNegativeReals)

#
# Constraints
#

# ensure that demand is satisfied in the first stage, accounting for cut-downs.
def demand_satisfied_first_stage_rule(model, i):
   return (0.0, sum([model.NumUnitsCutFirstStage[j,i] for j in model.ProductSizes if j >= i]) - model.DemandsFirstStage[i], None)
model.DemandSatisfiedFirstStage = Constraint(model.ProductSizes, rule=demand_satisfied_first_stage_rule)

###### CC: chance constraint #######
## instead of requiring demand always be met in the second stage,
## we set up a variable that indicates whether it is met
##def demand_satisfied_second_stage_rule(i, model):
##   return (0.0, sum([model.NumUnitsCutSecondStage[j,i] for j in model.ProductSizes if j >= i]) - model.DemandsSecondStage[i], None)    
##model.DemandSatisfiedSecondStage = Constraint(model.ProductSizes, rule=demand_satisfied_second_stage_rule)

model.dIndicator = Var(domain=Boolean)# indicate that all demans is met (for scenario)
model.dforsize = Var(model.ProductSizes, domain=Boolean)  # indicate that demand is met for size
model.lambdaMult = Param(initialize=0.0, mutable=True)

# The production capacity per time stage serves as a simple upper bound for "M".
def establish_dforsize_rule(model, i):
   return model.dforsize[i]-1.0 <= (sum([model.NumUnitsCutSecondStage[j,i] for j in model.ProductSizes if j >= i]) - model.DemandsSecondStage[i]) / model.Capacity

model.establish_dforsize = Constraint(model.ProductSizes, rule=establish_dforsize_rule)

# it "wants to be a one" so don't let it unless it should be
def establish_dIndicator_rule(model):
   return model.dIndicator * model.NumSizes <= summation(model.dforsize)

model.establish_dIndicator = Constraint(rule=establish_dIndicator_rule)

# if the chance constraint is not desired, then runef can give the fully admissible solutions
# by using the following AllDemandMet constraint
# (this should be commented-out to get a chance constraint)
##def AllDemandMet_rule(model):
##   return model.dIndicator >= 1.0
##model.AllDemandMet = Constraint(rule=AllDemandMet_rule)

# ensure that you don't produce any units if the decision has been made to disable producion.
def enforce_production_first_stage_rule(model, i):
   # The production capacity per time stage serves as a simple upper bound for "M".
   return (None, model.NumProducedFirstStage[i] - model.Capacity * model.ProduceSizeFirstStage[i], 0.0)

def enforce_production_second_stage_rule(model, i):
   # The production capacity per time stage serves as a simple upper bound for "M".   
   return (None, model.NumProducedSecondStage[i] - model.Capacity * model.ProduceSizeSecondStage[i], 0.0)

model.EnforceProductionBinaryFirstStage = Constraint(model.ProductSizes, rule=enforce_production_first_stage_rule)
model.EnforceProductionBinarySecondStage = Constraint(model.ProductSizes, rule=enforce_production_second_stage_rule)

# ensure that the production capacity is not exceeded for each time stage.
def enforce_capacity_first_stage_rule(model):
   return (None, sum([model.NumProducedFirstStage[i] for i in model.ProductSizes]) - model.Capacity, 0.0)

def enforce_capacity_second_stage_rule(model):
   return (None, sum([model.NumProducedSecondStage[i] for i in model.ProductSizes]) - model.Capacity, 0.0)    

model.EnforceCapacityLimitFirstStage = Constraint(rule=enforce_capacity_first_stage_rule)
model.EnforceCapacityLimitSecondStage = Constraint(rule=enforce_capacity_second_stage_rule)

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

model.EnforceInventoryFirstStage = Constraint(model.ProductSizes, rule=enforce_inventory_first_stage_rule)
model.EnforceInventorySecondStage = Constraint(model.ProductSizes, rule=enforce_inventory_second_stage_rule)

# stage-specific cost computations.
def first_stage_cost_rule(model):
   production_costs = sum([model.SetupCosts[i] * model.ProduceSizeFirstStage[i] + \
                          model.UnitProductionCosts[i] * model.NumProducedFirstStage[i] \
                          for i in model.ProductSizes])
   cut_costs = sum([model.UnitReductionCost * model.NumUnitsCutFirstStage[i,j] \
                   for (i,j) in model.NumUnitsCutDomain if i != j])
   return (model.FirstStageCost - production_costs - cut_costs) == 0.0

model.ComputeFirstStageCost = Constraint(rule=first_stage_cost_rule)

def second_stage_cost_rule(model):
   production_costs = sum([model.SetupCosts[i] * model.ProduceSizeSecondStage[i] + \
                          model.UnitProductionCosts[i] * model.NumProducedSecondStage[i] \
                          for i in model.ProductSizes])
   cut_costs = sum([model.UnitReductionCost * model.NumUnitsCutSecondStage[i,j] \
                   for (i,j) in model.NumUnitsCutDomain if i != j])
   return (model.SecondStageCost - production_costs - cut_costs) == 0.0    

model.ComputeSecondStageCost = Constraint(rule=second_stage_cost_rule)

#
# PySP Auto-generated Objective
#
# minimize: sum of StageCostVariables
#
# A active scenario objective equivalent to that generated by PySP is
# included here for informational purposes.
def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

