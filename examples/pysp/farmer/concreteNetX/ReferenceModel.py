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
# Farmer: NetworkX to create tree & gratitiously using indexed cost expressions

import networkx 

import pyomo.environ as pyo

model = pyo.ConcreteModel()

#
# Parameters
#

model.CROPS = pyo.Set(initialize=['WHEAT','CORN','SUGAR_BEETS'])

model.TOTAL_ACREAGE = 500.0

model.PriceQuota = {'WHEAT':100000.0,'CORN':100000.0,'SUGAR_BEETS':6000.0}

model.SubQuotaSellingPrice = {'WHEAT':170.0,'CORN':150.0,'SUGAR_BEETS':36.0}

model.SuperQuotaSellingPrice = {'WHEAT':0.0,'CORN':0.0,'SUGAR_BEETS':10.0}

model.CattleFeedRequirement = {'WHEAT':200.0,'CORN':240.0,'SUGAR_BEETS':0.0}

model.PurchasePrice = {'WHEAT':238.0,'CORN':210.0,'SUGAR_BEETS':100000.0}

model.PlantingCostPerAcre = {'WHEAT':150.0,'CORN':230.0,'SUGAR_BEETS':260.0}

model.Yield = pyo.Param(model.CROPS,
                        within=pyo.NonNegativeReals,
                        initialize=0.0,
                        mutable=True)

#
# Variables
#

model.DevotedAcreage = pyo.Var(model.CROPS, bounds=(0.0, model.TOTAL_ACREAGE))

model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

#
# Constraints
#

def ConstrainTotalAcreage_rule(model):
    return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE

model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

def EnforceCattleFeedRequirement_rule(model, i):
    return model.CattleFeedRequirement[i] <= (model.Yield[i] * model.DevotedAcreage[i]) + model.QuantityPurchased[i] - model.QuantitySubQuotaSold[i] - model.QuantitySuperQuotaSold[i]

model.EnforceCattleFeedRequirement = pyo.Constraint(model.CROPS, rule=EnforceCattleFeedRequirement_rule)

def LimitAmountSold_rule(model, i):
    return model.QuantitySubQuotaSold[i] + model.QuantitySuperQuotaSold[i] - (model.Yield[i] * model.DevotedAcreage[i]) <= 0.0

model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

def EnforceQuotas_rule(model, i):
    return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])

model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

# Stage-specific cost computations;

def ComputeFirstStageCost_rule(model):
    return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)
model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
    expr -= pyo.sum_product(model.SubQuotaSellingPrice, model.QuantitySubQuotaSold)
    expr -= pyo.sum_product(model.SuperQuotaSellingPrice, model.QuantitySuperQuotaSold)
    return expr
model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

# Gratitiously using an indexed cost Expression
# (you really could/should use the two you already have)
StageSet = pyo.RangeSet(2)
def cost_rule(m, stage):
    # Just assign the expressions to the right stage
    if stage == 1:
        return model.FirstStageCost
    if stage == 2:
        return model.SecondStageCost
model.CostExpressions = pyo.Expression(StageSet, rule=cost_rule)

def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

#
# Stochastic Data
#
Yield = {}
Yield['BelowAverageScenario'] = \
    {'WHEAT':2.0,'CORN':2.4,'SUGAR_BEETS':16.0}
Yield['AverageScenario'] = \
    {'WHEAT':2.5,'CORN':3.0,'SUGAR_BEETS':20.0}
Yield['AboveAverageScenario'] = \
    {'WHEAT':3.0,'CORN':3.6,'SUGAR_BEETS':24.0}

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.Yield.store_values(Yield[scenario_name])

    return instance

def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
    g = networkx.DiGraph()

    ce1 = "CostExpressions[1]"
    g.add_node("Root",
               cost = ce1,
               variables = ["DevotedAcreage[*]"],
               derived_variables = [])

    ce2 = "CostExpressions[2]"
    g.add_node("BelowAverageScenario",
               cost = ce2,
               variables = ["QuantitySubQuotaSold[*]",
                            "QuantitySuperQuotaSold[*]",
                            "QuantityPurchased[*]"],
               derived_variables = [])
    g.add_edge("Root", "BelowAverageScenario", weight=0.3333)

    g.add_node("AverageScenario",
               cost = ce2,
               variables = ["QuantitySubQuotaSold[*]",
                            "QuantitySuperQuotaSold[*]",
                            "QuantityPurchased[*]"],
               derived_variables = [])
    g.add_edge("Root", "AverageScenario", weight=0.3333)

    g.add_node("AboveAverageScenario",
               cost = ce2,
               variables = ["QuantitySubQuotaSold[*]",
                            "QuantitySuperQuotaSold[*]",
                            "QuantityPurchased[*]"],
               derived_variables = [])
    g.add_edge("Root", "AboveAverageScenario", weight=0.3334)

    return g

