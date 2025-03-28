#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

# Medium-term Purchasing Contracts problem from http://minlp.org/library/lib.php?lib=GDP
# This model maximizes profit in a short-term horizon in which various contracts
# are available for purchasing raw materials. The model decides inventory levels,
# amounts to purchase, amount sold, and flows through the process nodes while
# maximizing profit. The four different contracts available are:
# FIXED PRICE CONTRACT: buy as much as you want at constant price
# DISCOUNT CONTRACT: quantities below minimum amount cost RegPrice. Any additional quantity
# above min amount costs DiscoutPrice.
# BULK CONTRACT: If more than min amount is purchased, whole purchase is at discount price.
# FIXED DURATION CONTRACT: Depending on length of time contract is valid, there is a purchase
# price during that time and min quantity that must be purchased


# This version of the model is a literal transcription of what is in
# ShortTermContractCH.gms from the website. Some data is hardcoded into this model,
# most notably the process structure itself and the mass balance information.


def build_model():
    model = pyo.AbstractModel()

    # Constants (data that was hard-coded in GAMS model)
    AMOUNT_UB = 1000
    COST_UB = 1e4
    MAX_AMOUNT_FP = 1000
    MIN_AMOUNT_FD_1MONTH = 0

    RandomConst_Line264 = 0.17
    RandomConst_Line265 = 0.83

    ###################
    # Sets
    ###################

    # T
    # t in GAMS
    model.TimePeriods = pyo.Set(ordered=True)

    # Available length contracts
    # p in GAMS
    model.Contracts_Length = pyo.Set()

    # JP
    # final(j) in GAMS
    # Finished products
    model.Products = pyo.Set()

    # JM
    # rawmat(J) in GAMS
    # Set of Raw Materials-- raw materials, intermediate products, and final products partition J
    model.RawMaterials = pyo.Set()

    # C
    # c in GAMS
    model.Contracts = pyo.Set()

    # I
    # i in GAMS
    model.Processes = pyo.Set()

    # J
    # j in GAMS
    model.Streams = pyo.Set()

    ##################
    # Parameters
    ##################

    # Q_it
    # excap(i) in GAMS
    model.Capacity = pyo.Param(model.Processes)

    # u_ijt
    # cov(i) in GAMS
    model.ProcessConstants = pyo.Param(model.Processes)

    # a_jt^U and d_jt^U
    # spdm(j,t) in GAMS
    model.SupplyAndDemandUBs = pyo.Param(model.Streams, model.TimePeriods, default=0)

    # d_jt^L
    # lbdm(j, t) in GAMS
    model.DemandLB = pyo.Param(model.Streams, model.TimePeriods, default=0)

    # delta_it
    # delta(i, t) in GAMS
    # operating cost of process i at time t
    model.OperatingCosts = pyo.Param(model.Processes, model.TimePeriods)

    # prices of raw materials under FP contract and selling prices of products
    # pf(j, t) in GAMS
    # omega_jt and pf_jt
    model.Prices = pyo.Param(model.Streams, model.TimePeriods, default=0)

    # Price for quantities less than min amount under discount contract
    # pd1(j, t) in GAMS
    model.RegPrice_Discount = pyo.Param(model.Streams, model.TimePeriods)
    # Discounted price for the quantity purchased exceeding the min amount
    # pd2(j,t0 in GAMS
    model.DiscountPrice_Discount = pyo.Param(model.Streams, model.TimePeriods)

    # Price for quantities below min amount
    # pb1(j,t) in GAMS
    model.RegPrice_Bulk = pyo.Param(model.Streams, model.TimePeriods)
    # Price for quantities above min amount
    # pb2(j, t) in GAMS
    model.DiscountPrice_Bulk = pyo.Param(model.Streams, model.TimePeriods)

    # prices with length contract
    # pl(j, p, t) in GAMS
    model.Prices_Length = pyo.Param(
        model.Streams, model.Contracts_Length, model.TimePeriods, default=0
    )

    # sigmad_jt
    # sigmad(j, t) in GAMS
    # Minimum quantity of chemical j that must be bought before receiving a Discount under discount contract
    model.MinAmount_Discount = pyo.Param(model.Streams, model.TimePeriods, default=0)

    # min quantity to receive discount under bulk contract
    # sigmab(j, t) in GAMS
    model.MinAmount_Bulk = pyo.Param(model.Streams, model.TimePeriods, default=0)

    # min quantity to receive discount under length contract
    # sigmal(j, p) in GAMS
    model.MinAmount_Length = pyo.Param(model.Streams, model.Contracts_Length, default=0)

    # main products of process i
    # These are 1 (true) if stream j is the main product of process i, false otherwise.
    # jm(j, i) in GAMS
    model.MainProducts = pyo.Param(model.Streams, model.Processes, default=0)

    # theta_jt
    # psf(j, t) in GAMS
    # Shortfall penalty of product j at time t
    model.ShortfallPenalty = pyo.Param(model.Products, model.TimePeriods)

    # shortfall upper bound
    # sfub(j, t) in GAMS
    model.ShortfallUB = pyo.Param(model.Products, model.TimePeriods, default=0)

    # epsilon_jt
    # cinv(j, t) in GAMS
    # inventory cost of material j at time t
    model.InventoryCost = pyo.Param(model.Streams, model.TimePeriods)

    # invub(j, t) in GAMS
    # inventory upper bound
    model.InventoryLevelUB = pyo.Param(model.Streams, model.TimePeriods, default=0)

    ## UPPER BOUNDS HARDCODED INTO GAMS MODEL

    # All of these upper bounds are hardcoded. So I am just leaving them that way.
    # This means they all have to be the same as each other right now.
    def getAmountUBs(model, j, t):
        return AMOUNT_UB

    def getCostUBs(model, j, t):
        return COST_UB

    model.AmountPurchasedUB_FP = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )
    model.AmountPurchasedUB_Discount = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )
    model.AmountPurchasedBelowMinUB_Discount = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )
    model.AmountPurchasedAboveMinUB_Discount = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )
    model.AmountPurchasedUB_FD = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )
    model.AmountPurchasedUB_Bulk = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getAmountUBs
    )

    model.CostUB_FP = pyo.Param(model.Streams, model.TimePeriods, initialize=getCostUBs)
    model.CostUB_FD = pyo.Param(model.Streams, model.TimePeriods, initialize=getCostUBs)
    model.CostUB_Discount = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getCostUBs
    )
    model.CostUB_Bulk = pyo.Param(
        model.Streams, model.TimePeriods, initialize=getCostUBs
    )

    ####################
    # VARIABLES
    ####################

    # prof in GAMS
    # will be objective
    model.Profit = pyo.Var()

    # f(j, t) in GAMS
    # mass flow rates in tons per time interval t
    model.FlowRate = pyo.Var(
        model.Streams, model.TimePeriods, within=pyo.NonNegativeReals
    )

    # V_jt
    # inv(j, t) in GAMS
    # inventory level of chemical j at time period t
    def getInventoryBounds(model, i, j):
        return (0, model.InventoryLevelUB[i, j])

    model.InventoryLevel = pyo.Var(
        model.Streams, model.TimePeriods, bounds=getInventoryBounds
    )

    # SF_jt
    # sf(j, t) in GAMS
    # Shortfall of demand for chemical j at time period t
    def getShortfallBounds(model, i, j):
        return (0, model.ShortfallUB[i, j])

    model.Shortfall = pyo.Var(
        model.Products, model.TimePeriods, bounds=getShortfallBounds
    )

    # amounts purchased under different contracts

    # spf(j, t) in GAMS
    # Amount of raw material j bought under fixed price contract at time period t
    def get_FP_bounds(model, j, t):
        return (0, model.AmountPurchasedUB_FP[j, t])

    model.AmountPurchased_FP = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_FP_bounds
    )

    # spd(j, t) in GAMS
    def get_Discount_Total_bounds(model, j, t):
        return (0, model.AmountPurchasedUB_Discount[j, t])

    model.AmountPurchasedTotal_Discount = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_Discount_Total_bounds
    )

    # Amount purchased below min amount for discount under discount contract
    # spd1(j, t) in GAMS
    def get_Discount_BelowMin_bounds(model, j, t):
        return (0, model.AmountPurchasedBelowMinUB_Discount[j, t])

    model.AmountPurchasedBelowMin_Discount = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_Discount_BelowMin_bounds
    )

    # spd2(j, t) in GAMS
    # Amount purchased above min amount for discount under discount contract
    def get_Discount_AboveMin_bounds(model, j, t):
        return (0, model.AmountPurchasedBelowMinUB_Discount[j, t])

    model.AmountPurchasedAboveMin_Discount = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_Discount_AboveMin_bounds
    )

    # Amount purchased under bulk contract
    # spb(j, t) in GAMS
    def get_bulk_bounds(model, j, t):
        return (0, model.AmountPurchasedUB_Bulk[j, t])

    model.AmountPurchased_Bulk = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_bulk_bounds
    )

    # spl(j, t) in GAMS
    # Amount purchased under Fixed Duration contract
    def get_FD_bounds(model, j, t):
        return (0, model.AmountPurchasedUB_FD[j, t])

    model.AmountPurchased_FD = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_FD_bounds
    )

    # costs

    # costpl(j, t) in GAMS
    # cost of variable length contract
    def get_CostUBs_FD(model, j, t):
        return (0, model.CostUB_FD[j, t])

    model.Cost_FD = pyo.Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FD)

    # costpf(j, t) in GAMS
    # cost of fixed duration contract
    def get_CostUBs_FP(model, j, t):
        return (0, model.CostUB_FP[j, t])

    model.Cost_FP = pyo.Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FP)

    # costpd(j, t) in GAMS
    # cost of discount contract
    def get_CostUBs_Discount(model, j, t):
        return (0, model.CostUB_Discount[j, t])

    model.Cost_Discount = pyo.Var(
        model.Streams, model.TimePeriods, bounds=get_CostUBs_Discount
    )

    # costpb(j, t) in GAMS
    # cost of bulk contract
    def get_CostUBs_Bulk(model, j, t):
        return (0, model.CostUB_Bulk[j, t])

    model.Cost_Bulk = pyo.Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_Bulk)

    # binary variables

    model.BuyFPContract = pyo.RangeSet(0, 1)
    model.BuyDiscountContract = pyo.Set(
        initialize=('BelowMin', 'AboveMin', 'NotSelected')
    )
    model.BuyBulkContract = pyo.Set(initialize=('BelowMin', 'AboveMin', 'NotSelected'))
    model.BuyFDContract = pyo.Set(
        initialize=('1Month', '2Month', '3Month', 'NotSelected')
    )

    ################
    # CONSTRAINTS
    ################

    # Objective: maximize profit
    def profit_rule(model):
        salesIncome = sum(
            model.Prices[j, t] * model.FlowRate[j, t]
            for j in model.Products
            for t in model.TimePeriods
        )
        purchaseCost = (
            sum(
                model.Cost_FD[j, t]
                for j in model.RawMaterials
                for t in model.TimePeriods
            )
            + sum(
                model.Cost_Discount[j, t]
                for j in model.RawMaterials
                for t in model.TimePeriods
            )
            + sum(
                model.Cost_Bulk[j, t]
                for j in model.RawMaterials
                for t in model.TimePeriods
            )
            + sum(
                model.Cost_FP[j, t]
                for j in model.RawMaterials
                for t in model.TimePeriods
            )
        )
        productionCost = sum(
            model.OperatingCosts[i, t]
            * sum(
                model.FlowRate[j, t] for j in model.Streams if model.MainProducts[j, i]
            )
            for i in model.Processes
            for t in model.TimePeriods
        )
        shortfallCost = sum(
            model.Shortfall[j, t] * model.ShortfallPenalty[j, t]
            for j in model.Products
            for t in model.TimePeriods
        )
        inventoryCost = sum(
            model.InventoryCost[j, t] * model.InventoryLevel[j, t]
            for j in model.Products
            for t in model.TimePeriods
        )
        return (
            salesIncome - purchaseCost - productionCost - inventoryCost - shortfallCost
        )

    model.profit = pyo.Objective(rule=profit_rule, sense=pyo.maximize)

    # flow of raw materials is the total amount purchased (across all contracts)
    def raw_material_flow_rule(model, j, t):
        return (
            model.FlowRate[j, t]
            == model.AmountPurchased_FD[j, t]
            + model.AmountPurchased_FP[j, t]
            + model.AmountPurchased_Bulk[j, t]
            + model.AmountPurchasedTotal_Discount[j, t]
        )

    model.raw_material_flow = pyo.Constraint(
        model.RawMaterials, model.TimePeriods, rule=raw_material_flow_rule
    )

    def discount_amount_total_rule(model, j, t):
        return (
            model.AmountPurchasedTotal_Discount[j, t]
            == model.AmountPurchasedBelowMin_Discount[j, t]
            + model.AmountPurchasedAboveMin_Discount[j, t]
        )

    model.discount_amount_total_rule = pyo.Constraint(
        model.RawMaterials, model.TimePeriods, rule=discount_amount_total_rule
    )

    # mass balance equations for each node
    # these are specific to the process network in this example.
    def mass_balance_rule1(model, t):
        return model.FlowRate[1, t] == model.FlowRate[2, t] + model.FlowRate[3, t]

    model.mass_balance1 = pyo.Constraint(model.TimePeriods, rule=mass_balance_rule1)

    def mass_balance_rule2(model, t):
        return model.FlowRate[5, t] == model.FlowRate[4, t] + model.FlowRate[8, t]

    model.mass_balance2 = pyo.Constraint(model.TimePeriods, rule=mass_balance_rule2)

    def mass_balance_rule3(model, t):
        return model.FlowRate[6, t] == model.FlowRate[7, t]

    model.mass_balance3 = pyo.Constraint(model.TimePeriods, rule=mass_balance_rule3)

    def mass_balance_rule4(model, t):
        return model.FlowRate[3, t] == 10 * model.FlowRate[5, t]

    model.mass_balance4 = pyo.Constraint(model.TimePeriods, rule=mass_balance_rule4)

    # process input/output constraints
    # these are also totally specific to the process network
    def process_balance_rule1(model, t):
        return model.FlowRate[9, t] == model.ProcessConstants[1] * model.FlowRate[2, t]

    model.process_balance1 = pyo.Constraint(
        model.TimePeriods, rule=process_balance_rule1
    )

    def process_balance_rule2(model, t):
        return model.FlowRate[10, t] == model.ProcessConstants[2] * (
            model.FlowRate[5, t] + model.FlowRate[3, t]
        )

    model.process_balance2 = pyo.Constraint(
        model.TimePeriods, rule=process_balance_rule2
    )

    def process_balance_rule3(model, t):
        return (
            model.FlowRate[8, t]
            == RandomConst_Line264 * model.ProcessConstants[3] * model.FlowRate[7, t]
        )

    model.process_balance3 = pyo.Constraint(
        model.TimePeriods, rule=process_balance_rule3
    )

    def process_balance_rule4(model, t):
        return (
            model.FlowRate[11, t]
            == RandomConst_Line265 * model.ProcessConstants[3] * model.FlowRate[7, t]
        )

    model.process_balance4 = pyo.Constraint(
        model.TimePeriods, rule=process_balance_rule4
    )

    # process capacity constraints
    # these are hardcoded based on the three processes and the process flow structure
    def process_capacity_rule1(model, t):
        return model.FlowRate[9, t] <= model.Capacity[1]

    model.process_capacity1 = pyo.Constraint(
        model.TimePeriods, rule=process_capacity_rule1
    )

    def process_capacity_rule2(model, t):
        return model.FlowRate[10, t] <= model.Capacity[2]

    model.process_capacity2 = pyo.Constraint(
        model.TimePeriods, rule=process_capacity_rule2
    )

    def process_capacity_rule3(model, t):
        return model.FlowRate[11, t] + model.FlowRate[8, t] <= model.Capacity[3]

    model.process_capacity3 = pyo.Constraint(
        model.TimePeriods, rule=process_capacity_rule3
    )

    # Inventory balance of final products
    # again, these are hardcoded.

    def inventory_balance1(model, t):
        prev = 0 if t == min(model.TimePeriods) else model.InventoryLevel[12, t - 1]
        return (
            prev + model.FlowRate[9, t]
            == model.FlowRate[12, t] + model.InventoryLevel[12, t]
        )

    model.inventory_balance1 = pyo.Constraint(
        model.TimePeriods, rule=inventory_balance1
    )

    def inventory_balance_rule2(model, t):
        if t != 1:
            return pyo.Constraint.Skip
        return (
            model.FlowRate[10, t] + model.FlowRate[11, t]
            == model.InventoryLevel[13, t] + model.FlowRate[13, t]
        )

    model.inventory_balance2 = pyo.Constraint(
        model.TimePeriods, rule=inventory_balance_rule2
    )

    def inventory_balance_rule3(model, t):
        if t <= 1:
            return pyo.Constraint.Skip
        return (
            model.InventoryLevel[13, t - 1]
            + model.FlowRate[10, t]
            + model.FlowRate[11, t]
            == model.InventoryLevel[13, t] + model.FlowRate[13, t]
        )

    model.inventory_balance3 = pyo.Constraint(
        model.TimePeriods, rule=inventory_balance_rule3
    )

    # Max capacities of inventories
    def inventory_capacity_rule(model, j, t):
        return model.InventoryLevel[j, t] <= model.InventoryLevelUB[j, t]

    model.inventory_capacity_rule = pyo.Constraint(
        model.Products, model.TimePeriods, rule=inventory_capacity_rule
    )

    # Shortfall calculation
    def shortfall_rule(model, j, t):
        return (
            model.Shortfall[j, t]
            == model.SupplyAndDemandUBs[j, t] - model.FlowRate[j, t]
        )

    model.shortfall = pyo.Constraint(
        model.Products, model.TimePeriods, rule=shortfall_rule
    )

    # maximum shortfall allowed
    def shortfall_max_rule(model, j, t):
        return model.Shortfall[j, t] <= model.ShortfallUB[j, t]

    model.shortfall_max = pyo.Constraint(
        model.Products, model.TimePeriods, rule=shortfall_max_rule
    )

    # maximum capacities of suppliers
    def supplier_capacity_rule(model, j, t):
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j, t]

    model.supplier_capacity = pyo.Constraint(
        model.RawMaterials, model.TimePeriods, rule=supplier_capacity_rule
    )

    # demand upper bound
    def demand_UB_rule(model, j, t):
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j, t]

    model.demand_UB = pyo.Constraint(
        model.Products, model.TimePeriods, rule=demand_UB_rule
    )

    # demand lower bound
    def demand_LB_rule(model, j, t):
        return model.FlowRate[j, t] >= model.DemandLB[j, t]

    model.demand_LB = pyo.Constraint(
        model.Products, model.TimePeriods, rule=demand_LB_rule
    )

    # FIXED PRICE CONTRACT

    # Disjunction for Fixed Price contract buying options
    def FP_contract_disjunct_rule(disjunct, j, t, buy):
        model = disjunct.model()
        if buy:
            disjunct.c = pyo.Constraint(
                expr=model.AmountPurchased_FP[j, t] <= MAX_AMOUNT_FP
            )
        else:
            disjunct.c = pyo.Constraint(expr=model.AmountPurchased_FP[j, t] == 0)

    model.FP_contract_disjunct = Disjunct(
        model.RawMaterials,
        model.TimePeriods,
        model.BuyFPContract,
        rule=FP_contract_disjunct_rule,
    )

    # Fixed price disjunction
    def FP_contract_rule(model, j, t):
        return [model.FP_contract_disjunct[j, t, buy] for buy in model.BuyFPContract]

    model.FP_disjunction = Disjunction(
        model.RawMaterials, model.TimePeriods, rule=FP_contract_rule
    )

    # cost constraint for fixed price contract (independent constraint)
    def FP_contract_cost_rule(model, j, t):
        return (
            model.Cost_FP[j, t] == model.AmountPurchased_FP[j, t] * model.Prices[j, t]
        )

    model.FP_contract_cost = pyo.Constraint(
        model.RawMaterials, model.TimePeriods, rule=FP_contract_cost_rule
    )

    # DISCOUNT CONTRACT

    # Disjunction for Discount contract
    def discount_contract_disjunct_rule(disjunct, j, t, buy):
        model = disjunct.model()
        if buy == 'BelowMin':
            disjunct.belowMin = pyo.Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j, t]
                <= model.MinAmount_Discount[j, t]
            )
            disjunct.aboveMin = pyo.Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j, t] == 0
            )
        elif buy == 'AboveMin':
            disjunct.belowMin = pyo.Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j, t]
                == model.MinAmount_Discount[j, t]
            )
            disjunct.aboveMin = pyo.Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j, t] >= 0
            )
        elif buy == 'NotSelected':
            disjunct.belowMin = pyo.Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j, t] == 0
            )
            disjunct.aboveMin = pyo.Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j, t] == 0
            )
        else:
            raise RuntimeError("Unrecognized choice for discount contract: %s" % buy)

    model.discount_contract_disjunct = Disjunct(
        model.RawMaterials,
        model.TimePeriods,
        model.BuyDiscountContract,
        rule=discount_contract_disjunct_rule,
    )

    # Discount contract disjunction
    def discount_contract_rule(model, j, t):
        return [
            model.discount_contract_disjunct[j, t, buy]
            for buy in model.BuyDiscountContract
        ]

    model.discount_contract = Disjunction(
        model.RawMaterials, model.TimePeriods, rule=discount_contract_rule
    )

    # cost constraint for discount contract (independent constraint)
    def discount_cost_rule(model, j, t):
        return (
            model.Cost_Discount[j, t]
            == model.RegPrice_Discount[j, t]
            * model.AmountPurchasedBelowMin_Discount[j, t]
            + model.DiscountPrice_Discount[j, t]
            * model.AmountPurchasedAboveMin_Discount[j, t]
        )

    model.discount_cost = pyo.Constraint(
        model.RawMaterials, model.TimePeriods, rule=discount_cost_rule
    )

    # BULK CONTRACT

    # Bulk contract buying options disjunct
    def bulk_contract_disjunct_rule(disjunct, j, t, buy):
        model = disjunct.model()
        if buy == 'BelowMin':
            disjunct.amount = pyo.Constraint(
                expr=model.AmountPurchased_Bulk[j, t] <= model.MinAmount_Bulk[j, t]
            )
            disjunct.price = pyo.Constraint(
                expr=model.Cost_Bulk[j, t]
                == model.RegPrice_Bulk[j, t] * model.AmountPurchased_Bulk[j, t]
            )
        elif buy == 'AboveMin':
            disjunct.amount = pyo.Constraint(
                expr=model.AmountPurchased_Bulk[j, t] >= model.MinAmount_Bulk[j, t]
            )
            disjunct.price = pyo.Constraint(
                expr=model.Cost_Bulk[j, t]
                == model.DiscountPrice_Bulk[j, t] * model.AmountPurchased_Bulk[j, t]
            )
        elif buy == 'NotSelected':
            disjunct.amount = pyo.Constraint(expr=model.AmountPurchased_Bulk[j, t] == 0)
            disjunct.price = pyo.Constraint(expr=model.Cost_Bulk[j, t] == 0)
        else:
            raise RuntimeError("Unrecognized choice for bulk contract: %s" % buy)

    model.bulk_contract_disjunct = Disjunct(
        model.RawMaterials,
        model.TimePeriods,
        model.BuyBulkContract,
        rule=bulk_contract_disjunct_rule,
    )

    # Bulk contract disjunction
    def bulk_contract_rule(model, j, t):
        return [
            model.bulk_contract_disjunct[j, t, buy] for buy in model.BuyBulkContract
        ]

    model.bulk_contract = Disjunction(
        model.RawMaterials, model.TimePeriods, rule=bulk_contract_rule
    )

    # FIXED DURATION CONTRACT

    def FD_1mo_contract(disjunct, j, t):
        model = disjunct.model()
        disjunct.amount1 = pyo.Constraint(
            expr=model.AmountPurchased_FD[j, t] >= MIN_AMOUNT_FD_1MONTH
        )
        disjunct.price1 = pyo.Constraint(
            expr=model.Cost_FD[j, t]
            == model.Prices_Length[j, 1, t] * model.AmountPurchased_FD[j, t]
        )

    model.FD_1mo_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_1mo_contract
    )

    def FD_2mo_contract(disjunct, j, t):
        model = disjunct.model()
        disjunct.amount1 = pyo.Constraint(
            expr=model.AmountPurchased_FD[j, t] >= model.MinAmount_Length[j, 2]
        )
        disjunct.price1 = pyo.Constraint(
            expr=model.Cost_FD[j, t]
            == model.Prices_Length[j, 2, t] * model.AmountPurchased_FD[j, t]
        )
        # only enforce these if we aren't in the last time period
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = pyo.Constraint(
                expr=model.AmountPurchased_FD[j, t + 1] >= model.MinAmount_Length[j, 2]
            )
            disjunct.price2 = pyo.Constraint(
                expr=model.Cost_FD[j, t + 1]
                == model.Prices_Length[j, 2, t] * model.AmountPurchased_FD[j, t + 1]
            )

    model.FD_2mo_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_2mo_contract
    )

    def FD_3mo_contract(disjunct, j, t):
        model = disjunct.model()
        # NOTE: I think there is a mistake in the GAMS file in line 327.
        # they use the bulk minamount rather than the length one.
        # I am doing the same here for validation purposes.
        disjunct.amount1 = pyo.Constraint(
            expr=model.AmountPurchased_FD[j, t] >= model.MinAmount_Bulk[j, 3]
        )
        disjunct.cost1 = pyo.Constraint(
            expr=model.Cost_FD[j, t]
            == model.Prices_Length[j, 3, t] * model.AmountPurchased_FD[j, t]
        )
        # check we aren't in one of the last two time periods
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = pyo.Constraint(
                expr=model.AmountPurchased_FD[j, t + 1] >= model.MinAmount_Length[j, 3]
            )
            disjunct.cost2 = pyo.Constraint(
                expr=model.Cost_FD[j, t + 1]
                == model.Prices_Length[j, 3, t] * model.AmountPurchased_FD[j, t + 1]
            )
        if t < model.TimePeriods[-2]:
            disjunct.amount3 = pyo.Constraint(
                expr=model.AmountPurchased_FD[j, t + 2] >= model.MinAmount_Length[j, 3]
            )
            disjunct.cost3 = pyo.Constraint(
                expr=model.Cost_FD[j, t + 2]
                == model.Prices_Length[j, 3, t] * model.AmountPurchased_FD[j, t + 2]
            )

    model.FD_3mo_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_3mo_contract
    )

    def FD_no_contract(disjunct, j, t):
        model = disjunct.model()
        disjunct.amount1 = pyo.Constraint(expr=model.AmountPurchased_FD[j, t] == 0)
        disjunct.cost1 = pyo.Constraint(expr=model.Cost_FD[j, t] == 0)
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = pyo.Constraint(
                expr=model.AmountPurchased_FD[j, t + 1] == 0
            )
            disjunct.cost2 = pyo.Constraint(expr=model.Cost_FD[j, t + 1] == 0)
        if t < model.TimePeriods[-2]:
            disjunct.amount3 = pyo.Constraint(
                expr=model.AmountPurchased_FD[j, t + 2] == 0
            )
            disjunct.cost3 = pyo.Constraint(expr=model.Cost_FD[j, t + 2] == 0)

    model.FD_no_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_no_contract
    )

    def FD_contract(model, j, t):
        return [
            model.FD_1mo_contract[j, t],
            model.FD_2mo_contract[j, t],
            model.FD_3mo_contract[j, t],
            model.FD_no_contract[j, t],
        ]

    model.FD_contract = Disjunction(
        model.RawMaterials, model.TimePeriods, rule=FD_contract
    )

    return model


if __name__ == "__main__":
    m = build_model().create_instance('medTermPurchasing_Literal_Hull.dat')
    pyo.TransformationFactory('gdp.bigm').apply_to(m)
    pyo.SolverFactory('gams').solve(
        m, solver='baron', tee=True, add_options=['option optcr=1e-6;']
    )
    m.profit.display()
