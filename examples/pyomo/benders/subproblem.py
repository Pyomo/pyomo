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

### Pyomo sub-problem for AMPL Bender's example, available
### from http://www.ampl.com/NEW/LOOP2/stoch2.mod

import pyomo.environ as pyo

model = pyo.AbstractModel(name="SubProblem")

# Declare suffixes
model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.lrc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.urc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# products
model.PROD = pyo.Set()

# number of weeks
model.T = pyo.Param(within=pyo.PositiveIntegers)


# derived set containing all valid week indices and subsets of interest.
def weeks_rule(model):
    return list(pyo.sequence(model.T()))


model.WEEKS = pyo.Set(initialize=weeks_rule, within=pyo.PositiveIntegers)


def two_plus_weeks_rule(model):
    return list(pyo.sequence(2, model.T()))


model.TWOPLUSWEEKS = pyo.Set(
    initialize=two_plus_weeks_rule, within=pyo.PositiveIntegers
)


def three_plus_weeks_rule(model):
    return list(pyo.sequence(3, model.T()))


model.THREEPLUSWEEKS = pyo.Set(
    initialize=three_plus_weeks_rule, within=pyo.PositiveIntegers
)

# tons per hour produced
model.rate = pyo.Param(model.PROD, within=pyo.PositiveReals)

# hours available in week
model.avail = pyo.Param(model.WEEKS, within=pyo.NonNegativeReals)

# limit on tons sold in week
model.market = pyo.Param(model.PROD, model.WEEKS, within=pyo.NonNegativeReals)

# cost per ton produced
model.prodcost = pyo.Param(model.PROD, within=pyo.NonNegativeReals)

# carrying cost/ton of inventory
model.invcost = pyo.Param(model.PROD, within=pyo.NonNegativeReals)

# projected revenue/ton
model.revenue = pyo.Param(model.PROD, model.WEEKS, within=pyo.NonNegativeReals)


# scenario probability
def unit_interval_validate(model, value):
    return (value >= 0.0) and (value <= 1.0)


model.prob = pyo.Param(validate=unit_interval_validate)

# inventory at end of first period.
model.inv1 = pyo.Param(model.PROD, within=pyo.NonNegativeReals, mutable=True)

# tons produced
model.Make = pyo.Var(model.PROD, model.TWOPLUSWEEKS, domain=pyo.NonNegativeReals)

# tons inventoried
model.Inv = pyo.Var(model.PROD, model.TWOPLUSWEEKS, domain=pyo.NonNegativeReals)


# tons sold
def sell_bounds(model, p, t):
    return (0, model.market[p, t])


model.Sell = pyo.Var(
    model.PROD, model.TWOPLUSWEEKS, within=pyo.NonNegativeReals, bounds=sell_bounds
)


def time_rule(model, t):
    return (
        sum([(1.0 / model.rate[p]) * model.Make[p, t] for p in model.PROD])
        - model.avail[t]
        <= 0.0
    )


model.Time = pyo.Constraint(model.TWOPLUSWEEKS, rule=time_rule)


def balance2_rule(model, p):
    return (model.Make[p, 2] + model.inv1[p]) - (
        model.Sell[p, 2] + model.Inv[p, 2]
    ) == 0.0


model.Balance2 = pyo.Constraint(model.PROD, rule=balance2_rule)


def balance_rule(model, p, t):
    return (model.Make[p, t] + model.Inv[p, t - 1]) - (
        model.Sell[p, t] + model.Inv[p, t]
    ) == 0.0


model.Balance = pyo.Constraint(model.PROD, model.THREEPLUSWEEKS, rule=balance_rule)


# the manual distribution of model.prob is ugly, but at the moment necessary; Pyomo
# expression simplification will be significantly improved in the near-term future.
def exp_stage2_profit_rule(model):
    return sum(
        [
            model.prob * model.revenue[p, t] * model.Sell[p, t]
            - model.prob * model.prodcost[p] * model.Make[p, t]
            - model.prob * model.invcost[p] * model.Inv[p, t]
            for p in model.PROD
            for t in model.TWOPLUSWEEKS
        ]
    )


model.Exp_Stage2_Profit = pyo.Objective(rule=exp_stage2_profit_rule, sense=pyo.maximize)
