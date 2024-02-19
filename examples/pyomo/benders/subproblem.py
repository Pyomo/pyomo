#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

### Pyomo sub-problem for AMPL Bender's example, available
### from http://www.ampl.com/NEW/LOOP2/stoch2.mod

from pyomo.core import *

model = AbstractModel(name="SubProblem")

# Declare suffixes
model.rc = Suffix(direction=Suffix.IMPORT)
model.lrc = Suffix(direction=Suffix.IMPORT)
model.urc = Suffix(direction=Suffix.IMPORT)
model.dual = Suffix(direction=Suffix.IMPORT)

# products
model.PROD = Set()

# number of weeks
model.T = Param(within=PositiveIntegers)


# derived set containing all valid week indices and subsets of interest.
def weeks_rule(model):
    return list(sequence(model.T()))


model.WEEKS = Set(initialize=weeks_rule, within=PositiveIntegers)


def two_plus_weeks_rule(model):
    return list(sequence(2, model.T()))


model.TWOPLUSWEEKS = Set(initialize=two_plus_weeks_rule, within=PositiveIntegers)


def three_plus_weeks_rule(model):
    return list(sequence(3, model.T()))


model.THREEPLUSWEEKS = Set(initialize=three_plus_weeks_rule, within=PositiveIntegers)

# tons per hour produced
model.rate = Param(model.PROD, within=PositiveReals)

# hours available in week
model.avail = Param(model.WEEKS, within=NonNegativeReals)

# limit on tons sold in week
model.market = Param(model.PROD, model.WEEKS, within=NonNegativeReals)

# cost per ton produced
model.prodcost = Param(model.PROD, within=NonNegativeReals)

# carrying cost/ton of inventory
model.invcost = Param(model.PROD, within=NonNegativeReals)

# projected revenue/ton
model.revenue = Param(model.PROD, model.WEEKS, within=NonNegativeReals)


# scenario probability
def unit_interval_validate(model, value):
    return (value >= 0.0) and (value <= 1.0)


model.prob = Param(validate=unit_interval_validate)

# inventory at end of first period.
model.inv1 = Param(model.PROD, within=NonNegativeReals, mutable=True)

# tons produced
model.Make = Var(model.PROD, model.TWOPLUSWEEKS, domain=NonNegativeReals)

# tons inventoried
model.Inv = Var(model.PROD, model.TWOPLUSWEEKS, domain=NonNegativeReals)


# tons sold
def sell_bounds(model, p, t):
    return (0, model.market[p, t])


model.Sell = Var(
    model.PROD, model.TWOPLUSWEEKS, within=NonNegativeReals, bounds=sell_bounds
)


def time_rule(model, t):
    return (
        sum([(1.0 / model.rate[p]) * model.Make[p, t] for p in model.PROD])
        - model.avail[t]
        <= 0.0
    )


model.Time = Constraint(model.TWOPLUSWEEKS, rule=time_rule)


def balance2_rule(model, p):
    return (model.Make[p, 2] + model.inv1[p]) - (
        model.Sell[p, 2] + model.Inv[p, 2]
    ) == 0.0


model.Balance2 = Constraint(model.PROD, rule=balance2_rule)


def balance_rule(model, p, t):
    return (model.Make[p, t] + model.Inv[p, t - 1]) - (
        model.Sell[p, t] + model.Inv[p, t]
    ) == 0.0


model.Balance = Constraint(model.PROD, model.THREEPLUSWEEKS, rule=balance_rule)


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


model.Exp_Stage2_Profit = Objective(rule=exp_stage2_profit_rule, sense=maximize)
