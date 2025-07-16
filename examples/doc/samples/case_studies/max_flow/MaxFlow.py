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

model = pyo.AbstractModel()

model.nodes = pyo.Set()
model.arcs = pyo.Set(within=model.nodes * model.nodes)
model.sources = pyo.Set(within=model.nodes)
model.sinks = pyo.Set(within=model.nodes)
model.upperBound = pyo.Param(model.arcs)
model.supply = pyo.Param(model.sources)
model.demand = pyo.Param(model.sinks)
model.amount = pyo.Var(model.arcs, within=pyo.NonNegativeReals)


def totalRule(model):
    expression = sum(model.amount[i, j] for (i, j) in model.arcs if j in model.sinks)
    return expression


model.maxFlow = pyo.Objective(rule=totalRule, sense=pyo.maximize)


def maxRule(model, arcIn, arcOut):
    constraint_equation = model.amount[arcIn, arcOut] <= model.upperBound[arcIn, arcOut]
    return constraint_equation


model.loadOnArc = pyo.Constraint(model.arcs, rule=maxRule)


def flowRule(model, node):
    if node in model.sources:
        flow_out = sum(model.amount[i, j] for (i, j) in model.arcs if i == node)
        constraint_equation = flow_out <= model.supply[node]

    elif node in model.sinks:
        flow_in = sum(model.amount[i, j] for (i, j) in model.arcs if j == node)
        constraint_equation = flow_in >= model.demand[node]

    else:
        amountIn = sum(model.amount[i, j] for (i, j) in model.arcs if j == node)
        amountOut = sum(model.amount[i, j] for (i, j) in model.arcs if i == node)
        constraint_equation = amountIn == amountOut

    return constraint_equation


model.flow = pyo.Constraint(model.nodes, rule=flowRule)
