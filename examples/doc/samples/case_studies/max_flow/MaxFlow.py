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

from pyomo.core import *

model = AbstractModel()

model.nodes = Set()
model.arcs = Set(within=model.nodes * model.nodes)
model.sources = Set(within=model.nodes)
model.sinks = Set(within=model.nodes)
model.upperBound = Param(model.arcs)
model.supply = Param(model.sources)
model.demand = Param(model.sinks)
model.amount = Var(model.arcs, within=NonNegativeReals)


def totalRule(model):
    expression = sum(model.amount[i, j] for (i, j) in model.arcs if j in model.sinks)
    return expression


model.maxFlow = Objective(rule=totalRule, sense=maximize)


def maxRule(model, arcIn, arcOut):
    constraint_equation = model.amount[arcIn, arcOut] <= model.upperBound[arcIn, arcOut]
    return constraint_equation


model.loadOnArc = Constraint(model.arcs, rule=maxRule)


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


model.flow = Constraint(model.nodes, rule=flowRule)
