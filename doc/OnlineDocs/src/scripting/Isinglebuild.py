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

# Isinglebuild.py
# NodesIn and NodesOut are created by a build action using the Arcs
import pyomo.environ as pyo

model = pyo.AbstractModel()

model.Nodes = pyo.Set()
model.Arcs = pyo.Set(dimen=2)

model.NodesOut = pyo.Set(model.Nodes, within=model.Nodes, initialize=[])
model.NodesIn = pyo.Set(model.Nodes, within=model.Nodes, initialize=[])


def Populate_In_and_Out(model):
    # loop over the arcs and put the end points in the appropriate places
    for i, j in model.Arcs:
        model.NodesIn[j].add(i)
        model.NodesOut[i].add(j)


model.In_n_Out = pyo.BuildAction(rule=Populate_In_and_Out)

model.Flow = pyo.Var(model.Arcs, domain=pyo.NonNegativeReals)
model.FlowCost = pyo.Param(model.Arcs)

model.Demand = pyo.Param(model.Nodes)
model.Supply = pyo.Param(model.Nodes)


def Obj_rule(model):
    return pyo.summation(model.FlowCost, model.Flow)


model.Obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)


def FlowBalance_rule(model, node):
    return (
        model.Supply[node]
        + sum(model.Flow[i, node] for i in model.NodesIn[node])
        - model.Demand[node]
        - sum(model.Flow[node, j] for j in model.NodesOut[node])
        == 0
    )


model.FlowBalance = pyo.Constraint(model.Nodes, rule=FlowBalance_rule)
