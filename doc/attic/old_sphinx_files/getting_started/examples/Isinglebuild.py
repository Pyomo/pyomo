# Isinglebuild.py
# NodesIn and NodesOut are created by a build action using the Arcs
from pyomo.environ import *

model = AbstractModel()

model.Nodes = Set()
model.Arcs = Set(dimen=2)

model.NodesOut = Set(model.Nodes, within=model.Nodes, initialize=[])
model.NodesIn = Set(model.Nodes, within=model.Nodes, initialize=[])

def Populate_In_and_Out(model):
    # loop over the arcs and put the end points in the appropriate places
    for (i,j) in model.Arcs:
        model.NodesIn[j].add(i)
        model.NodesOut[i].add(j)

model.In_n_Out = BuildAction(rule = Populate_In_and_Out)

model.Flow = Var(model.Arcs, domain=NonNegativeReals)
model.FlowCost = Param(model.Arcs)

model.Demand = Param(model.Nodes)
model.Supply = Param(model.Nodes)

def Obj_rule(model):
    return summation(model.FlowCost, model.Flow)
model.Obj = Objective(rule=Obj_rule, sense=minimize)

def FlowBalance_rule(model, node):
    return model.Supply[node] \
     + sum(model.Flow[i, node] for i in model.NodesIn[node]) \
     - model.Demand[node] \
     - sum(model.Flow[node, j] for j in model.NodesOut[node]) \
     == 0
model.FlowBalance = Constraint(model.Nodes, rule=FlowBalance_rule)
