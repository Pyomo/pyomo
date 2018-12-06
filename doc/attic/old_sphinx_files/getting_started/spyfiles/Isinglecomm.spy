# Isinglecomm.py
# NodesIn and NodesOut are intialized using the Arcs
from pyomo.environ import *

model = AbstractModel()

model.Nodes = Set()
model.Arcs = Set(dimen=2)

def NodesOut_init(model, node):
    retval = []
    for (i,j) in model.Arcs:
        if i == node:
            retval.append(j)
    return retval
model.NodesOut = Set(model.Nodes, initialize=NodesOut_init)

def NodesIn_init(model, node):
    retval = []
    for (i,j) in model.Arcs:
        if j == node:
            retval.append(i)
    return retval
model.NodesIn = Set(model.Nodes, initialize=NodesIn_init)

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