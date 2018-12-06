# singlecomm.py - simple version of single commodity flow
from pyomo.environ import *

model = AbstractModel()

model.Nodes = Set()
model.Arcs = model.Nodes * model.Nodes ;

model.Flow = Var(model.Arcs, domain=NonNegativeReals)
model.FlowCost = Param(model.Arcs, default = 0.0)

model.Demand = Param(model.Nodes)
model.Supply = Param(model.Nodes)

def Obj_rule(model):
    return summation(model.FlowCost, model.Flow)
model.Obj = Objective(rule=Obj_rule, sense=minimize)

def FlowBalance_rule(model, node):
    return model.Supply[node] \
     + sum(model.Flow[i, node] for i in model.Nodes) \
     - model.Demand[node] \
     - sum(model.Flow[node, j] for j in model.Nodes) \
     == 0
model.FlowBalance = Constraint(model.Nodes, rule=FlowBalance_rule)
