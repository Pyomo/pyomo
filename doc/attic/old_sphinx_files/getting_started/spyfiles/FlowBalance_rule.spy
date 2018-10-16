def FlowBalance_rule(model, node):
    return model.Supply[node] \
     + sum(model.Flow[i, node] for i in model.Nodes if (i,node) in model.Arcs) \
     - model.Demand[node] \
     - sum(model.Flow[node, j] for j in model.Nodes if (j,node) in model.Arcs) \
     == 0