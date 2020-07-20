def NodesIn_init(model, node):
    retval = []
    for (i,j) in model.Arcs:
        if j == node:
            retval.append(i)
    return retval
model.NodesIn = Set(model.Nodes, initialize=NodesIn_init)
