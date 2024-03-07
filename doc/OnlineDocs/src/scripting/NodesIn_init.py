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


def NodesIn_init(model, node):
    retval = []
    for i, j in model.Arcs:
        if j == node:
            retval.append(i)
    return retval


model.NodesIn = Set(model.Nodes, initialize=NodesIn_init)
