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


def pipe_rule(pipe, i):
    m = pipe.model()
    pipe.flow = Var()
    pipe.pIn = Var(within=NonNegativeReals)
    pipe.pOut = Var(within=NonNegativeReals)
    pipe.pDrop = Constraint(
        expr=pipe.pIn - pipe.pOut == m.friction * m.pipe_length[i] * pipe.flow
    )

    pipe.IN = Connector()
    pipe.IN.add(-pipe.flow, "flow")
    pipe.IN.add(pipe.pIn, "pressure")

    pipe.OUT = Connector()
    pipe.OUT.add(pipe.flow)
    pipe.OUT.add(pipe.pOut, "pressure")


def node_rule(node, i):
    def _mass_balance(node, flows):
        return node.model().demands[i] == sum_product(flows)

    node.flow = VarList()
    node.pressure = Var(within=NonNegativeReals)
    node.port = Connector()
    # node.port.add(node.flow,
    #              aggregate=lambda n,v: n.model().demands[id] == sum_product(v))
    node.port.add(node.flow, aggregate=_mass_balance)
    node.port.add(node.pressure)


def _src_rule(model, pipe):
    return model.nodes[value(model.pipe_links[pipe, 0])].port == model.pipes[pipe].IN


def _sink_rule(model, pipe):
    return model.nodes[value(model.pipe_links[pipe, 1])].port == model.pipes[pipe].OUT


model = AbstractModel()
model.PIPES = Set()
model.NODES = Set()

model.friction = Param(within=NonNegativeReals)
model.pipe_length = Param(model.PIPES, within=NonNegativeReals)
model.pipe_links = Param(model.PIPES, [0, 1])
model.demands = Param(model.NODES, within=Reals, default=0)

model.pipes = Block(model.PIPES, rule=pipe_rule)
model.nodes = Block(model.NODES, rule=node_rule)

# Connect the network
model.network_src = Constraint(model.PIPES, rule=_src_rule)
model.network_sink = Constraint(model.PIPES, rule=_sink_rule)


# Solve so the minimum pressure in the network is 0
def _obj(model):
    return sum(model.nodes[n].pressure for n in model.NODES)


model.obj = Objective(rule=_obj)
