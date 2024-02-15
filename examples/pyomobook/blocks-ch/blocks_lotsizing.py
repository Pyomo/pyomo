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

import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.T = pyo.RangeSet(5)  # time periods

i0 = 5.0  # initial inventory
c = 4.6  # setup cost
h_pos = 0.7  # inventory holding cost
h_neg = 1.2  # shortage cost
P = 5.0  # maximum production amount

# demand during period t
d = {1: 5.0, 2: 7.0, 3: 6.2, 4: 3.1, 5: 1.7}


# @blockrule:
# create a block for a single time period
def lotsizing_block_rule(b, t):
    # define the variables
    b.y = pyo.Var(domain=pyo.Binary)
    b.x = pyo.Var(domain=pyo.NonNegativeReals)
    b.i = pyo.Var()
    b.i0 = pyo.Var()
    b.i_pos = pyo.Var(domain=pyo.NonNegativeReals)
    b.i_neg = pyo.Var(domain=pyo.NonNegativeReals)

    # define the constraints
    b.inventory = pyo.Constraint(expr=b.i == b.i0 + b.x - d[t])
    b.pos_neg = pyo.Constraint(expr=b.i == b.i_pos - b.i_neg)
    b.prod_indicator = pyo.Constraint(expr=b.x <= P * b.y)


model.lsb = pyo.Block(model.T, rule=lotsizing_block_rule)
# @:blockrule


# link the inventory variables between blocks
def i_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i0
    return m.lsb[t].i0 == m.lsb[t - 1].i


model.i_linking = pyo.Constraint(model.T, rule=i_linking_rule)


# construct the objective function over all the blocks
def obj_rule(m):
    return sum(
        c * m.lsb[t].y + h_pos * m.lsb[t].i_pos + h_neg * m.lsb[t].i_neg for t in m.T
    )


model.obj = pyo.Objective(rule=obj_rule)

### solve the problem
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# print the results
for t in model.T:
    print('Period: {0}, Prod. Amount: {1}'.format(t, pyo.value(model.lsb[t].x)))
