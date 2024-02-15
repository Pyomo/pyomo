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

# @decl2:
model.B = pyo.Set(initialize=[2, 3, 4])
model.C = pyo.Set(initialize=[(1, 4), (9, 16)])
# @:decl2

# @decl6:
F_init = {}
F_init[2] = [1, 3, 5]
F_init[3] = [2, 4, 6]
F_init[4] = [3, 5, 7]
model.F = pyo.Set([2, 3, 4], initialize=F_init)
# @:decl6


# @decl8:
def J_init(model, i, j):
    return range(0, i * j)


model.J = pyo.Set(model.B, model.B, initialize=J_init)
# @:decl8

# @decl12:
model.P = pyo.Set(initialize=[1, 2, 3, 5, 7])


def filter_rule(model, x):
    return x not in model.P


model.Q = pyo.Set(initialize=range(1, 10), filter=filter_rule)
# @:decl12

# @decl20:
model.R = pyo.Set([1, 2, 3])
model.R[1] = [1]
model.R[2] = [1, 2]
# @:decl20

model.pprint(verbose=True)
