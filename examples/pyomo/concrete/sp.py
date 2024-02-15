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

# sp.py
from pyomo.environ import *
from sp_data import *  # define c, b, h, and d

scenarios = range(1, 6)

M = ConcreteModel()
M.x = Var(within=NonNegativeReals)


def b_rule(B, i):
    B.y = Var()
    B.l = Constraint(expr=B.y >= (c - b) * M.x + b * d[i])
    B.u = Constraint(expr=B.y >= (c + h) * M.x + h * d[i])
    return B


M.B = Block(scenarios, rule=b_rule)


def o_rule(M):
    return sum(M.B[i].y for i in scenarios) / 5.0


M.o = Objective(rule=o_rule)
