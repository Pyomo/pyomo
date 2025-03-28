#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# sp.py
import pyomo.environ as pyo
from sp_data import c, b, h, d  # define c, b, h, and d

scenarios = range(1, 6)

M = pyo.ConcreteModel()
M.x = pyo.Var(within=pyo.NonNegativeReals)


def b_rule(B, i):
    B.y = pyo.Var()
    B.l = pyo.Constraint(expr=B.y >= (c - b) * M.x + b * d[i])
    B.u = pyo.Constraint(expr=B.y >= (c + h) * M.x + h * d[i])
    return B


M.B = pyo.Block(scenarios, rule=b_rule)


def o_rule(M):
    return sum(M.B[i].y for i in scenarios) / 5.0


M.o = pyo.Objective(rule=o_rule)
