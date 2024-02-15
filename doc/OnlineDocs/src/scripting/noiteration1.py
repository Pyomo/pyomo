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

# noiteration1.py

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create a solver
opt = SolverFactory('glpk')

#
# A simple model with binary variables and
# an empty constraint list.
#
model = pyo.ConcreteModel()
model.n = pyo.Param(default=4)
model.x = pyo.Var(pyo.RangeSet(model.n), within=pyo.Binary)


def o_rule(model):
    return pyo.summation(model.x)


model.o = pyo.Objective(rule=o_rule)
model.c = pyo.ConstraintList()

results = opt.solve(model)

if pyo.value(model.x[2]) == 0:
    print("The second index has a zero")
else:
    print("x[2]=", pyo.value(model.x[2]))
