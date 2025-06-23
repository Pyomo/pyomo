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

# simple.py
import pyomo.environ as pyo

M = pyo.ConcreteModel()
M.x1 = pyo.Var()
M.x2 = pyo.Var(bounds=(-1, 1))
M.x3 = pyo.Var(bounds=(1, 2))
M.o = pyo.Objective(
    expr=M.x1**2 + (M.x2 * M.x3) ** 4 + M.x1 * M.x3 + M.x2 * pyo.sin(M.x1 + M.x3) + M.x2
)

model = M
