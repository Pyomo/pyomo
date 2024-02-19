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
# simple.py
from pyomo.environ import *

M = ConcreteModel()
M.x1 = Var()
M.x2 = Var(bounds=(-1, 1))
M.x3 = Var(bounds=(1, 2))
M.o = Objective(
    expr=M.x1**2 + (M.x2 * M.x3) ** 4 + M.x1 * M.x3 + M.x2 * sin(M.x1 + M.x3) + M.x2
)

model = M
