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

# rosen.py
from pyomo.environ import *

M = ConcreteModel()
M.x = Var()
M.y = Var()
M.o = Objective(expr=(M.x - 1) ** 2 + 100 * (M.y - M.x**2) ** 2)

model = M
