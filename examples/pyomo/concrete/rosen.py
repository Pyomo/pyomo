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

# rosen.py
import pyomo.environ as pyo

M = pyo.ConcreteModel()
M.x = pyo.Var()
M.y = pyo.Var()
M.o = pyo.Objective(expr=(M.x - 1) ** 2 + 100 * (M.y - M.x**2) ** 2)

model = M
