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

# sodacan.py
import pyomo.environ as pyo
from math import pi

M = pyo.ConcreteModel()
M.r = pyo.Var(bounds=(0, None))
M.h = pyo.Var(bounds=(0, None))
M.o = pyo.Objective(expr=2 * pi * M.r * (M.r + M.h))
M.c = pyo.Constraint(expr=pi * M.h * M.r**2 == 355)
