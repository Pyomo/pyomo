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

# sodacan.py
from pyomo.environ import *
from math import pi

M = ConcreteModel()
M.r = Var(bounds=(0, None))
M.h = Var(bounds=(0, None))
M.o = Objective(expr=2 * pi * M.r * (M.r + M.h))
M.c = Constraint(expr=pi * M.h * M.r**2 == 355)
