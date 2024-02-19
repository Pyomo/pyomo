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
#
# linear1.py
#

from pyomo.environ import *
from pyomo.mpec import *


a = 100

model = ConcreteModel()
model.x1 = Var(bounds=(-2, 2))
model.x2 = Var(bounds=(-1, 1))

model.f = Objective(expr=-model.x1 - 2 * model.x2)

model.c = Complementarity(expr=complements(model.x1 >= 0, model.x2 >= 0))
