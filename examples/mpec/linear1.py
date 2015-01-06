#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# linear1.py
#

from pyomo.core import *
from pyomo.mpec import *


a = 100

model = ConcreteModel()
model.x1 = Var(bounds=(-2,2))
model.x2 = Var(bounds=(-1,1))

model.f = Objective(expr=- model.x1 - model.x2)

model.c = Complementarity(expr=complements(model.x1 >= 0, model.x2 >= 0))

