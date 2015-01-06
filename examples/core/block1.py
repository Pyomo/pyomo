#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *

model = ConcreteModel()
model.x = Var()

model.b = Block()
model.b.x = Var()

model.o = Objective(expr=(model.x-1.0)**2 + (model.b.x - 2.0)**2)

