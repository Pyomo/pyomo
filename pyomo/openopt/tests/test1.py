#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *

model = ConcreteModel()
model.x = Var([1,2], initialize=1.0, bounds=(0,1))
model.o = Objective(expr=(model.x[1]+1)**2 + (model.x[2]+1)**2)

