#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.environ
from pyomo.core import *
from pyomo.bilevel import *

model = ConcreteModel()
model.x = Var(bounds=(1,2))
model.y = Var(bounds=(1,2))
model.o = Objective(expr=model.x + model.y)

model.sub = SubModel(fixed=model.x)
model.sub.z = Var(bounds=(-1,1))
model.sub.o = Objective(expr=model.x*model.sub.z, sense=maximize)
model.sub.c = Constraint(expr=model.y + model.sub.z <= 2)

