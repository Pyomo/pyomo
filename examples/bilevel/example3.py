#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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

