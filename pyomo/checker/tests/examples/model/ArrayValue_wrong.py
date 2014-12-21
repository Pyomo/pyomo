#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()
model.y = Var([10])
model.s = RangeSet(10)
model.z = Var(model.s)

model.y.value = 42
model.z.value = 42
