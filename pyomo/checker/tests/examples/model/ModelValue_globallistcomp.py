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
model.S = RangeSet(10)
model.X = Var(model.S)

if sum(model.X[i] for i in model.S) <= 10.0:
    pass
if sum(value(model.X[i]) for i in model.S) <= 10.0:
    pass
