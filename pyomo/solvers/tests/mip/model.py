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

model.A = RangeSet(1,4)

model.x = Var(model.A)

def obj_rule(model):
    return summation(model.x)
model.obj = Objective(rule=obj_rule)
