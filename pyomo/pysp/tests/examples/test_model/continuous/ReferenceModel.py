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

model.x = Var(bounds=(0,10))

model.c = Param()

def FirstStageCost_rule(model):
    return model.c*model.x
model.FirstStageCost = Expression(rule=FirstStageCost_rule)

def SecondStageCost_rule(model):
    return 0
model.SecondStageCost = Expression(rule=SecondStageCost_rule)
