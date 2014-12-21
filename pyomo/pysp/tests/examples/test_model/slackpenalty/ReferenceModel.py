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

model.x = Var(within=Boolean)
model.y = Var()
model.slackbool = Var(within=Boolean)

model.a = Param()
model.b = Param()
model.c = Param()
model.M = Param()

def y_is_x_rule(model):
    return model.y == model.x
model.y_is_x = Constraint(rule=y_is_x_rule)

def slacker_rule(model):
    return model.a * model.y + model.slackbool >= model.b
model.slacker = Constraint(rule=slacker_rule)

def FirstStageCost_rule(model):
    return 0
model.FirstStageCost = Expression(rule=FirstStageCost_rule)

def SecondStageCost_rule(model):
    return model.c * model.y + model.M * model.slackbool
model.SecondStageCost = Expression(rule=SecondStageCost_rule)

def Obj_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Obj = Objective(rule=Obj_rule)
