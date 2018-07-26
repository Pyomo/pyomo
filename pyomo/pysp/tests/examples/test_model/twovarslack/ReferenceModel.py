#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()

model.xIndex = RangeSet(2)

model.x = Var(model.xIndex, within=Boolean)
model.y = Var()
model.slackbool = Var(within=Boolean)

model.a = Param()
model.b = Param()
model.c = Param()
model.M = Param()

# you can get y=one with anything, but for y=zero, you need all x zero
def y_is_geq_x_rule(model, i):
    return model.y >= model.x[i]
model.y_is_geq_x = Constraint(model.xIndex, rule=y_is_geq_x_rule)

# you can get y=zero with anything, but for y=one, you need at least one one
# but if you don't have at least one one, you have to have y=0
def y_is_leq_sum_x_rule(model):
    return model.y <= sum_product(model.x)
model.y_is_leq_sum_x = Constraint(rule=y_is_leq_sum_x_rule)

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
