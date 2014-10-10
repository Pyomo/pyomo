from pyomo.core import *

model = AbstractModel()

model.x = Var(bounds=(0,10))

model.c = Param()

def FirstStageCost_rule(model):
    return model.c*model.x
model.FirstStageCost = Expression(rule=FirstStageCost_rule)

def SecondStageCost_rule(model):
    return 0
model.SecondStageCost = Expression(rule=SecondStageCost_rule)
