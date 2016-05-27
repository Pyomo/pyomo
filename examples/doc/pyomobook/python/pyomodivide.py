# pyomodivide.py
import pyomo.environ
from pyomo.core import *

# @all:
model = AbstractModel()

model.pParm = Param(within=Integers, default = 2)
model.wParm = Param(within=PositiveIntegers, default = 4)
model.aVar  = Var(within=NonNegativeReals)

def MyConstraintRule(model):
    if float(value(model.pParm)) / float(value(model.wParm)) > 0.6:
        return model.aVar / model.wParm <= 0.9
    else:
        return model.aVar / model.wParm <= 0.8
model.MyConstraint = Constraint(rule=MyConstraintRule)

def MyObjectiveRule(model):
    return model.wParm * model.aVar
model.MyObjective = Objective(rule=MyObjectiveRule,
                                          sense=maximize)
# @:all

instance = model.create_instance()

instance.pprint()
