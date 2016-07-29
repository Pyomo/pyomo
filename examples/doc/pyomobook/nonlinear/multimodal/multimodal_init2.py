from pyomo.environ import *
from math import pi

model = ConcreteModel()
# @init:
model.x = Var(initialize = 2.0, bounds=(0,4))
model.y = Var(initialize = 2.0, bounds=(0,4))
# @:init

def multimodal(m):
    return (2-cos(pi*m.x)-cos(pi*m.y)) * (m.x**2) * (m.y**2)
model.obj = Objective(rule=multimodal, sense=minimize)
