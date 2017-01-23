# multimodal_init1.py
from pyomo.environ import *
from math import pi

model = ConcreteModel()
model.x = Var(initialize = 0.25, bounds=(0,4))
model.y = Var(initialize = 0.25, bounds=(0,4))

def multimodal(m):
    return (2-cos(pi*m.x)-cos(pi*m.y)) * (m.x**2) * (m.y**2)
model.obj = Objective(rule=multimodal, sense=minimize)
