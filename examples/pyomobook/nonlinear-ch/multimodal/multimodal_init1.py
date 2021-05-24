# multimodal_init1.py
import pyomo.environ as pyo
from math import pi

model = pyo.ConcreteModel()
model.x = pyo.Var(initialize = 0.25, bounds=(0,4))
model.y = pyo.Var(initialize = 0.25, bounds=(0,4))

def multimodal(m):
    return (2-pyo.cos(pi*m.x)-pyo.cos(pi*m.y)) * (m.x**2) * (m.y**2)
model.obj = pyo.Objective(rule=multimodal, sense=pyo.minimize)

status = pyo.SolverFactory('ipopt').solve(model)
pyo.assert_optimal_termination(status)
print(pyo.value(model.x), pyo.value(model.y))
