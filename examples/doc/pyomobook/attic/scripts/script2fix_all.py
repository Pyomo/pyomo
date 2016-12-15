from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import pi

model = ConcreteModel()

model.x = Var(bounds=(0,4))
model.y = Var(bounds=(0,4))

model.obj = Objective(expr= \
    (2 - cos(pi*model.x) - cos(pi*model.y)) * \
    (model.x**2) * (model.y**2))

model.y.fix(3.5)
model.x.value = 3.5

with SolverFactory("ipopt") as solver:
    solver.solve(model)

print("First   x was %f and y was %f"
      % (model.x.value, model.y.value))

model.x.fixed = True
model.y.fixed = False

with SolverFactory("ipopt") as solver:
    solver.solve(model)

print("Next    x was %f and y was %f"
      % (model.x.value, model.y.value))

model.x.fixed = False
model.y.fixed = True

with SolverFactory("ipopt") as solver:
    solver.solve(model)

print("Finally x was %f and y was %f"
      % (model.x.value, model.y.value))
