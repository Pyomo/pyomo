from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import pi

model = ConcreteModel()

model.x = Var(bounds=(0,4))
model.y = Var(bounds=(0,4))

model.obj = Objective(expr= \
    (2 - cos(pi*model.x) - cos(pi*model.y)) * \
    (model.x**2) * (model.y**2))

with SolverFactory("ipopt") as solver:
    model.x, model.y = 0.25, 0.25
    print("x0=%s, y0=%s" % (model.x.value, model.y.value))
    solver.solve(model)
    print("x*=%s, y*=%s" % (model.x.value, model.y.value))
    print("")

    model.x, model.y = 2.5, 2.5
    print("x0=%s, y0=%s" % (model.x.value, model.y.value))
    solver.solve(model)
    print("x*=%s, y*=%s" % (model.x.value, model.y.value))
