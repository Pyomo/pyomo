from pyomo.environ import *

model = AbstractModel()

model.x = Var(initialize = 1.5)
model.y = Var(initialize = 1.5)

def rosenbrock(MOD):
    return (1.0-MOD.x)**2 \
        + 100.0*(MOD.y - MOD.x**2)**2
model.obj = Objective(rule=rosenbrock, sense=minimize)
