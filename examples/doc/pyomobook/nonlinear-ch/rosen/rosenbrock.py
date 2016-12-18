# rosenbrock.py
# A Pyomo model for the Rosenbrock problem
from pyomo.environ import *

model = AbstractModel()
model.x = Var(initialize=1.5)
model.y = Var(initialize=1.5)

def rosenbrock(model):
    return (1.0-model.x)**2 \
        + 100.0*(model.y - model.x**2)**2
model.obj = Objective(rule=rosenbrock, sense=minimize)
