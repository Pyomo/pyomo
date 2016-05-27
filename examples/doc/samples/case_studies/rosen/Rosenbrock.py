# @intro:
from pyomo.core import *

model = AbstractModel()
# @:intro
# @vars:
model.x = Var(initialize = 1.5)
model.y = Var(initialize = 1.5)
# @:vars
# @obj:
def rosenbrock(amodel):
    return (1.0-amodel.x)**2 \
        + 100.0*(amodel.y - amodel.x**2)**2
model.obj = Objective(rule=rosenbrock, sense=minimize)
# @:obj
