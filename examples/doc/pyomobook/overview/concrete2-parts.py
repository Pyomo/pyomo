import pyomo.environ
from pyomo.core import *

model = ConcreteModel()

# @var:
model.x = Var([1,2], within=NonNegativeReals)
# @:var

# @expr:
model.obj = Objective(expr=model.x[1] + 2*model.x[2])
model.con1 = Constraint(expr=3*model.x[1] + 4*model.x[2]>=1)
model.con2 = Constraint(expr=2*model.x[1] + 5*model.x[2]>=2)
# @:expr
