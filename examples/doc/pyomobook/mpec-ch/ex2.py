# ex2.py
from pyomo.environ import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

model.f1 = Objective(expr=2*model.x - model.y)

model.compl = Complementarity(
                expr=complements(0 <= model.y,
                                 model.y >= model.x))

# @transform:
xfrm = TransformationFactory("mpec.simple_nonlinear")
transformed = xfrm.create_using(model)
# @:transform
