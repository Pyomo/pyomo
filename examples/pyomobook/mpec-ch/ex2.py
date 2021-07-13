# ex2.py
import pyomo.environ as pyo
from pyomo.mpec import *


model = pyo.ConcreteModel()

model.x = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(within=pyo.NonNegativeReals)

model.f1 = pyo.Objective(expr=2*model.x - model.y)

model.compl = Complementarity(
                expr=complements(0 <= model.y,
                                 model.y >= model.x))

# @transform:
xfrm = pyo.TransformationFactory("mpec.simple_nonlinear")
transformed = xfrm.create_using(model)
# @:transform
