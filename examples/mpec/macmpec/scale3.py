# scale1.py
#
import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *

a = 100

model = ConcreteModel()
model.x1 = Var()
model.x2 = Var()

model.f = Objective(expr=(a*model.x1 - 1)**2 + a*(model.x2 - 1)**2)

model.c = Complementarity(expr=complements(model.x1 >= 0, model.x2 >= 0))

