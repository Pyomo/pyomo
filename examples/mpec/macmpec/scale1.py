# scale1.py
# QLR-AY-LCP-2-0-1
#
import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *

a = 100

model = ConcreteModel()
model.x1 = Var()
model.x2 = Var()

model.f = Objective(expr=(a*model.x1 - 1)**2 + (model.x2 - 1)**2)

model.c = Complementarity(expr=complements(model.x1 >= 0, model.x2 >= 0))

