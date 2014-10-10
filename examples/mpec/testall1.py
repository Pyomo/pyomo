#
# A test model to verify the internal representation of the
# complementarity conditions.
#

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import Complementarity

model = ConcreteModel()
model.y  = Var()
model.x1 = Var()
model.x2 = Var()
model.x3 = Var()

# y + x1 >= 0  _|_  x1 + 2*x2 + 3*x3 >= 1
model.f1 = Complementarity(expr=(model.y + model.x1 >= 0, model.x1 + 2*model.x2 + 3*model.x3 >= 1))
# y + x2 >= 0  _|_  x2 - x3 <= -1
model.f2 = Complementarity(expr=(model.y + model.x2 >= 0, model.x2 - model.x3 <= -1))
# y + x3 >= 0  _|_  x1 + x2 >= -1
model.f3 = Complementarity(expr=(model.y + model.x3 >= 0, model.x1 + model.x2 >= -1))

model.h1 = Complementarity(expr=(model.x1 + 2*model.x2 + 3*model.x3 >= 1, model.y + model.x1 >= 0))
model.h2 = Complementarity(expr=(model.x2 - model.x3 <= -1, model.y + model.x2 >= 0))
model.h3 = Complementarity(expr=(model.x1 + model.x2 >= -1, model.y + model.x3 >= 0))

# x1 + 2*x2 + 3*x3 = 1  _|_  y + x3
model.h4 = Complementarity(expr=(model.x1 + 2*model.x2 == 1, model.y + model.x3))
model.h5 = Complementarity(expr=(model.y + model.x3, model.x1 + 2*model.x2 == 1))

# 1 <= x1 + 2*x2 <= 2  _|_  y + x3
model.h6 = Complementarity(expr=(-1 <= model.x1 + 2*model.x2 <= 2, model.y + model.x3))
model.h7 = Complementarity(expr=(model.y + model.x3, -1 <= model.x1 + 2*model.x2 <= 2))

model.pprint()


instance = TransformationFactory('mpec.simple_nonlinear').apply(model)
instance = model.transform('mpec.simple_nonlinear')
#instance = TransformationFactory('mpec.simple_disjunction').apply(model)

instance.pprint()
