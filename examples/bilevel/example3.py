import coopr.environ
from coopr.pyomo import *
from coopr.bilevel import *

model = ConcreteModel()
model.x = Var(bounds=(1,2))
model.y = Var(bounds=(1,2))
model.o = Objective(expr=model.x + model.y)

model.sub = SubModel(fixed=model.x)
model.sub.z = Var(bounds=(-1,1))
model.sub.o = Objective(expr=model.x*model.sub.z, sense=maximize)
model.sub.c = Constraint(expr=model.y + model.sub.z <= 2)

