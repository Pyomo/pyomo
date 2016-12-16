# @pyomo:
from pyomo.environ import *
from pyomo.bilevel import *

model = ConcreteModel()
model.x = Var(bounds=(1,2))
model.v = Var(bounds=(1,2))
model.sub = SubModel()
model.sub.y = Var(bounds=(1,2))
model.sub.w = Var(bounds=(-1,1))

model.o = Objective(expr=model.x + model.sub.y + model.v)
model.c = Constraint(expr=model.x + model.v >= 1.5)
model.sub.o = Objective(expr=model.x+model.sub.w, sense=maximize)
model.sub.c = Constraint(expr=model.sub.y + model.sub.w <= 2.5)
# @:pyomo

model.pprint()
