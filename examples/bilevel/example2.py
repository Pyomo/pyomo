from pyomo.core import *
from pyomo.bilevel import *
# From slide 5...

model = ConcreteModel()
model.a = Param()
model.b = Param()
model.m = Param()
model.x = Var()
model.z = Var()
model.constr = Constraint(expr=model.a*model.x + model.b*model.z <= model.m)

model.sub.obj = Objective( expr=model.a*model.x + model.b*model.z,
                           sense=minimize )

def _submodel(m):
    sub.c = Param()
    sub.d = Param()
    sub.x = Param()
    sub.y = Var()

    sub.constr = Constraint(expr=sub.c*sub.x + sub.d*sub.y <= sub.model().m)

    sub.obj = Objective( expr=sub.c*sub.x + sub.d*sub.y,
                         sense=maximize )
    
model.sub = SubModel( rule=_submodel
                      map={model.a : 'c', 
                           model.b : 'd',
                           model.x : 'x',
                           model.z : 'y'},
                      solution=pessismistic,
                  )
