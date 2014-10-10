# linear1.py
#
import coopr.environ
from coopr.pyomo import *
from coopr.mpec import Complementarity

a = 100

model = ConcreteModel()
model.x1 = Var(bounds=(-2,2))
model.x2 = Var(bounds=(-1,1))

model.f = Objective(expr=- model.x1 - model.x2)

model.c = Complementarity(expr=(model.x1 >= 0, model.x2 >= 0))


#model = TransformationFactory('mpec.simple_disjunction').apply(model)
#model.pprint()
#model = TransformationFactory('gdp.bigm').apply(model)
if False:
    model.c.to_standard_form()
    model.reclassify_component_type(model.c, Block)
    model.pprint()
