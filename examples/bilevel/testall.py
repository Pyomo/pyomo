import pyomo.environ
from pyomo.core import *
from pyomo.bilevel import *

def pyomo_create_model(options, model_options):

    model = ConcreteModel()
    model.z = Var(within=NonPositiveReals)
    model.x1 = Var(bounds=(0, None))
    model.x2 = Var(bounds=(None, 0))
    model.x3 = Var(bounds=(1, None))
    model.x4 = Var(bounds=(None, -2))
    model.x5 = Var(bounds=(-3, 4))
    model.y1 = Var([0,1], bounds=(0, None))
    model.y2 = Var([0,1], bounds=(None, 0))
    model.y3 = Var([0,1], bounds=(1, None))
    model.y4 = Var([0,1], bounds=(None, -2))
    model.y5 = Var([0,1], bounds=(-3, 4))
    model.o = Objective(expr=model.z*(10*model.x1 + 11*model.x2 + 12*model.x3 + 13*model.x4), sense=maximize)

    # Create a submodel
    # The argument indicates the lower-level decision variables
    model.sub = SubModel(fixed=model.z)
    model.sub.o = Objective(expr=model.o.expr, sense=minimize)
    model.sub.c1 = Constraint(expr=20*model.x1 + 21*model.x2 + 22*model.x3 + 23*model.x4 + 24*model.x5 == 25)
    model.sub.c2 = Constraint(expr=30*model.x1 + 31*model.x2 + 32*model.x3 + 33*model.x4 + 34*model.x5 <= 35)
    model.sub.c3 = Constraint(expr=40*model.x1 + 41*model.x2 + 42*model.x3 + 43*model.x4 + 44*model.x5 >= 45)

    return model

