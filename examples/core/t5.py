#
# A duality example adapted from
#    http://www.stanford.edu/~ashishg/msande111/notes/chapter4.pdf
#
from pyomo.environ import *

def pyomo_create_model(options, model_options):

    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.o = Objective(expr=3*model.x1 + 2.5*model.x2, sense=minimize)

    model.c1 = Constraint(expr=4.44*model.x1 <= 100)
    model.c2 = Constraint(expr=6.67*model.x2 <= 100)
    model.c3 = Constraint(expr=4*model.x1 + 2.86*model.x2 <= 100)
    model.c4 = Constraint(expr=3*model.x1 + 6*model.x2 <= 100)

    return model

