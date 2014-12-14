from pyomo.environ import *

def pyomo_create_model(options, model_options):

    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonPositiveReals)
    model.x3 = Var(within=Reals)

    model.o = Objective(expr=model.x1 + 2*model.x2 + 3*model.x3, sense=maximize)

    model.c1 = Constraint(expr= - model.x1 + 3*model.x2 == 5)
    model.c2 = Constraint(expr=2*model.x1 - model.x2 + 3*model.x3 >= 6)
    model.c3 = Constraint(expr=model.x3 <= 4)

    return model
