from pyomo.core import *

def create_model(N):
    model = ConcreteModel()

    model.A = RangeSet(N)
    model.x = Var(model.A, bounds=(1,2))

    expr=sum(2*model.x[i]*model.x[i+1] for i in model.A if (i+1) in model.A)
    model.obj = Objective(expr=expr)

    return model

def pyomo_create_model(options=None, model_options=None):
    return create_model(100000)
