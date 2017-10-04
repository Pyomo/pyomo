from pyomo.environ import *

def create_model(N):

    model = ConcreteModel()

    model.A = RangeSet(N)
    model.x = Var(model.A, bounds=(1,2))

    with nonlinear_expression as expr:
        for i in model.A:
            if not (i+1) in model.A:
                continue
            expr += i*(model.x[i]*model.x[i+1]+1)
    model.obj = Objective(expr=expr)

    return model

def pyomo_create_model(options=None, model_options=None):
    return create_model(100)
