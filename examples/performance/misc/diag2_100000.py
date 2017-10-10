from pyomo.environ import *

def create_model(N):
    model = ConcreteModel()

    model.A = RangeSet(N)
    model.x = Var(model.A)

    with linear_expression as expr:
        expr=sum((i*model.x[i] for i in model.A), expr)
    model.obj = Objective(expr=expr)

    def c_rule(model, i):
        with linear_expression as expr:
            expr = (N-i+1)*model.x[i]
        return expr >= N
    model.c = Constraint(model.A)

    return model

def pyomo_create_model(options=None, model_options=None):
    return create_model(100000)
