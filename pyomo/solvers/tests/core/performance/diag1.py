from pyomo.core import *

def create_model(N):
    model = ConcreteModel()

    for i in range(1,N+1):
        setattr(model, 'x'+str(i), Var())

    expr=sum(i*getattr(model, 'x'+str(i)) for i in range(1,N+1))
    model.obj = Objective(expr=expr)

    for i in range(1,N+1):
        setattr(model, 'c'+str(i), Constraint(expr = (N-i+1)*getattr(model, 'x'+str(i)) >= N))

    return model

def pyomo_create_model(options=None, model_options=None):
    return create_model(1000000)
