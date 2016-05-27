from pyomo.core import *

def create_model(N):
    model = ConcreteModel()

    model.A = RangeSet(N)
    def domain_rule(model, i):
        return Reals
    model.x = Var(model.A, domain=domain_rule)

    expr=sum(i*model.x[i] for i in model.A)
    model.obj = Objective(expr=expr)

    def c_rule(model, i):
        return (N-i+1)*model.x[i] >= N
    model.c = Constraint(model.A)

    return model

def pyomo_create_model(options=None, model_options=None):
    return create_model(100000)
