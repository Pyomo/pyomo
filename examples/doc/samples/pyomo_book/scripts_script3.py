from pyomo.environ import *

def create_myinstance(data):
    model = ConcreteModel()
    model.x = Var()
    model.o = Objective(expr= model.x)
    model.c = Constraint(expr= model.x >= data)
    return model

with SolverFactory("glpk") as opt:
    for data in [1.0, 2.0, 3.0]:
        instance = create_myinstance(data)
        opt.solve(instance)
        print("Objective: %s" % (instance.o()))
