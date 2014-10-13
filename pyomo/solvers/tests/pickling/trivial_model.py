from pyomo.environ import *

# Trivial model
def define_model():

    model = ConcreteModel()
    model.x = Var()
    model.obj = Objective(expr=model.x)
    model.con = Constraint(expr=model.x >= 1)

    return model
