# Example: modeling a complementarity condition as a 
#   disjunction
#
# Specifying BigM suffix values for the gdp.bigm transformation

from pyomo.core import *
from pyomo.gdp import *

def build_model():
    model = ConcreteModel()

    # x >= 0 _|_ y>=0
    model.x = Var(bounds=(0,None))
    model.y = Var(bounds=(0,None))

    # Two conditions
    def _d(disjunct, flag):
        model = disjunct.model()
        if flag:
            # x == 0
            disjunct.c = Constraint(expr=model.x == 0)
        else:
            # y == 0
            disjunct.c = Constraint(expr=model.y == 0)
        disjunct.BigM = Suffix()
        disjunct.BigM[disjunct.c] = 1
    model.d = Disjunct([0,1], rule=_d)

    # Define the disjunction
    def _c(model):
        return [model.d[0], model.d[1]]
    model.c = Disjunction(rule=_c)

    model.C = Constraint(expr=model.x+model.y <= 1)

    model.o = Objective(expr=2*model.x+3*model.y, sense=maximize)

    return model
