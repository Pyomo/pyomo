"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Disjunctions.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()

model.x = Var()
model.y = Var()

# @Disjunct_and_disjunction
from pyomo.gdp import *
# Two conditions
def _d(disjunct, flag):
    model = disjunct.model()
    if flag:
        # x == 0
        disjunct.c = Constraint(expr=model.x == 0)
    else:
        # y == 0
        disjunct.c = Constraint(expr=model.y == 0)
model.d = Disjunct([0,1], rule=_d)

# Define the disjunction
def _c(model):
    return [model.d[0], model.d[1]]
model.c = Disjunction(rule=_c)
# @Disjunct_and_disjunction
