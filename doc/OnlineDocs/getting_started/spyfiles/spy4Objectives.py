"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Objectives.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()

model.x = Var([1,2])
model.y = Var()
model.p = 2.0

# @Objective_function_expression
def ObjRule(model):
    return 2*model.x[1] + 3*model.x[2]
model.g = Objective(rule=ObjRule)
# @Objective_function_expression


# @Objective_refer_parameters
def profrul(model):
    return summation(model.p, model.x) + model.y
model.Obj = Objective(rule=ObjRule, sense=maximize)
# @Objective_refer_parameters
