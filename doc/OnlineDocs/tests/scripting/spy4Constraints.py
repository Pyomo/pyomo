"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Constraints.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()
# @Inequality_constraints_2expressions
model.x = Var()

def aRule(model):
   return model.x >= 2
model.Boundx = Constraint(rule=aRule)

def bRule(model):
   return (2, model.x, None)
model.boundx = Constraint(rule=bRule)
# @Inequality_constraints_2expressions

model = ConcreteModel()
model.J = Set(initialize=['butter','scones'])
model.x = Var(model.J)
# @Constraint_example
def teaOKrule(model):
    return(model.x['butter'] + model.x['scones'] == 3)
model.TeaConst = Constraint(rule=teaOKrule)
# @Constraint_example

# @Passing_elements_crossproduct
model.A = RangeSet(1,10)
model.a = Param(model.A, within=PositiveReals)
model.ToBuy = Var(model.A)
def bud_rule(model, i):
    return model.a[i]*model.ToBuy[i] <= i
aBudget = Constraint(model.A, rule=bud_rule)
# @Passing_elements_crossproduct
