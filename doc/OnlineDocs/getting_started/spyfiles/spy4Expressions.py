"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Expressions.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()

# @Buildup_expression_switch
switch = 3

model.A = RangeSet(1, 10)
model.c = Param(model.A)
model.d = Param()
model.x = Var(model.A, domain=Boolean)

def pi_rule(model):
    accexpr = summation(model.c, model.x)
    if switch >= 2:
        accexpr = accexpr - model.d
    return accexpr >= 0.5
PieSlice = Constraint(rule=pi_rule)
# @Buildup_expression_switch

# @Abstract_wrong_usage
model.A = RangeSet(1, 10)
model.c = Param(model.A)
model.d = Param()
model.x = Var(model.A, domain=Boolean)

def pi_rule(model):
    accexpr = summation(model.c, model.x)
    if model.d >= 2:  # NOT in an abstract model!!
        accexpr = accexpr - model.d
    return accexpr >= 0.5
PieSlice = Constraint(rule=pi_rule)
# @Abstract_wrong_usage

# @Declare_piecewise_constraints
#model.pwconst = Piecewise(indexes, yvar, xvar, **Keywords)
#model.pwconst = Piecewise(yvar,xvar,**Keywords)
# @Declare_piecewise_constraints

# @f_rule_Function_examples
# A function that changes with index
def f(model,j,x):
   if (j == 2):
      return x**2 + 1.0
   else:
      return x**2 + 5.0

# A nonlinear function
f = lambda model,x : exp(x) + value(model.p)

# A step function
f = [0,0,1,1,2,2]
# @f_rule_Function_examples

# @Keyword_assignment_example
kwds = {'pw_constr_type':'EQ','pw_repn':'SOS2','sense':maximize,'force_pw':True}
# @Keyword_assignment_example

# @Expression_objects_illustration
model = ConcreteModel()
model.x = Var(initialize=1.0)
def _e(m,i):
    return m.x*i
model.e = Expression([1,2,3], rule=_e)

instance = model.create_instance()

print (value(instance.e[1])) # -> 1.0
print (instance.e[1]())           # -> 1.0
print (instance.e[1].value)  # -> a pyomo expression object

# Change the underlying expression
instance.e[1].value = instance.x**2

#... solve
#... load results

# print the value of the expression given the loaded optimal solution
print (value(instance.e[1]))
# @Expression_objects_illustration

# @Define_python_function
def f(x, p):
    return x + p
# @Define_python_function

# @Generate_new_expression
model = ConcreteModel()
model.x = Var()

# create a Pyomo expression
e1 = model.x + 5

# create another Pyomo expression
# e1 is copied when generating e2
e2 = e1 + model.x
# @Generate_new_expression
