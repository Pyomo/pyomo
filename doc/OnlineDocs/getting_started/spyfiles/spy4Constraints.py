"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Constraints.rst in testable form
"""
# @Constraint_example
>>> def teaOKrule(model):
>>>     return(model.x['butter'] + model.x['scones'] == 3)
>>> model.TeaConst = Constraint(rule=teaOKrule)
# @Constraint_example

# @Inequality_constraints_2expressions
>>> model.x = Var()
>>>
>>> def aRule(model):
>>>    return model.x >= 2
>>> Boundx = Constraint(rule=aRule)
>>>
>>> def bRule(model):
>>>    return (2, model.x, None)
>>> Boundx = Constraint(rule=bRule)
# @Inequality_constraints_2expressions

# @Passing_elements_crossproduct
>>> model.A = RangeSet(1,10)
>>> model.a = Param(model.A, within=PostiveReals)
>>> model.ToBuy = Var(model.A)
>>> def bud_rule(model, i):
>>>     return model.a[i]*model.ToBuy[i] <= i
>>> aBudget = Constraint(model.A, rule=bud_rule)
# @Passing_elements_crossproduct
