Constraints
===========

Most constraints are specified using equality or inequality expressions
that are created using a rule, which is a Python function. For example, if the variable
``model.x`` has the indexes 'butter' and 'scones', then this constraint limits
the sum for them to be exactly three:

>>> def teaOKrule(model):
>>>     return(model.x['butter'] + model.x['scones'] == 3)
>>> model.TeaConst = Constraint(rule=teaOKrule)

Instead of expressions involving equality (==) or inequalities (`<=` or `>=`),
constraints can also be expressed using a 3-tuple if the form (lb, expr, ub)
where lb and ub can be ``None``, which is interpreted as
lb `<=` expr `<=` ub. Variables can appear only in the middle expr. For example,
the following two constraint declarations have the same meaning:

>>> model.x = Var()
>>>
>>> def aRule(model):
>>>    return model.x >= 2
>>> Boundx = Constraint(rule=aRule)
>>>
>>> def bRule(model):
>>>    return (2, model.x, None)
>>> Boundx = Constraint(rule=bRule)

For this simple example, it would also be possible to declare
``model.x`` with a ``bound`` option to accomplish the same thing.

Constraints (and objectives) can be indexed by lists or sets. When
the declaration contains lists or sets as arguments, the elements are iteratively
passed to the rule function. If there is more than one, then the cross product
is sent. For example the following constraint could be interpreted as
placing a budget of :math:`i` on the :math:`i^{\mbox{th}}` item
to buy where the cost per item is given by the parameter ``model.a``:

>>> model.A = RangeSet(1,10)
>>> model.a = Param(model.A, within=PostiveReals)
>>> model.ToBuy = Var(model.A)
>>> def bud_rule(model, i):
>>>     return model.a[i]*model.ToBuy[i] <= i
>>> aBudget = Constraint(model.A, rule=bud_rule)

NOTE: Python and Pyomo are case sensitive so ``model.a`` is not the same
as ``model.A``.
