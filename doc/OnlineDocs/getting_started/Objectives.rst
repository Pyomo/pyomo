Objectives
==========

An objective is a function of variables that returns a value that an optimization package attempts to maximize or minimize. The ``Objective`` function in Pyomo declares
an objective. Although other mechanisms are possible, this function is typically
passed the name of another function that gives the expression.
Here is a very simple version of such a function that assumes ``model.x`` has
previously been declared as a ``Var``:

>>> def ObjRule(model):
>>>     return 2*model.x[1] + 3*model.x[2]
>>> model.g = Objective(rule=ObjRule)


It is more common for an objective function to refer to parameters as in this example
that assumes that ``model.p`` has been declared as a parameters and that ``model.x`` has been declared with
the same index set, while ``model.y`` has been declared as a singleton:

>>> def profrul(model):
>>>    return summation(model.p, model.x) + model.y
>>> model.Obj = Objective(rule=ObjRule, sense=maximize)

This example uses the ``sense`` option to specify maximization. The default sense is
``minimize``.
