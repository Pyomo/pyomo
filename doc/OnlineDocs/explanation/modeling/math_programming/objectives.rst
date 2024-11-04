Objectives
==========

An objective is a function of variables that returns a value that an
optimization package attempts to maximize or minimize. The ``Objective``
function in Pyomo declares an objective. Although other mechanisms are
possible, this function is typically passed the name of another function
that gives the expression. Here is a very simple version of such a
function that assumes ``model.x`` has previously been declared as a
``Var``:

.. doctest::
    :hide:

    >>> import pyomo.environ as pyo
    >>> model = pyo.ConcreteModel()
    >>> model.x = pyo.Var([1,2,3])
    >>> model.y = pyo.Var()
    >>> model.p = pyo.Param([1,2,3], initialize=1)

.. doctest::

    >>> def ObjRule(model):
    ...     return 2*model.x[1] + 3*model.x[2]
    >>> model.obj1 = pyo.Objective(rule=ObjRule)

It is more common for an objective function to refer to parameters as in
this example that assumes that ``model.p`` has been declared as a
``Param`` and that ``model.x`` has been declared with the same index
set, while ``model.y`` has been declared as a singleton:

.. doctest::

    >>> def ObjRule(model):
    ...     return pyo.summation(model.p, model.x) + model.y
    >>> model.obj2 = pyo.Objective(rule=ObjRule, sense=pyo.maximize)

This example uses the ``sense`` option to specify maximization. The
default sense is ``minimize``.
