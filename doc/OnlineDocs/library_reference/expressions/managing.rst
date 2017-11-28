.. |p| raw:: html

   <p />

Managing Expressions
====================

Cloning Expressions
-------------------

Expressions are automatically cloned only during certain expression
transformations.  Since this can be an expensive operation, the
:data:`clone_counter <pyomo.core.expr.current.clone_counter>` context
manager object is provided to track the number of times the
:func:`clone_expression <pyomo.core.expr.current.clone_expression>`
function is executed.

For example:

.. doctest::

    >>>from pyomo.environ import *
    >>>from pyomo.core.expr import current as EXPR

    >>>M = ConcreteModel()
    >>>M.x = Var()

    >>> with EXPR.current.clone_counter:
    >>>     start = pyomo.core.expr.current.clone_counter.count
    >>>     e1 = sin(M.x)
    >>>     e2 = e1.clone()
    >>>     total = pyomo.core.expr.current.clone_counter.count - start
    >>>     assert(total == 1)

Evaluating Expressions
----------------------

Expressions can be evaluated when all variables and parameters in
the expression have a value.  The :func:`value <pyomo.core.expr.value>`
function can be used to walk the expression tree and compute the
value of an expression.  For example:

.. doctest::

    >>>from pyomo.environ import *
    >>>import math

    >>>M = ConcreteModel()
    >>>M.x = Var()
    >>>M.x.value = math.pi/2.0
    >>>val = value(M.x)
    >>>assert(math.isclose(val,0.0))

Additionally, expressions define the :func:`__call__` method, so the
following is another way to compute the value of an expression:

.. doctest::

    >>>val = M.x()
    >>>assert(math.isclose(val,0.0))

If a parameter or variable is undefined, then the :func:`value <pyomo.core.expr.value>` function and :func:`__call__` method will raise an exception.  This 
exception can be suppressed using the :attr:`exception` option.  For example:

.. doctest::

    >>>from pyomo.environ import *
    >>>import math

    >>>M = ConcreteModel()
    >>>M.x = Var()
    >>>val = value(M.x, exception=False)
    >>>assert(val is None)

This option is useful in contexts where adding a try block is inconvenient 
in your modeling script.

.. note::

    Both the :func:`value <pyomo.core.expr.value>` function and
    :func:`__call__` method call the :func:`evaluate_expression
    <pyomo.core.expr.current.evaluate_expression>` function.  In
    practice, this function will be slightly faster, but the
    difference is only meaningful when expressions are evaluated
    many times.

Identifying Components and Variables
------------------------------------

Expression transformations sometimes need to find all nodes in an
expression tree that are of a given type.  Pyomo contains two utility
functions that support this functionality.  First, the
:func:`identify_components <pyomo.core.expr.current.identify_components>`
function is a generator function that walks the expression tree and yields all 
nodes whose type is in a specified set of node types.  For example:

.. doctest::

    >>>from pyomo.environ import *
    >>>from pyomo.core.expr import current as EXPR

    >>>M = ConcreteModel()
    >>>M.x = Var()
    >>>M.p = Param(mutable=True)

    >>>e = M.p+M.x
    >>>s = set([type(M.p)])
    >>>assert(list(EXPR.identify_components(e, s)), [M.p])

The :func:`identify_variables <pyomo.core.expr.current.identify_variables>`
function is a generator function that yields all nodes that are
variables.  Pyomo uses several different classes to represent variables,
but this set of variable types does not need to be specified by the user.
However, the :attr:`include_fixed` flag can be specified to omit fixed
variables.  For example:

.. doctest::

    >>>from pyomo.environ import *
    >>>from pyomo.core.expr import current as EXPR

    >>>M = ConcreteModel()
    >>>M.x = Var()
    >>>M.y = Var()

    >>>e = M.x+M.y
    >>>M.y.value = 1
    >>>M.y.fixed = True

    >>>assert(set(EXPR.identify_variables(e)), set([M.x, M.y]))
    >>>assert(set(EXPR.identify_variables(e, include_fixed=False)), set([M.x]))



Walking an Expression Tree with a Visitor Class
-----------------------------------------------

TODO

Examples
~~~~~~~~

TODO


