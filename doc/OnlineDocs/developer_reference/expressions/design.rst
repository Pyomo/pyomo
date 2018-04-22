.. |p| raw:: html

   <p />

Design Details
==============

.. warning::
    Pyomo expression trees are not composed of Python
    objects from a single class hierarchy.  Consequently, Pyomo
    relies on duck typing to ensure that valid expression trees are 
    created.

Most Pyomo expression trees have the following form

1. Interior nodes are objects that inherit from the :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>` class.  These objects typically have one or more child nodes.  Linear expression nodes do not have child nodes, but they are treated as interior nodes in the expression tree because they references other leaf nodes.

2. Leaf nodes are numeric values, parameter components and variable components, which represent the *inputs* to the expresion.

Expression Classes
------------------

Expression classes typically represent unary and binary operations.  The following table 
describes the standard operators in Python and their associated Pyomo expression class:

========== ============= =============================================================================
Operation  Python Syntax Pyomo Class
========== ============= =============================================================================
sum        ``x + y``     :class:`SumExpression <pyomo.core.expr.current.SumExpression>`
product    ``x * y``     :class:`ProductExpression <pyomo.core.expr.current.ProductExpression>`
negation   ``- x``       :class:`NegationExpression <pyomo.core.expr.current.NegationExpression>`
reciprocal ``1 / x``     :class:`ReciprocalExpression <pyomo.core.expr.current.ReciprocalExpression>`
power      ``x ** y``    :class:`PowExpression <pyomo.core.expr.current.PowExpression>`
inequality ``x <= y``    :class:`InequalityExpression <pyomo.core.expr.current.InequalityExpression>`
equality   ``x == y``    :class:`EqualityExpression <pyomo.core.expr.current.EqualityExpression>`
========== ============= =============================================================================

Additionally, there are a variety of other Pyomo expression classes that capture more general 
logical relationships, which are summarized in the following table:

==================== ====================================   ========================================================================================
Operation            Example                                Pyomo Class
==================== ====================================   ========================================================================================
exernal function     ``myfunc(x,y,z)``                      :class:`ExternalFunctionExpression <pyomo.core.expr.current.ExternalFunctionExpression>`
logical if-then-else ``Expr_if(IF=x, THEN=y, ELSE=z)``      :class:`Expr_ifExpression <pyomo.core.expr.current.Expr_ifExpression>`
intrinsic function   ``sin(x)``                             :class:`UnaryFunctionExpression <pyomo.core.expr.current.UnaryFunctionExpression>`
absolute function    ``abs(x)``                             :class:`AbsExpression <pyomo.core.expr.current.AbsExpression>`
==================== ====================================   ========================================================================================

Expression objects are immutable.  Specifically, the list of
arguments to an expression object (a.k.a. the list of child nodes
in the tree) cannot be changed after an expression class is
constructed.  To enforce this property, expression objects have a
standard API for accessing expression arguments:

* :attr:`args` - a class property that returns a generator that yields the expression arguments
* :attr:`arg(i)` - a function that returns the ``i``-th argument
* :attr:`nargs()` - a function that returns the number of expression arguments

.. warning::

    Developers should never use the :attr:`_args_` property directly!
    The semantics for the use of this data has changed since earlier
    versions of Pyomo.  For example, in some expression classes the
    the value :func:`nargs()` may not equal :const:`len(_args_)`!

Expression trees can be categorized in four different ways:

* constant expressions - expressions that do not contain numeric constants and immutable parameters.
* mutable expressions - expressions that contain mutable parameters but no variables.
* potentially variable expressions - expressions that contain variables, which may be fixed.
* fixed expressions - expressions that contain variables, all of which are fixed.

These three categories are illustrated with the following example:

.. literalinclude:: ../../tests/expr/design_categories.spy

The following table describes four different simple expressions
that consist of a single model component, and it shows how they
are categorized:

======================== ===== ===== ===== =====
Category                 m.p   m.q   m.x   m.y
======================== ===== ===== ===== =====
constant                 True  False False False
not potentially variable True  True  False False
potentially_variable     False False True  True
fixed                    True  True  False True
======================== ===== ===== ===== =====

Expressions classes contain methods to test whether an expression
tree is in each of these categories.  Additionally, Pyomo includes
custom expression classes for expression trees that are *not potentially
variable*.  These custom classes will not normally be used by
developers, but they provide an optimization of the checks for
potentially variability.

Special Expression Classes
--------------------------

The following classes are *exceptions* to the design principles describe above.

Named Expressions
~~~~~~~~~~~~~~~~~

Named expressions allow for changes to an expression after it has
been constructed.  For example, consider the expression ``f`` defined
with the :class:`Expression <pyomo.core.base.Expression>` component:

.. literalinclude:: ../../tests/expr/design_named_expression.spy

Although ``f`` is an immutable expression, whose definition is
fixed, a sub-expressions is the named expression ``M.e``.  Named
expressions have a mutable value.  In other words, the expression
that they point to can change.  Thus, a change to the value of
``M.e`` changes the expression tree for any expression that includes
the named expression.

.. note::

    The named expression classes are not implemented as sub-classes
    of :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>`.
    This reflects design constraints related to the fact that these
    are modeling components that belong to class hierarchies other
    than the expression class hierarchy, and Pyomo's design prohibits
    the use of multiple inheritance for these classes.

Linear Expressions
~~~~~~~~~~~~~~~~~~

Pyomo includes a special expression class for linear expressions.
The class :class:`LinearExpression
<pyomo.core.expr.current.LinearExpression>` provides a compact
description of linear polynomials.  Specifically, it includes a
constant value :attr:`constant` and two lists for coefficients and
variables: :attr:`linear_coefs` and :attr:`linear_vars`.

This expression object does not have arguments, and thus it is
treated as a leaf node by Pyomo visitor classes.  Further, the
expression API functions described above do not work with this
class.  Thus, developers need to treat this class differently when
walking an expression tree (e.g. when developing a problem
transformation).

Sum Expressions
~~~~~~~~~~~~~~~

Pyomo does not have a binary sum expression class.  Instead,
it has an ``n``-ary summation class, :class:`SumExpression
<pyomo.core.expr.current.SumExpression>`.  This expression class
treats sums as ``n``-ary sums for efficiency reasons;  many large
optimization models contain large sums. But note tht this class
maintains the immutability property described above.  This class
shares an underlying list of arguments with other :class:`SumExpression
<pyomo.core.expr.current.SumExpression>` objects. A particular
object owns the first ``n`` arguments in the shared list, but
different objects may have different values of ``n``.

This class acts like a normal immutable expression class, and the
API described above works normally.  But direct access to the shared
list could have unexpected results.

Mutable Expressions
~~~~~~~~~~~~~~~~~~~

Finally, Pyomo includes several **mutable** expression classes
that are private.  These are not intended to be used by users, but
they might be useful for developers in contexts where the developer
can appropriately control how the classes are used.  Specifically,
immutability eliminates side-effects where changes to a sub-expression
unexpectedly create changes to the expression tree.  But within the context of
model transformations, developers may be able to limit the use of
expressions to avoid these side-effects.  The following mutable private classes
are available in Pyomo:

:class:`_MutableSumExpression <pyomo.core.expr.current._MutableSumExpression>` 
    This class
    is used in the :data:`nonlinear_expression <pyomo.core.expr.current.nonlinear_expression>` context manager to
    efficiently combine sums of nonlinear terms.
:class:`_MutableLinearExpression <pyomo.core.expr.current._MutableLinearExpression>` 
    This class
    is used in the :data:`linear_expression <pyomo.core.expr.current.linear_expression>` context manager to
    efficiently combine sums of linear terms.



Expression Semantics
--------------------

Pyomo clear semantics regarding what is considered a valid leaf and
interior node.

The following classes are valid interior nodes:

* Subclasses of :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>`

* Classes that that are *duck typed* to match the API of the :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>` class.  For example, the named expression class :class:`Expression <pyomo.core.expr.current.Expression>`.

The following classes are valid leaf nodes:

* Members of :data:`nonpyomo_leaf_types <pyomo.core.expr.numvalue.nonpyomo_leaf_types>`, which includes standard numeric data types like :const:`int`, :const:`float` and :const:`long`, as well as numeric data types defined by `numpy` and other commonly used packages.  This set also includes :class:`NonNumericValue <pyomo.core.expr.numvalue.NonNumericValue>`, which is used to wrap non-numeric arguments to the :class:`ExternalFunctionExpression <pyomo.core.expr.current.current.ExternalFunctionExpression>` class.

* Parameter component classes like :class:`SimpleParam <pyomo.core.base.param.SimpleParam>` and :class:`_ParamData <pyomo.core.base.param._ParamData>`, which arise in expression trees when the parameters are declared as mutable.  (Immutable parameters are identified when generating expressions, and they are replaced with their associated numeric value.)

* Variable component classes like :class:`SimpleVar <pyomo.core.base.var.SimpleVar>` and :class:`_GeneralVarData <pyomo.core.base.var._GeneralVarData>`, which often arise in expression trees.  <pyomo.core.expr.current.pyomo5_variable_types>`.

.. note::

    In some contexts the :class:`LinearExpression
    <pyomo.core.expr.current.LinearExpression>` class can be treated
    as an interior node, and sometimes it can be treated as a leaf.
    This expression object does not have any child arguments, so
    ``nargs()`` is zero.  But this expression references variables
    and parameters in a linear expression, so in that sense it does
    not represent a leaf node in the tree.



Context Managers
----------------

Pyomo defines several context managers that can be used to declare
the form of expressions, and to define a mutable expression object that
efficiently manages sums.

The :data:`linear_expression <pyomo.core.expr.current.linear_expression>` 
object is a context manager that can be used to declare a linear sum.  For
example, consider the following two loops:

.. literalinclude:: ../../tests/expr/design_cm1.spy

The first apparent difference in these loops is that the value of
``s`` is explicitly initialized while ``e`` is initialized when the
context manager is entered.  However, a more fundamental difference
is that the expression representation for ``s`` differs from ``e``.
Each term added to ``s`` results in a new, immutable expression.
By contrast, the context manager creates a mutable expression
representation for ``e``.  This difference allows for both (a) a
more efficient processing of each sum, and (b) a more compact
representation for the expression.

The difference between :data:`linear_expression
<pyomo.core.expr.current.linear_expression>` and
:data:`nonlinear_expression <pyomo.core.expr.current.nonlinear_expression>`
is the underlying representation that each supports.  Note that
both of these are instances of context manager classes.  In
singled-threaded applications, these objects can be safely used to
construct different expressions with different context declarations.

Finally, note that these context managers can be passed into the :attr:`start`
method for the :func:`quicksum <pyomo.core.util.quicksum>` function.  For example:

.. literalinclude:: ../../tests/expr/design_cm2.spy

This sum contains terms for ``M.x[i]`` and ``M.y[i]``.  The syntax
in this example is not intuitive because the sum is being stored
in ``e``.

.. note::

    We do not generally expect users or developers to use these
    context managers.  They are used by the :func:`quicksum
    <pyomo.core.util.quicksum>` and :func:`sum_product
    <pyomo.core.util.sum_product>` functions to accelerate expression
    generation, and there are few cases where the direct use of
    these context managers would provide additional utility to users
    and developers.

