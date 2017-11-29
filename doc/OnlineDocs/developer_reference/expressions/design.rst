.. |p| raw:: html

   <p />

Design Details
==============

.. warning::
    Unfortunately, Pyomo expression trees are not composed of Python
    objects from a single class hierarchy.  There are fundamental
    design constraints (referenced below) that impact the form of
    Pyomo expression trees.

Most Pyomo expression trees have the following form:

#. Interior nodes are objects that inherit from the :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>` class.

#. Leaf nodes are numeric values, parameter components and variable components, which represent the *inputs* to the expresion.

Expression Classes
------------------

Expression classes typically represent unary and binary operations.  The following table 
describes the standard operators in Python and their associated Pyomo expression class:

========== ============= =============================================================================
Operation  Python Syntax Pyomo Class
========== ============= =============================================================================
sum        ``x + y``     :class:`ViewSumExpression <pyomo.core.expr.current.ViewSumExpression>`
product    ``x * y``     :class:`ProductExpression <pyomo.core.expr.current.ProductExpression>`
negation   ``- x``       :class:`NegationExpression <pyomo.core.expr.current.NegationExpression>`
reciprocal ``1 / x``     :class:`ReciprocalExpression <pyomo.core.expr.current.ReciprocalExpression>`
power      ``x ** y``    :class:`PowerExpression <pyomo.core.expr.current.PowerExpression>`
inequality ``x <= y``    :class:`InequalityExpression <pyomo.core.expr.current.InequalityExpression>`
equality   ``x == y``    :class:`EqualityExpression <pyomo.core.expr.current.EqualityExpression>`
========== ============= =============================================================================

Additionally, there are a variety of other Pyomo expression classes that capture more general 
logical relationships, which are summarized in the following table:

==================== ====================================   ========================================================================================
Operation            Example                                Pyomo Class
==================== ====================================   ========================================================================================
exernal function     ``myfunc(x,y,z)``                      :class:`ExternalFunctionExpression <pyomo.core.expr.current.ExternalFunctionExpression>`
logical if-then-else ``Expr_if(IF_=x, THEN_=y, ELSE_=z)``   :class:`Expr_if <pyomo.core.expr.current.Expr_if>`
intrinsic function   ``sin(x)``                             :class:`UnaryFunctionExpression <pyomo.core.expr.current.UnaryFunctionExpression>`
absolute function    ``abs(x)``                             :class:`AbsExpression <pyomo.core.expr.current.AbsExpression>`
==================== ====================================   ========================================================================================

Expressions objects are immutable.  Specifically, the list of
arguments to an expression object (a.k.a. the list of child nodes
in the tree) cannot be changed after an expression class is
constructed.  To enforce this property, expression objects have a
standard API for accessing expression arguments:

* :attr:`args` - a class property that returns a generator that yields the expression arguments
* :attr:`arg(i)` - a function that returns the ``i``-th argument
* :attr:`nargs()` - a function that returns the number of expression arguments

.. warning::

    Developers should never use the :attr:`_args_` property directly!  The semantics for
    the use of this data has changed.  For example, in some expression classes the
    the value :func:`nargs()` may not equal :const:`len(_args_)`!

Expression trees can be categorized in four different ways:

* constant expressions - expressions that do not contain numeric constants and immutable parameters.
* mutable expressions - expressions that contain mutable parameters but no variables.
* potentially variable expressions - expressions that contain variables, which may be fixed.
* fixed expressions - expressions that contain variables, all of which are fixed.

These three categories are illustrated with the following example::

    m = ConcreteModel()
    m.p = Param(default=10, mutable=False)
    m.q = Param(default=10, mutable=True)
    m.x = Var()
    m.y = Var(initialize=1)
    m.y.fixed = True

The following table describes four diffrent simple expressions,
which consist of a single model component, and it shows how they
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
custom expression classes for expression trees that are not potentially
variable.  These custom classes will not normally be used by
developers, but they provide an optimization of the checks for
potentially variability.

Special Expression Classes
--------------------------

The following classes are *exceptions* to some of the design principles describe above.

Named Expressions
~~~~~~~~~~~~~~~~~

Pyomo includes several classes that *named expressions*, which allow for flexible changes to 
an expression after it has been constructed.  For example, consider the expression ``f`` defined
with the :class:`Expression <pyomo.core.base.Expression>` compoennt::

    M = ConcreteModel()
    M.v = Var()
    M.w = Var()

    M.e = Expression(expr=2*M.v)
    f = M.e + 3                     # f == 2*v + 3
    M.e += M.w                      # f == 2*v + 3 + w

Although ``f`` is an immutable expression, whose definition is
fixed, a sub-expressions is the named expression ``M.e``.  Named
expressions have a mutable value.  In other words, the expression
that they point to can change.  Thus, a change to the value of
``M.e`` changes the expression tree for any expression that includes
the named expression.

.. note::

    The named expression classes are not currently implemented as
    sub-classes of :class:`ExpressionBase
    <pyomo.core.expr.current.ExpressionBase>`.  This reflects design
    constraints related to the fact that these are modeling components
    that belong to class hierarchies other than the expression class
    hierarchy, and Pyomo uses code optimizations that prohibit
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

ViewSum Expressions
~~~~~~~~~~~~~~~~~~~

Pyomo does not have a *normal* binary sum expression class.  Instead,
it has an ``n``-ary summation class, :class:`ViewSumExpression
<pyomo.core.expr.current.ViewSumExpression>`.  This expression class
treats sums as ``n``-ary sums for efficiency reasons;  many large
optimization models contain large sums. But note tht this class
maintains the immutability property described above.  This class
shares an underlying list of arguments with other :class:`ViewSumExpression
<pyomo.core.expr.current.ViewSumExpression>` objects. A particular
object owns the first ``n`` arguments in the shared list, but
different objects may have different values of ``n``.

This class acts like a normal immutable expression class, and the
API described above works fine.  But direct access to the shared
list could have unexpected results.

Mutable Expressions
~~~~~~~~~~~~~~~~~~~

Finally, Pyomo includes several **mutable** experession classes
that are private.  These are not intended to be used by users, but
they might be useful for developers in contexts where the developer
can appropriately control how the classes are used.  Specifically,
immutability eliminates side-effects where changes to a sub-expression
unexpectedly create changes to the expression tree.  But within the context of
model transformations, developers may be able to limit the use of
expressions to avoid these side-effects.  The following mutable private classes
are available in Pyomo:

:class:`_MutableViewSumExpression <pyomo.core.expr.current._MutableViewSumExpression>` 
    This class
    is used in the :data:`nonlinear_expression <pyomo.core.expr.current.nonlinear_expression>` context manager to
    efficiently combine sums of nonlinear terms.
:class:`_MutableLinearExpression <pyomo.core.expr.current._MutableLinearExpression>` 
    This class
    is used in the :data:`linear_expression <pyomo.core.expr.current.linear_expression>` context manager to
    efficiently combine sums of linear terms.



Expression Leaves
-----------------

Unfortunately, Pyomo has weak semantics regarding what is considered
a valid leaf node.  As noted earlier, the following data and classes
are commonly included as leaf nodes in Pyomo expressions:

:data:`native_numeric_types <pyomo.core.expr.numvalue>`
    Standard numeric data types like :const:`int`, :const:`float`
    and :const:`long` are included in this class, as well as numeric
    data types defined by `numpy` and other commonly used packages.

parameter objects:
    Parameter component classes like :class:`SimpleParam
    <pyomo.core.base.param.SimpleParam>` and :class:`_ParamData
    <pyomo.core.base.param._ParamData>` may arise in expression
    trees, especially when the parameters are declared as mutable.
    (Immutable parameters are identified when generating expressions,
    and they are replaced with their associated numeric value.)

variable objects:
    Variable component classes like :class:`SimpleVar
    <pyomo.core.base.var.SimpleVar>` and :class:`_GeneralVarData
    <pyomo.core.base.var._GeneralVarData>` often arise in expression
    trees.  Pyomo defines a variety of variable types, which are
    included in the set :data:`pyomo5_variable_types
    <pyomo.core.expr.current.pyomo5_variable_types>`.

**However**, there are context where additional leaf node types can
arise.  Specifically, the :class:`ExternalFunctionExpression
<pyomo.core.expr.current.current.ExternalFunctionExpression>` class
can be defined with arbitrary function arguments.  Specifically,
constant arguments like tuples and strings may be natural, depending
on the nature of the external function.

