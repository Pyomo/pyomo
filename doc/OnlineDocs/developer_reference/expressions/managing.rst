.. |p| raw:: html

   <p />

Managing Expressions
====================

Creating a String Representation of an Expression
-------------------------------------------------

There are several ways that string representations can be created
from an expression, but the :func:`expression_to_string
<pyomo.core.expr.current.expression_to_string>` function provides
the most flexible mechanism for generating a string representation.
The options to this function control distinct aspects of the string
representation.

Algebraic vs. Nested Functional Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default string representation is an algebraic form, which closely
mimics the Python operations used to construct an expression.  The
:data:`verbose` flag can be set to :const:`True` to generate a
string representation that is a nested functional form.  For example:

.. literalinclude:: ../../tests/expr/managing_ex1.spy

Labeler and Symbol Map
~~~~~~~~~~~~~~~~~~~~~~

The string representation used for variables in expression can be customized to
define different label formats.  If the :data:`labeler` option is specified, then this
function (or class functor) is used to generate a string label used to represent the variable.  Pyomo
defines a variety of labelers in the `pyomo.core.base.label` module.  For example, the
:class:`NumericLabeler` defines a functor that can be used to sequentially generate
simple labels with a prefix followed by the variable count:

.. literalinclude:: ../../tests/expr/managing_ex2.spy

The :data:`smap` option is used to specify a symbol map object
(:class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>`), which
caches the variable label data.  This option is normally specified
in contexts where the string representations for many expressions
are being generated.  In that context, a symbol map ensures that
variables in different expressions have a consistent label in their
associated string representations.


Standardized String Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :data:`standardize` option can be used to re-order the string 
representation to print polynomial terms before nonlinear terms.  By
default, :data:`standardize` is :const:`False`, and the string
representation reflects the order in which terms were combined to
form the expression.  Pyomo does not guarantee that the string 
representation exactly matches the Python expression order, since
some simplification and re-ordering of terms is done automatically to
improve the efficiency of expression generation.  But in most cases
the string representation will closely correspond to the 
Python expression order.

If :data:`standardize` is :const:`True`, then the pyomo expression
is processed to identify polynomial terms, and the string representation
consists of the constant and linear terms followed by
an expression that contains other nonlinear terms.  For example:

.. literalinclude:: ../../tests/expr/managing_ex3.spy

Other Ways to Generate String Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two other standard ways to generate string representations:

* Call the :func:`__str__` magic method (e.g. using the Python :func:`str()` function.  This 
  calls :func:`expression_to_string <pyomo.core.expr.current.expression_to_string>` with
  the option :data:`standardize` equal to :const:`True` (see below).

* Call the :func:`to_string` method on the :class:`ExpressionBase <pyomo.core.expr.current.ExpressionBase>` class.
  This defaults to calling :func:`expression_to_string <pyomo.core.expr.current.expression_to_string>` with
  the option :data:`standardize` equal to :const:`False` (see below).

In practice, we expect at the :func:`__str__` magic method will be
used by most users, and the standardization of the output provides
a consistent ordering of terms that should make it easier to interpret
expressions.


Cloning Expressions
-------------------

Expressions are automatically cloned only during certain expression
transformations.  Since this can be an expensive operation, the
:data:`clone_counter <pyomo.core.expr.current.clone_counter>` context
manager object is provided to track the number of times the
:func:`clone_expression <pyomo.core.expr.current.clone_expression>`
function is executed.

For example:

.. literalinclude:: ../../tests/expr/managing_ex4.spy

Evaluating Expressions
----------------------

Expressions can be evaluated when all variables and parameters in
the expression have a value.  The :func:`value <pyomo.core.expr.value>`
function can be used to walk the expression tree and compute the
value of an expression.  For example:

.. literalinclude:: ../../tests/expr/managing_ex5.spy

Additionally, expressions define the :func:`__call__` method, so the
following is another way to compute the value of an expression:

.. literalinclude:: ../../tests/expr/managing_ex6.spy

If a parameter or variable is undefined, then the :func:`value
<pyomo.core.expr.value>` function and :func:`__call__` method will
raise an exception.  This exception can be suppressed using the
:attr:`exception` option.  For example:

.. literalinclude:: ../../tests/expr/managing_ex7.spy

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

.. literalinclude:: ../../tests/expr/managing_ex8.spy

The :func:`identify_variables <pyomo.core.expr.current.identify_variables>`
function is a generator function that yields all nodes that are
variables.  Pyomo uses several different classes to represent variables,
but this set of variable types does not need to be specified by the user.
However, the :attr:`include_fixed` flag can be specified to omit fixed
variables.  For example:

.. literalinclude:: ../../tests/expr/managing_ex9.spy

Walking an Expression Tree with a Visitor Class
-----------------------------------------------

Many of the utility functions defined above are implemented by
walking an expression tree and performing an operation at nodes in
the tree.  For example, evaluating an expression is performed using
a post-order depth-first search process where the value of a node
is computed using the values of its children.

Walking an expression tree can be tricky, and the code requires intimate
knowledge of the design of the expression system.  Pyomo includes
several classes that define so-called visitor patterns for walking
expression tree:

:class:`SimpleExpressionVisitor <pyomo.core.expr.current.SimpleExpressionVisitor>`
    A :func:`visitor` method is called for each node in the tree,
    and the visitor class collects information about the tree.

:class:`ExpressionValueVisitor <pyomo.core.expr.current.ExpressionValueVisitor>`
    When the :func:`visitor` method is called on each node in the
    tree, the *values* of its children have been computed.  The
    *value* of the node is returned from :func:`visitor`.

:class:`ExpressionReplacementVisitor <pyomo.core.expr.current.ExpressionReplacementVisitor>`
    When the :func:`visitor` method is called on each node in the
    tree, it may clone or otherwise replace the node using objects
    for its children (which themselves may be clones or replacements
    from the original child objects).  The new node object is
    returned from :func:`visitor`.

These classes define a variety of suitable tree search methods:

* :class:`SimpleExpressionVisitor <pyomo.core.expr.current.SimpleExpressionVisitor>`

  * **xbfs**: breadth-first search where leaf nodes are immediately visited
  * **xbfs_yield_leaves**: breadth-first search where leaf nodes are immediately visited, and the visit method yields a value

* :class:`ExpressionValueVisitor <pyomo.core.expr.current.ExpressionValueVisitor>`

  * **dfs_postorder_stack**: postorder depth-first search using a stack

* :class:`ExpressionReplacementVisitor <pyomo.core.expr.current.ExpressionReplacementVisitor>`

  * **dfs_postorder_stack**: postorder depth-first search using a stack

.. note::

    The PyUtilib visitor classes define several other search methods
    that could be used with Pyomo expressions.  But these are the 
    only search methods currently used within Pyomo.

To implement a visitor object, a user creates a subclass of one of these 
classes.  Only one of a few methods will need to be defined to
implement the visitor:

:func:`visitor`
    Defines the operation that is performed when a node is visited.  In
    the :class:`ExpressionValueVisitor <pyomo.core.expr.current.ExpressionValueVisitor>` and :class:`ExpressionReplacementVisitor <pyomo.core.expr.current.ExpressionReplacementVisitor>` visitor classes, this 
    method returns a value that is used by its parent node.

:func:`visiting_potential_leaf`
    Checks if the search should terminate with this node.  If no,
    then this method returns the tuple ``(False, None)``.  If yes,
    then this method returns ``(False, value)``, where *value* is
    computed by this method.  This method is not used in the
    :class:`SimpleExpressionVisitor
    <pyomo.core.expr.current.SimpleExpressionVisitor>` visitor
    class.

:func:`finalize`
    This method defines the final value that is returned from the 
    visitor.  This is not normally redefined.

Detailed documentation of the APIs for these methods is provided
with the class documentation for these visitors.

SimpleExpressionVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that counts the number
of nodes in an expression (including leaf nodes).  Consider the following
class:

.. literalinclude:: ../../tests/expr/managing_visitor1.spy

The class constructor creates a counter, and the :func:`visit` method 
increments this counter for every node that is visited.  The :func:`finalize`
method returns the value of this counter after the tree has been walked.  The
following function illustrates this use of this visitor class:

.. literalinclude:: ../../tests/expr/managing_visitor2.spy


ExpressionValueVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that clones the
expression tree (including leaf nodes).  Consider the following
class:

.. literalinclude:: ../../tests/expr/managing_visitor3.spy

The :func:`visit` method creates a new expression node with children
specified by :attr:`values`.  The :func:`visiting_potential_leaf`
method performs a :func:`deepcopy` on leaf nodes, which are native
Python types or non-expression objects.

.. literalinclude:: ../../tests/expr/managing_visitor4.spy


ExpressionReplacementVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that replaces
variables with scaled variables, using a mutable parameter that
can be modified later.  the following
class:

.. literalinclude:: ../../tests/expr/managing_visitor5.spy

No :func:`visit` method needs to be defined.  The
:func:`visiting_potential_leaf` function identifies variable nodes
and returns a product expression that contains a mutable parameter.
The :class:`_LinearExpression` class has a different representation
that embeds variables.  Hence, this class must be handled 
in a separate condition that explicitly transforms this sub-expression.

.. literalinclude:: ../../tests/expr/managing_visitor6.spy

The :func:`scale_expression` function is called with an expression and 
a dictionary, :attr:`scale`, that maps variable ID to model parameter.  For example:

.. literalinclude:: ../../tests/expr/managing_visitor7.spy
