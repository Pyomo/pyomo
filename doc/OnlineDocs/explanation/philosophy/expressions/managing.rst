.. |p| raw:: html

   <p />

Managing Expressions
====================

Creating a String Representation of an Expression
-------------------------------------------------

There are several ways that string representations can be created
from an expression, but the :func:`expression_to_string
<pyomo.core.expr.expression_to_string>` function provides
the most flexible mechanism for generating a string representation.
The options to this function control distinct aspects of the string
representation.

Algebraic vs. Nested Functional Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default string representation is an algebraic form, which closely
mimics the Python operations used to construct an expression.  The
:data:`verbose` flag can be set to :const:`True` to generate a
string representation that is a nested functional form.  For example:

.. literalinclude:: /src/expr/managing_ex1.spy

Labeler and Symbol Map
~~~~~~~~~~~~~~~~~~~~~~

The string representation used for variables in expression can be
customized to define different label formats.  If the :data:`labeler`
option is specified, then this function (or class functor) is used to
generate a string label used to represent the variable.  Pyomo defines a
variety of labelers in the `pyomo.core.base.label` module.  For example,
the :class:`NumericLabeler` defines a functor that can be used to
sequentially generate simple labels with a prefix followed by the
variable count:

.. literalinclude:: /src/expr/managing_ex2.spy

The :data:`smap` option is used to specify a symbol map object
(:class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>`), which
caches the variable label data.  This option is normally specified
in contexts where the string representations for many expressions
are being generated.  In that context, a symbol map ensures that
variables in different expressions have a consistent label in their
associated string representations.


Other Ways to Generate String Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two other standard ways to generate string representations:

* Call the :func:`__str__` magic method (e.g. using the Python
  :func:`str()` function.  This calls :func:`expression_to_string
  <pyomo.core.expr.expression_to_string>`, using the default values for
  all arguments.

* Call the :func:`to_string` method on the
  :class:`ExpressionBase<pyomo.core.expr.ExpressionBase>` class.  This
  calls :func:`expression_to_string
  <pyomo.core.expr.expression_to_string>` and accepts the same arguments.


Evaluating Expressions
----------------------

Expressions can be evaluated when all variables and parameters in
the expression have a value.  The :func:`value <pyomo.core.expr.value>`
function can be used to walk the expression tree and compute the
value of an expression.  For example:

.. literalinclude:: /src/expr/managing_ex5.spy

Additionally, expressions define the :func:`__call__` method, so the
following is another way to compute the value of an expression:

.. literalinclude:: /src/expr/managing_ex6.spy

If a parameter or variable is undefined, then the :func:`value
<pyomo.core.expr.value>` function and :func:`__call__` method will
raise an exception.  This exception can be suppressed using the
:attr:`exception` option.  For example:

.. literalinclude:: /src/expr/managing_ex7.spy

This option is useful in contexts where adding a try block is inconvenient
in your modeling script.

.. note::

    Both the :func:`value <pyomo.core.expr.value>` function and
    :func:`__call__` method call the :func:`evaluate_expression
    <pyomo.core.expr.evaluate_expression>` function.  In
    practice, this function will be slightly faster, but the
    difference is only meaningful when expressions are evaluated
    many times.

Identifying Components and Variables
------------------------------------

Expression transformations sometimes need to find all nodes in an
expression tree that are of a given type.  Pyomo contains two utility
functions that support this functionality.  First, the
:func:`identify_components <pyomo.core.expr.identify_components>`
function is a generator function that walks the expression tree and yields all
nodes whose type is in a specified set of node types.  For example:

.. literalinclude:: /src/expr/managing_ex8.spy

The :func:`identify_variables <pyomo.core.expr.identify_variables>`
function is a generator function that yields all nodes that are
variables.  Pyomo uses several different classes to represent variables,
but this set of variable types does not need to be specified by the user.
However, the :attr:`include_fixed` flag can be specified to omit fixed
variables.  For example:

.. literalinclude:: /src/expr/managing_ex9.spy

Walking an Expression Tree with a Visitor Class
-----------------------------------------------

Many of the utility functions defined above are implemented by
walking an expression tree and performing an operation at nodes in
the tree.  For example, evaluating an expression is performed using
a post-order depth-first search process where the value of a node
is computed using the values of its children.

Walking an expression tree can be tricky, and the code requires intimate
knowledge of the design of the expression system.  Pyomo includes
several classes that define visitor patterns for walking expression
tree:

:class:`StreamBasedExpressionVisitor <pyomo.core.expr.StreamBasedExpressionVisitor>`
    The most general and extensible visitor class.  This visitor
    implements an event-based approach for walking the tree inspired by
    the ``expat`` library for processing XML files.  The visitor has
    seven event callbacks that users can hook into, providing very
    fine-grained control over the expression walker.

:class:`ExpressionValueVisitor <pyomo.core.expr.ExpressionValueVisitor>`
    When the :func:`visitor` method is called on each node in the
    tree, the *values* of its children have been computed.  The
    *value* of the node is returned from :func:`visitor`.

:class:`ExpressionReplacementVisitor <pyomo.core.expr.ExpressionReplacementVisitor>`
    When the :func:`visitor` method is called on each node in the
    tree, it may clone or otherwise replace the node using objects
    for its children (which themselves may be clones or replacements
    from the original child objects).  The new node object is
    returned from :func:`visitor`.

These classes define a variety of suitable tree search methods:

* :class:`StreamBasedExpressionVisitor <pyomo.core.expr.StreamBasedExpressionVisitor>`

  * ``walk_expression``: depth-first traversal of the expression tree.

* :class:`ExpressionReplacementVisitor <pyomo.core.expr.ExpressionReplacementVisitor>`

  * ``walk_expression``: depth-first traversal of the expression tree.

* :class:`ExpressionValueVisitor <pyomo.core.expr.ExpressionValueVisitor>`

  * ``dfs_postorder_stack``: postorder depth-first search using a
    nonrecursive stack


To implement a visitor object, a user needs to provide specializations
for specific events.  For legacy visitors based on the PyUtilib visitor
pattern (e.g., :class:`ExpressionValueVisitor`), one must create a
subclass and override at least one of the following:

:func:`visit`
    Defines the operation that is performed when a node is visited.  In
    the :class:`ExpressionValueVisitor
    <pyomo.core.expr.ExpressionValueVisitor>` and
    :class:`ExpressionReplacementVisitor
    <pyomo.core.expr.ExpressionReplacementVisitor>` visitor classes,
    this method returns a value that is used by its parent node.

:func:`visiting_potential_leaf`
    Checks if the search should terminate with this node.  If no,
    then this method returns the tuple ``(False, None)``.  If yes,
    then this method returns ``(False, value)``, where *value* is
    computed by this method.

:func:`finalize`
    This method defines the final value that is returned from the
    visitor.  This is not normally redefined.

For modern visitors based on the :class:`StreamBasedExpressionVisitor
<pyomo.core.expr.StreamBasedExpressionVisitor>`, one can either define a
subclass, pass the callbacks to an instance of the base class, or assign
the callbacks as attributes on an instance of the base class.  The
:class:`StreamBasedExpressionVisitor
<pyomo.core.expr.StreamBasedExpressionVisitor>` provides seven
callbacks, which are documented in the class documentation.

Detailed documentation of the APIs for these methods is provided
with the class documentation for these visitors.

StreamBasedExpressionVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that counts the number
of nodes in an expression (including leaf nodes).  Consider the following
class:

.. literalinclude:: /src/expr/managing_visitor1.spy

The :func:`initializeWalker` method creates a counter, and the
:func:`exitNode` method increments this counter for every node that is
visited.  The :func:`finalizeResult` method returns the value of this
counter after the tree has been walked.  The following function
illustrates this use of this visitor class:

.. literalinclude:: /src/expr/managing_visitor2.spy


ExpressionValueVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that clones the
expression tree (including leaf nodes).  Consider the following
class:

.. literalinclude:: /src/expr/managing_visitor3.spy

The :func:`visit` method creates a new expression node with children
specified by :attr:`values`.  The :func:`visiting_potential_leaf`
method performs a :func:`deepcopy` on leaf nodes, which are native
Python types or non-expression objects.

.. literalinclude:: /src/expr/managing_visitor4.spy


ExpressionReplacementVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that replaces
variables with scaled variables, using a mutable parameter that
can be modified later.  the following
class:

.. literalinclude:: /src/expr/managing_visitor5.spy

No other method need to be defined.  The
:func:`beforeChild` method identifies variable nodes
and returns a product expression that contains a mutable parameter.

.. literalinclude:: /src/expr/managing_visitor6.spy

The :func:`scale_expression` function is called with an expression and
a dictionary, :attr:`scale`, that maps variable ID to model parameter.  For example:

.. literalinclude:: /src/expr/managing_visitor7.spy
