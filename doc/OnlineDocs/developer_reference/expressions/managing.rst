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

.. doctest::

    class SizeofVisitor(SimpleExpressionVisitor):

        def __init__(self):
            self.counter = 0

        def visit(self, node):
            self.counter += 1

        def finalize(self):
            return self.counter

The class constructor creates a counter, and the :func:`visit` method 
increments this counter for every node that is visited.  The :func:`finalize`
method returns the value of this counter after the tree has been walked.  The
following function illustrates this use of this visitor class:

.. doctest::

    def sizeof_expression(expr):
        #
        # Create the visitor object
        #
        visitor = SizeofVisitor()
        #
        # Compute the value using the :func:`xbfs` search method.
        #
        return visitor.xbfs(expr)


ExpressionValueVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that clones the
expression tree (including leaf nodes).  Consider the following
class:

.. doctest::

    class CloneVisitor(ExpressionValueVisitor):

        def __init__(self):
            self.memo = {'__block_scope__': { id(None): False }}

        def visit(self, node, values):
            #
            # Clone the interior node
            #
            return node.construct_clone(tuple(values), self.memo)

        def visiting_potential_leaf(self, node):
            #
            # Clone leaf nodes in the expression tree
            #
            if node.__class__ in native_numeric_types or\
               node.__class__ not in pyomo5_expression_types:\
                return True, copy.deepcopy(node, self.memo)

            return False, None

The :func:`visit` method creates a new expression node with children
specified by :attr:`values`.  The :func:`visiting_potential_leaf`
method performs a :func:`deepcopy` on leaf nodes, which are native
Python types or non-expression objects.

.. doctest::

    def clone_expression(expr):
        #
        # Create the visitor object
        #
        visitor = CloneVisitor()
        #
        # Clone the expression using the :func:`dfs_postorder_stack` 
        # search method.
        #
        return visitor.dfs_postorder_stack(expr)


ExpressionReplacementVisitor Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we describe an visitor class that replaces
variables with scaled variables, using a mutable parameter that
can be modified later.  the following
class:

.. doctest::

    from pyomo.environ import *
    from pyomo.core.expr import current as EXPR

    class ScalingVisitor(EXPR.ExpressionReplacementVisitor):

        def __init__(self, scale):
            super(ScalingVisitor, self).__init__()
            self.scale = scale

        def visiting_potential_leaf(self, node):
            #
            # Clone leaf nodes in the expression tree
            #
            if node.__class__ in EXPR.pyomo5_variable_types:
                return True, self.scale[id(node)]*node

            if isinstance(node, EXPR._LinearExpression):
                node_ = copy.deepcopy(node)
                node_.constant = node.constant
                node_.linear_vars = copy.copy(node.linear_vars)
                node_.linear_coefs = []
                for i,v in enumerate(node.linear_vars):
                    node_.linear_coefs.append( node.linear_coefs[i]*self.scale[id(v)] )
                return True, node_

            return False, None

No :func:`visit` method needs to be defined.  The
:func:`visiting_potential_leaf` function identifies variable nodes
and returns a product expression that contains a mutable parameter.
The :class:`_LinearExpression` class has a different representation
that embeds variables.  Hence, this class must be handled 
in a separate condition that explicitly transforms this sub-expression.

.. doctest::

    def scale_expression(expr, scale):
        #
        # Create the visitor object
        #
        visitor = ScalingVisitor(scale)
        #
        # Scale the expression using the :func:`dfs_postorder_stack` 
        # search method.
        #
        return visitor.dfs_postorder_stack(expr)

The :func:`scale_expression` function is called with an expression and 
a dictionary, :attr:`scale`, that maps variable ID to model parameter.  For example:

.. doctest::

    M = ConcreteModel()
    M.x = Var(range(5))
    M.p = Param(range(5), mutable=True)

    scale={}
    for i in M.x:
      scale[id(M.x[i])] = M.p[i]

    e = Sum(M.x[i] for i in M.x)
    f = scale_expression(e,scale)
    print(f)
    # p[0]*x[0] + p[1]*x[1] + p[2]*x[2] + p[3]*x[3] + p[4]*x[4]

