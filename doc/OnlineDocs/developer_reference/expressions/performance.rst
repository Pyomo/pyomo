.. |p| raw:: html

   <p />

Building Expressions Faster
===========================

Expression Generation
---------------------

Pyomo expressions can be constructed using native binary operators
in Python.  For example, a sum can be created in a simple loop:

.. doctest::

    >>>M = ConcreteModel()
    >>>M.x = Var(range(5))

    >>>s = 0
    >>>for i in range(5):
    >>>     s = s + M.x[i]

Additionally, Pyomo expressions can be constructed using functions
that iteratively apply Python binary operators.  For example, the
Python :func:`sum` function can be used to replace the previous
loop:

.. doctest::

    >>>s = sum(M.x[i] for i in range(5))

The :func:`sum` function is both more compact and more efficient.
Using :func:`sum` avoids the creation of temporary variables, and
the summation logic is executed in the Python interpreter while the
loop is interpreted.


Linear, Quadratic and General Nonlinear Expressions
---------------------------------------------------

Pyomo can express a very wide range of algebraic expressions, and
there are three general classes of expressions that are recognized
by Pyomo:

 * **linear polynomials**
 * **quadratic polynomials**
 * **nonlinear expressions**, including higher-order polynomials and
   expressions with intrinsic functions

These classes of expressions are leveraged to efficiently generate
compact representations of expressions, and to transform expression
trees into standard forms used to interface with solvers.  There
are clear distinctions between (a) linear and quadratic polynomials,
and (b) linear polynomials and nonlinear expressions.  However, the
not all quadratic polynomials are recognized;  in other words, many
quadratic expressions are treated as nonlinear expressions.

For example, consider the following quadratic polynomial:

.. doctest::

    >>>s = sum(M.x[i] for i in range(5))**2

This quadratic polynomial is treated as a nonlinear expression
unless the expression is explicilty processed to identify quadratic
terms.  This *lazy* identification of of quadratic terms allows
Pyomo to tailor the search for quadratic terms only when they are
explicitly needed.

In fact, Pyomo also identifies linear terms in a lazy manner.  But
the utility functions and context managers discussed below allow
the user to declare that an expression is linear, which allows for
more efficient processing of expressions as well as a more compact
representation of the linear polynomial.


Pyomo Utility Functions
-----------------------

Pyomo includes several similar functions that can be used to 
create expressions:

:func:`prod <pyomo.core.util.prod>` 
    A function to compute a product of Pyomo expressions.

:func:`Sum <pyomo.core.util.Sum>` 
    A function to efficiently compute a sum of Pyomo expressions.

:func:`summation <pyomo.core.util.summation>`
    A function that computes a generalized dot product.

prod
~~~~

The :func:`prod <pyomo.core.util.prod>` function is analogous to the builtin
:func:`sum` function.  Its main argument is a variable length
argument list, :attr:`args`, which represents expressions that are multiplied
together.  For example:

.. doctest::

    >>>M = ConcreteModel()
    >>>M.x = Var(range(5))
    >>>M.y = Var()

    The product M.x[0] * M.x[1] * ... * M.x[4]
    >>>prod(M.x)

    The product M.x[0]*M.z
    >>>prod(M.x[0], M.z)

    The product M.z*(M.x[0] + ... + M.x[4])
    >>>prod(sum(M.x), M.z)

Sum
~~~

The behavior of the :func:`Sum <pyomo.core.util.Sum>` function is
similar to the builtin :func:`sum` function, but this function often
generates a more compact Pyomo expression. Its main argument is a
variable length argument list, :attr:`args`, which represents
expressions that are summed together.  However, the summation is
customized based on the :attr:`start` and :attr:`linear` arguments.

The :attr:`start` defines the initial value for summation, which
defaults to zero.  If this value is not a numeric value, then the
:func:`Sum <pyomo.core.util.Sum>` sets the initial value to
:attr:`start` and executes a simple loop to sum the terms.  This
allows the sum to be stored in an object that is passed into
the function. (See the example using a context manager below.)

If :attr:`start` is a numeric value, then the :attr:`linear` argument
determines how the sum is processed:

* If :attr:`linear` is :const:`False`, then the terms in :attr:`args`
are assumed to be nonlinear.
* If :attr:`linear` is :const:`False`, then the terms in :attr:`args`
are assumed to be linear.
* If :attr:`linear` is :const:`None`, the first term in :attr:`args`
is analyze to determine whether the terms are linear or nonlinear.

This allows the :func:`Sum <pyomo.core.util.Sum>` function to
customize the expression representation used, and specifically a
more compact representation is used for linear polynomials.

Altogether, the :func:`Sum <pyomo.core.util.Sum>` function is generally 
faster than the builtin :func:`sum` function, and it generates a more
compact representation for linear polynomials.

.. Warning::

    By default, :attr:`linear` is :const:`None`.  While this allows
    for efficient expression generation in normal cases, there are
    circumstances where the inspection of the first
    term in :attr:`args` is misleading.  Consider the following
    example:

    .. doctest::

        >>>M = ConcreteModel()
        >>>M.x = Var(range(5))

        >>>Sum(M.x[i]**2 if i > 0 else M.x[i] for i in range(5))

    The first term created by the generator is linear, but the
    subsequent terms are nonlinear.  Pyomo does not gracefully
    process these nonlinear terms, so the user must specify the
    correct value for :attr:`linear`.

summation
~~~~~~~~~

The :func:`summation <pyomo.core.util.summation>` function supports
a generalized dot product.  The :attr:`args` argument contains one
or more generators that are used to create terms in the summation.
If the :attr:`args` argument contains a single generator, then its
sequence of terms are summed together; the sum is equivalent to
calling :func:`Sum <pyomo.core.util.Sum>`.  If two or more generators are
provided, then the result is the summation of their terms multiplied
together.  For example:

.. doctest::

    >>>M = ConcreteModel()
    >>>M.z = RangeSet(5)
    >>>M.x = Var(range(10))
    >>>M.y = Var(range(10))

    Sum the elements of x
    >>>summation(M.x)

    Sum the product of elements in x and y
    >>>summation(M.x, M.y)

    Sum the product of elements in x and y, over the index set z
    >>>summation(M.x, M.y, index=M.z)

The :attr:`denom` argument specifies generators whose terms are in 
the denominator.  For example:

.. doctest::

    Sum the product of x_i/y_i
    >>>summation(M.x, denom=M.y)

    Sum the product of 1/(x_i*y_i)
    >>>summation(denom=(M.x, M.y))

The terms summed by this function are explicitly specified, so :func:`summation <pyomo.core.util.summation>` can identify whether the resulting expression
is linear, quadratic or nonlinear.  Consequently, this function is
typically faster than simple loops, and it generates compact representations
of expressions..

Finally, note that the :func:`dot_product <pyomo.core.util.dot_product>` function is an aliase for func:`summation <pyomo.core.util.summation>`.

Context Managers
----------------

Pyomo defines several context managers that can be used to declare
the form of expressions, and to define a mutable expression object that
efficiently manages sums.

The :data:`linear_expression <pyomo.core.expr.current.linear_expression>` 
object is a context manager that can be used to declare a linear sum.  For
example, consider the following two loops:

.. doctest::

    >>>M = ConcreteModel()
    >>>M.x = Var(range(5))

    >>>s = 0
    >>>for i in range(5):
    >>>     s += M.x[i]

    >>>with linear_expression as e:
    >>>     for i in range(5):
    >>>         e += M.x[i]

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
method for the :func:`Sum <pyomo.core.util.Sum>` function.  For example:

.. doctest::

    >>>M = ConcreteModel()
    >>>M.x = Var(range(5))
    >>>M.y = Var(range(5))

    >>>with linear_expression as e:
    >>>     Sum(M.x, start=e)
    >>>     Sum(M.y, start=e)

This sum contains terms for ``M.x[i]`` and ``M.y[i]``.  The syntax
in this example is not intuitive because the sum is being stored
in ``e``.

.. note::

    We do not generally expect users or developers to use these
    context managers.  They are used by the :func:`Sum
    <pyomo.core.util.Sum>` and :func:`summation
    <pyomo.core.util.summation>` functions to accelerate expression
    generation, and there are few cases where the direct use of
    these context managers would provide additional utility to users
    and developers.

