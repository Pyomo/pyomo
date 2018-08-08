.. |p| raw:: html

   <p />

Building Expressions Faster
===========================

Expression Generation
---------------------

Pyomo expressions can be constructed using native binary operators
in Python.  For example, a sum can be created in a simple loop:

.. literalinclude:: ../../tests/expr/performance_loop1.spy

Additionally, Pyomo expressions can be constructed using functions
that iteratively apply Python binary operators.  For example, the
Python :func:`sum` function can be used to replace the previous
loop:

.. literalinclude:: ../../tests/expr/performance_loop2.spy

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
trees into standard forms used to interface with solvers.  Note
that There not all quadratic polynomials are recognized by Pyomo;
in other words, some quadratic expressions are treated as nonlinear
expressions.

For example, consider the following quadratic polynomial:

.. literalinclude:: ../../tests/expr/performance_loop3.spy

This quadratic polynomial is treated as a nonlinear expression
unless the expression is explicilty processed to identify quadratic
terms.  This *lazy* identification of of quadratic terms allows
Pyomo to tailor the search for quadratic terms only when they are
explicitly needed.

Pyomo Utility Functions
-----------------------

Pyomo includes several similar functions that can be used to 
create expressions:

:func:`prod <pyomo.core.util.prod>` 
    A function to compute a product of Pyomo expressions.

:func:`quicksum <pyomo.core.util.quicksum>` 
    A function to efficiently compute a sum of Pyomo expressions.

:func:`sum_product <pyomo.core.util.sum_product>`
    A function that computes a generalized dot product.

prod
~~~~

The :func:`prod <pyomo.core.util.prod>` function is analogous to the builtin
:func:`sum` function.  Its main argument is a variable length
argument list, :attr:`args`, which represents expressions that are multiplied
together.  For example:

.. literalinclude:: ../../tests/expr/performance_prod.spy

quicksum
~~~~~~~~

The behavior of the :func:`quicksum <pyomo.core.util.quicksum>` function is
similar to the builtin :func:`sum` function, but this function often
generates a more compact Pyomo expression. Its main argument is a
variable length argument list, :attr:`args`, which represents
expressions that are summed together.  For example:

.. literalinclude:: ../../tests/expr/performance_quicksum.spy

The summation is customized based on the :attr:`start` and
:attr:`linear` arguments.  The :attr:`start` defines the initial
value for summation, which defaults to zero.  If :attr:`start` is
a numeric value, then the :attr:`linear` argument determines how
the sum is processed:

* If :attr:`linear` is :const:`False`, then the terms in :attr:`args` are assumed to be nonlinear.
* If :attr:`linear` is :const:`True`, then the terms in :attr:`args` are assumed to be linear.
* If :attr:`linear` is :const:`None`, the first term in :attr:`args` is analyze to determine whether the terms are linear or nonlinear.

This argument allows the :func:`quicksum <pyomo.core.util.quicksum>`
function to customize the expression representation used, and
specifically a more compact representation is used for linear
polynomials.  The :func:`quicksum <pyomo.core.util.quicksum>`
function can be slower than the builtin :func:`sum` function,
but this compact representation can generate problem representations
more quickly.

Consider the following example:

.. literalinclude:: ../../tests/expr/quicksum_runtime.spy

The sum consists of linear terms because the exponents are one.
The following output illustrates that quicksum can identify this
linear structure to generate expressions more quickly:

.. literalinclude:: ../../tests/expr/quicksum.log
    :language: none

If :attr:`start` is not a numeric value, then the :func:`quicksum
<pyomo.core.util.quicksum>` sets the initial value to :attr:`start`
and executes a simple loop to sum the terms.  This allows the sum
to be stored in an object that is passed into the function (e.g. the linear context manager 
:data:`linear_expression <pyomo.core.expr.current.linear_expression>`).

.. Warning::

    By default, :attr:`linear` is :const:`None`.  While this allows
    for efficient expression generation in normal cases, there are
    circumstances where the inspection of the first
    term in :attr:`args` is misleading.  Consider the following
    example:

    .. literalinclude:: ../../tests/expr/performance_warning.spy

    The first term created by the generator is linear, but the
    subsequent terms are nonlinear.  Pyomo gracefully transitions
    to a nonlinear sum, but in this case :func:`quicksum <pyomo.core.util.quicksum>`
    is doing additional work that is not useful.

sum_product
~~~~~~~~~~~

The :func:`sum_product <pyomo.core.util.sum_product>` function supports
a generalized dot product.  The :attr:`args` argument contains one
or more components that are used to create terms in the summation.
If the :attr:`args` argument contains a single components, then its
sequence of terms are summed together; the sum is equivalent to
calling :func:`quicksum <pyomo.core.util.quicksum>`.  If two or more components are
provided, then the result is the summation of their terms multiplied
together.  For example:

.. literalinclude:: ../../tests/expr/performance_sum_product1.spy

The :attr:`denom` argument specifies components whose terms are in 
the denominator.  For example:

.. literalinclude:: ../../tests/expr/performance_sum_product2.spy

The terms summed by this function are explicitly specified, so
:func:`sum_product <pyomo.core.util.sum_product>` can identify
whether the resulting expression is linear, quadratic or nonlinear.
Consequently, this function is typically faster than simple loops,
and it generates compact representations of expressions..

Finally, note that the :func:`dot_product <pyomo.core.util.dot_product>`
function is an alias for :func:`sum_product <pyomo.core.util.sum_product>`.

