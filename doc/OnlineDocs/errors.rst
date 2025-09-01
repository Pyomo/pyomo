Common Warnings/Errors
======================

..
   NOTE to developers: as we use section links to direct users, it is
   critical that the "IDs" are unique.  When adding a new extended
   warning / error description, DO NOT renumber existing entries.  Also,
   for backwards compatibility, DO NOT recycle old ID (no longer used)
   numbers.

.. testsetup::

   import pyomo.environ as pyo
   # Ensure that all logged messages are sent to stdout
   # (so they show up in the doctest output and can be tested)
   import pyomo.common.log as _log
   _log.pyomo_handler.__class__ = _log.StdoutHandler

.. py:currentmodule:: pyomo.environ


.. ===================================================================
.. Extended descriptions for Pyomo warnings
.. ===================================================================

Warnings
--------

.. _W1001:

W1001: Setting Var value not in domain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When setting :class:`Var` values (by either calling :meth:`Var.set_value()`
or setting the :attr:`value` attribute), Pyomo will validate the
incoming value by checking that the value is ``in`` the
:attr:`Var.domain`.  Any values not in the domain will generate this
warning:

.. doctest::

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(domain=pyo.Integers)
   >>> m.x = 0.5
   WARNING (W1001): Setting Var 'x' to a value `0.5` (float) not in domain
        Integers.
        See also https://pyomo.readthedocs.io/en/stable/errors.html#w1001
   >>> print(m.x.value)
   0.5


Users can bypass all domain validation by setting the value using:

.. doctest::

   >>> m.x.set_value(0.75, skip_validation=True)
   >>> print(m.x.value)
   0.75



.. _W1002:

W1002: Setting Var value outside the bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When setting :py:class:`Var` values (by either calling :meth:`set_value()`
or setting the :attr:`value` attribute), Pyomo will validate the
incoming value by checking that the value is within the range specified by
:attr:`Var.bounds`.  Any values outside the bounds will generate this
warning:

.. doctest::

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(domain=pyo.Integers, bounds=(1, 5))
   >>> m.x = 0
   WARNING (W1002): Setting Var 'x' to a numeric value `0` outside the bounds
       (1, 5).
       See also https://pyomo.readthedocs.io/en/stable/errors.html#w1002
   >>> print(m.x.value)
   0

Users can bypass all domain validation by setting the value using:

.. doctest::

   >>> m.x.set_value(10, skip_validation=True)
   >>> print(m.x.value)
   10



.. _W1003:

W1003: Unexpected RecursionError walking an expression tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyomo leverages a recursive walker (the
:py:class:`~pyomo.core.expr.visitor.StreamBasedExpressionVisitor`) to
traverse (walk) expression trees.  For most expressions, this recursive
walker is the most efficient.  However, Python has a relatively shallow
recursion limit (generally, 1000 frames).  The recursive walker is
designed to monitor the stack depth and cleanly switch to a nonrecursive
walker before hitting the stack limit.  However, there are two (rare)
cases where the Python stack limit can still generate a
:py:exc:`RecursionError` exception:

#. Starting the walker with fewer than
   :py:data:`pyomo.core.expr.visitor.RECURSION_LIMIT` available frames.
#. Callbacks that require more than 2 *
   :py:data:`pyomo.core.expr.visitor.RECURSION_LIMIT` frames.

The (default) recursive walker will catch the exception and restart the
walker from the beginning in non-recursive mode, issuing this warning.
The caution is that any partial work done by the walker before the
exception was raised will be lost, potentially leaving the walker in an
inconsistent state.  Users can avoid this by

- avoiding recursive callbacks
- restructuring the system design to avoid triggering the walker with
  few available stack frames
- directly calling the
  :py:meth:`~pyomo.core.expr.visitor.StreamBasedExpressionVisitor.walk_expression_nonrecursive()`
  walker method

.. doctest::
   :skipif: (on_github_actions and system_info[0].startswith('win')) \
            or system_info[2] == 'PyPy'

   >>> import sys
   >>> import pyomo.core.expr.visitor as visitor
   >>> from pyomo.core.tests.unit.test_visitor import fill_stack
   >>> expression_depth = visitor.StreamBasedExpressionVisitor(
   ...     exitNode=lambda node, data: max(data) + 1 if data else 1)
   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var()
   >>> @m.Expression(range(35))
   ... def e(m, i):
   ...     return m.e[i-1] if i else m.x
   >>> expression_depth.walk_expression(m.e[34])
   36
   >>> fill_stack(sys.getrecursionlimit() - visitor.get_stack_depth() - 30,
   ...            expression_depth.walk_expression,
   ...            m.e[34])
   WARNING (W1003): Unexpected RecursionError walking an expression tree.
       See also https://pyomo.readthedocs.io/en/stable/errors.html#w1003
   36
   >>> fill_stack(sys.getrecursionlimit() - visitor.get_stack_depth() - 30,
   ...            expression_depth.walk_expression_nonrecursive,
   ...            m.e[34])
   36


.. ===================================================================
.. Extended descriptions for Pyomo errors
.. ===================================================================

Errors
------

.. _E2001:

E2001: Variable domains must be an instance of a Pyomo Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variable domains are always Pyomo :class:`Set` or :class:`RangeSet`
objects.  This includes global sets like ``Reals``, ``Integers``,
``Binary``, ``NonNegativeReals``, etc., as well as model-specific
:class:`Set` instances.  The :attr:`Var.domain` setter will attempt to
convert assigned values to a Pyomo `Set`, with any failures leading to
this warning (and an exception from the converter):

.. doctest::

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var()
   >>> m.x.domain = 5
   Traceback (most recent call last):
      ...
   TypeError: Cannot create a Set from data that does not support __contains__...
   ERROR (E2001): 5 is not a valid domain. Variable domains must be an instance
       of a Pyomo Set or convertible to a Pyomo Set.
       See also https://pyomo.readthedocs.io/en/stable/errors.html#e2001



.. ===================================================================
.. Extended descriptions for Pyomo exceptions
.. ===================================================================

.. Exceptions
.. ----------

.. .. _X101:
