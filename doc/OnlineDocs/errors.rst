Common Warnings/Errors
======================

..
   NOTE to developers: as we use section links to direct users, it is
   critical that the "IDs" are unique.  When adding a new extended
   warning / error description, DO NOT renumber existing entries.  Also,
   for backwards compatibility, DO NOT recycle old ID (no longer used)
   numbers.

.. doctest::
   :hide:

   >>> import pyomo.environ as pyo

.. py:currentmodule:: pyomo.environ


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
       of a Pyomo Set or convertable to a Pyomo Set.
       See also https://pyomo.readthedocs.io/en/stable/errors.html#e2001


Exceptions
----------

.. _X101:
