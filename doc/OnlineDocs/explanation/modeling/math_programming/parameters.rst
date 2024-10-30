Parameters
==========

.. currentmodule:: pyomo.environ
.. doctest::
   :hide:

   >>> import pyomo.environ as pyo
   >>> model = pyo.ConcreteModel()

The word "parameters" is used in many settings. When discussing a Pyomo
model, we use the word to refer to data that must be provided in order
to find an optimal (or good) assignment of values to the decision
variables.  Parameters are declared as instances of a :class:`Param`
class, which
takes arguments that are somewhat similar to the :class:`Set` class. For
example, the following code snippet declares sets ``model.A`` and
``model.B``, and then a parameter ``model.P`` that is indexed by
``model.A`` and ``model.B``:

.. testcode::

   model.A = pyo.RangeSet(1,3)
   model.B = pyo.Set()
   model.P = pyo.Param(model.A, model.B)

In addition to sets that serve as indexes, :class:`Param` takes
the following options:

- ``default`` = The parameter value absent any other specification.
- ``doc`` = A string describing the parameter.
- ``initialize`` = A function (or Python object) that returns data used to
  initialize the parameter values.
- ``mutable`` = Boolean value indicating if the Param values are allowed
  to change after the Param is initialized.
- ``validate`` = A callback function that takes the model, proposed
  value, and indices of the proposed value; returning ``True`` if the value
  is valid.  Returning ``False`` will generate an exception.
- ``within`` = Set used for validation; it specifies the domain of 
  valid parameter values.

These options perform in the same way as they do for :class:`Set`. For
example, given ``model.A`` with values ``{1, 2, 3}``, then there are many
ways to create a parameter that represents a square matrix with 9, 16, 25 on the
main diagonal and zeros elsewhere, here are two ways to do it. First using a
Python object to initialize:

.. testcode::

   v={}
   v[1,1] = 9
   v[2,2] = 16
   v[3,3] = 25
   model.S1 = pyo.Param(model.A, model.A, initialize=v, default=0)

And now using an initialization function that is automatically called
once for each index tuple (remember that we are assuming that
``model.A`` contains ``{1, 2, 3}``)

.. testcode::

   def s_init(model, i, j):
       if i == j:
           return i*i
       else:
           return 0.0
   model.S2 = pyo.Param(model.A, model.A, initialize=s_init)

In this example, the index set contained integers, but index sets need
not be numeric. It is very common to use strings.

.. note::

   Data specified in an input file will override the data specified by
   the ``initialize`` option.

Parameter values can be checked by a validation function. In the
following example, the every value of the parameter ``T`` (indexed by
``model.A``) is checked
to be greater than 3.14159. If a value is provided that is less than
that, the model instantiation will be terminated and an error message
issued. The validation function should be written so as to return
``True`` if the data is valid and ``False`` otherwise.

.. testcode::

   t_data = {1: 10, 2: 3, 3: 20}

   def t_validate(model, v, i):
       return v > 3.14159

   model.T = pyo.Param(model.A, validate=t_validate, initialize=t_data)

This example will prodice the following error, indicating that the value
provided for ``T[2]`` failed validation:

.. testoutput::

   Traceback (most recent call last):
     ...
   ValueError: Invalid parameter value: T[2] = '3', value type=<class 'int'>.
       Value failed parameter validation rule
