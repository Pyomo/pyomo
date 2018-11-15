Parameters
==========

The word "parameters" is used in many settings. When discussing a
Pyomo model, we use the word to refer to data that must be provided in
order to find an optimal (or good) assignment of values to the decision variables.
Parameters are declared with the ``Param`` function, which takes arguments
that are somewhat similar to the ``Set`` function. For example, the following code snippet declares sets
``model.A``, ``model.B`` and then a parameter array ``model.P`` that is indexed by ``model.A``:

.. literalinclude:: spyfiles/spy4Parameters_ABPSets.spy
   :language: python

In addition to sets that serve as indexes, the ``Param`` function takes
the following command options:

.. autosummary::
   :nosignatures:

- default = The value absent any other specification.
- doc = String describing the parameter
- initialize = A function (or Python object) that returns the members to initialize the parameter values.
- validate = A boolean function with arguments that are the prospective parameter value, the parameter indices and the model.
- within = Set used for validation; it specifies the domain of the parameter values.

These options perform in the same way as they do for ``Set``. For example,
suppose that ``Model.A = RangeSet(1,3)``, then there are many ways to create a parameter that is a square matrix with 9, 16, 25 on the main diagonal zeros elsewhere, here are two ways to do it. First using a Python object to initialize:

.. literalinclude:: spyfiles/spy4Parameters_Param_python.spy
   :language: python

And now using an initialization function that is automatically called once for
each index tuple (remember that we are assuming that ``model.A`` contains
1,2,3)

.. literalinclude:: spyfiles/spy4Parameters_Param_def.spy
   :language: python

In this example, the index set contained integers, but index sets need not be numeric. It is very common to use strings.

NOTE: Data specified in an input file will override the data specified by the initialize options.

Parameter values can be checked by a validation function. In the following example, the parameter S indexed by ``model.A``
and is checked to be greater than 3.14159. If a value is provided that is less than that, the model instantation would be terminated
and an error message issued. The function used to validate should be written so as to return ``True`` if the data is valid
and ``False`` otherwise.

.. literalinclude:: spyfiles/spy4Parameters_Valuecheck_validation.spy
   :language: python
