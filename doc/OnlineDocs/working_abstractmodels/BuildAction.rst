.. _BuildAction:

.. _abstract2piecebuild.py:

.. _Isinglebuild.py:

``BuildAction`` and ``BuildCheck``
==================================

This is a somewhat advanced topic. In some cases, it is desirable to
trigger actions to be done as part of the model building process. The
``BuildAction`` function provides this capability in a Pyomo model.  It
takes as arguments optional index sets and a function to peform the
action.  For example,

.. literalinclude:: ../script_spy_files/abstract2piecebuild_BuildAction_example.spy
   :language: python

calls the function ``bpts_build`` for each member of ``model.J``. The
function ``bpts_build`` should have the model and a variable for the
members of ``model.J`` as formal arguments. In this example, the
following would be a valid declaration for the function:

.. literalinclude:: ../script_spy_files/abstract2piecebuild_Function_valid_declaration.spy
   :language: python


A full example, which extends the :ref:`abstract2.py` and
:ref:`abstract2piece.py` examples, is

.. literalinclude:: ../script_spy_files/abstract2piecebuild.spy
   :language: python

This example uses the build action to create a model component with
breakpoints for a :ref:`piecewise` function.  The ``BuildAction`` is
triggered by the assignment to ``model.BuildBpts``. This object is not
referenced again, the only goal is to cause the execution of
``bpts_build,`` which places data in the ``model.bpts`` dictionary.
Note that if ``model.bpts`` had been a ``Set``, then it could have been
created with an ``initialize`` argument to the ``Set``
declaration. Since it is a special-purpose dictionary to support the
:ref:`piecewise` functionality in Pyomo, we use a ``BuildAction``.

Another application of ``BuildAction`` can be intialization of Pyomo
model data from Python data structures, or efficient initialization of
Pyomo model data from other Pyomo model data. Consider the
:ref:`Isinglecomm.py` example. Rather than using an initialization for
each list of sets ``NodesIn`` and ``NodesOut`` separately using
``initialize``, it is a little more efficient and probably a little
clearer, to use a build action.

The full model is:

.. literalinclude:: ../script_spy_files/Isinglebuild.py
   :language: python

for this model, the same data file can be used as for Isinglecomm.py in
:ref:`Isinglecomm.py` such as the toy data file:

.. literalinclude:: ../script_spy_files/Isinglecomm.dat

Build actions can also be a way to implement data validation,
particularly when multiple Sets or Parameters must be analyzed. However,
the the ``BuildCheck`` component is prefered for this purpose. It
executes its rule just like a ``BuildAction`` but will terminate the
construction of the model instance if the rule returns ``False``.
