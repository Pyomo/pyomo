Objectives
==========

An objective is a function of variables that returns a value that an
optimization package attempts to maximize or minimize. The ``Objective``
function in Pyomo declares an objective. Although other mechanisms are
possible, this function is typically passed the name of another function
that gives the expression. Here is a very simple version of such a
function that assumes ``model.x`` has previously been declared as a
``Var``:

.. literalinclude:: ../script_spy_files/spy4Objectives_Objective_function_expression.spy
   :language: python

It is more common for an objective function to refer to parameters as in
this example that assumes that ``model.p`` has been declared as a
``Param`` and that ``model.x`` has been declared with the same index
set, while ``model.y`` has been declared as a singleton:

.. literalinclude:: ../script_spy_files/spy4Objectives_Objective_refer_parameters.spy
   :language: python

This example uses the ``sense`` option to specify maximization. The
default sense is ``minimize``.
