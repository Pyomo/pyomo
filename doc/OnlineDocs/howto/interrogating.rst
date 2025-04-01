Interrogating Models
====================

.. _VarAccess:

Accessing Variable Values
-------------------------

Primal Variable Values
^^^^^^^^^^^^^^^^^^^^^^

Often, the point of optimization is to get optimal values of
variables. Some users may want to process the values in a script. We
will describe how to access a particular variable from a Python script
as well as how to access all variables from a Python script and from a
callback. This should enable the reader to understand how to get the
access that they desire. The Iterative example given above also
illustrates access to variable values.

One Variable from a Python Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming the model has been instantiated and solved and the results have
been loaded back into the instance object, then we can make use of the
fact that the variable is a member of the instance object and its value
can be accessed using its ``value`` member. For example, suppose the
model contains a variable named ``quant`` that is a singleton (has no
indexes) and suppose further that the name of the instance object is
``instance``. Then the value of this variable can be accessed using
``pyo.value(instance.quant)``. Variables with indexes can be referenced
by supplying the index.

Consider the following very simple example, which is similar to the
iterative example. This is a concrete model. In this example, the value
of ``x[2]`` is accessed.

.. literalinclude:: /src/scripting/noiteration1.py
   :language: python

.. note::

   If this script is run without modification, Pyomo is likely to issue
   a warning because there are no constraints. The warning is because
   some solvers may fail if given a problem instance that does not have
   any constraints.

All Variables from a Python Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with one variable, we assume that the model has been instantiated
and solved. Assuming the instance object has the name ``instance``,
the following code snippet displays all variables and their values:

   >>> for v in instance.component_objects(pyo.Var, active=True):
   ...     print("Variable",v)  # doctest: +SKIP
   ...     for index in v:
   ...         print ("   ",index, pyo.value(v[index]))  # doctest: +SKIP


Alternatively,

   >>> for v in instance.component_data_objects(pyo.Var, active=True):
   ...     print(v, pyo.value(v))  # doctest: +SKIP

This code could be improved by checking to see if the variable is not
indexed (i.e., the only index value is ``None``), then the code could
print the value without the word ``None`` next to it.

Assuming again that the model has been instantiated and solved and the
results have been loaded back into the instance object. Here is a code
snippet for fixing all integers at their current value:

    >>> for var in instance.component_data_objects(pyo.Var, active=True):
    ...     if not var.is_continuous():
    ...         print ("fixing "+str(v))  # doctest: +SKIP
    ...         var.fixed = True # fix the current value


Another way to access all of the variables (particularly if there are
blocks) is as follows (this particular snippet assumes that instead of
`import pyomo.environ as pyo` `from pyo.environ import *` was used):

.. literalinclude:: /src/scripting/block_iter_example_compprintloop.spy
   :language: python

.. _ParamAccess:

Accessing Parameter Values
--------------------------

Accessing parameter values is completely analogous to accessing variable
values. For example, here is a code snippet to print the name and value
of every Parameter in a model:

   >>> for parmobject in instance.component_objects(pyo.Param, active=True):
   ...     nametoprint = str(str(parmobject.name))
   ...     print ("Parameter ", nametoprint)  # doctest: +SKIP
   ...     for index in parmobject:
   ...         vtoprint = pyo.value(parmobject[index])
   ...         print ("   ",index, vtoprint)  # doctest: +SKIP


Accessing Duals
---------------

Access to dual values in scripts is similar to accessing primal variable
values, except that dual values are not captured by default so
additional directives are needed before optimization to signal that
duals are desired.

To get duals without a script, use the ``pyomo`` option
``--solver-suffixes='dual'`` which will cause dual values to be included
in output.  Note: In addition to duals (``dual``) , reduced costs
(``rc``) and slack values (``slack``) can be requested. All suffixes can
be requested using the ``pyomo`` option ``--solver-suffixes='.*'``

.. warning::

   Some of the duals may have the value ``None``, rather than ``0``.

Access Duals in a Python Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To signal that duals are desired, declare a Suffix component with the
name "dual" on the model or instance with an IMPORT or IMPORT_EXPORT
direction.

.. literalinclude:: /src/scripting/driveabs2_Create_dual_suffix_component.spy
   :language: python

See the section on Suffixes :ref:`Suffixes` for more information on
Pyomo's Suffix component. After the results are obtained and loaded into
an instance, duals can be accessed in the following fashion.

.. literalinclude:: /src/scripting/driveabs2_Access_all_dual.spy
   :language: python

The following snippet will only work, of course, if there is a
constraint with the name ``AxbConstraint`` that has and index, which is
the string ``Film``.

.. literalinclude:: /src/scripting/driveabs2_Access_one_dual.spy
   :language: python

Here is a complete example that relies on the file ``abstract2.py`` to
provide the model and the file ``abstract2.dat`` to provide the
data. Note that the model in ``abstract2.py`` does contain a constraint
named ``AxbConstraint`` and ``abstract2.dat`` does specify an index for
it named ``Film``.

.. literalinclude:: /src/scripting/driveabs2.spy
   :language: python

Concrete models are slightly different because the model is the
instance. Here is a complete example that relies on the file
``concrete1.py`` to provide the model and instantiate it.

.. literalinclude:: /src/scripting/driveconc1.py
   :language: python

Accessing Slacks
----------------

The functions ``lslack()`` and ``uslack()`` return the upper and lower
slacks, respectively, for a constraint.

