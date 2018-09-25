Variables
=========

Variables are intended to ultimately be given values by an optimization
package. They are declared and optionally bounded, given initial values,
and documented using the Pyomo ``Var`` function. If index sets are given
as arguments to this function they are used to index the variable. Other
optional directives include:

* bounds = A function (or Python object) that gives a (lower,upper)
  bound pair for the variable
* domain = A set that is a super-set of the values the variable can take
  on.
* initialize = A function (or Python object) that gives a starting value
  for the variable; this is particularly important for non-linear models
* within = (synonym for ``domain``)

The following code snippet illustrates some aspects of these options by
declaring a *singleton* (i.e. unindexed) variable named
``model.LumberJack`` that will take on real values between zero and 6
and it initialized to be 1.5:

.. literalinclude:: ../script_spy_files/spy4Variables_Declare_singleton_variable.spy
   :language: python

Instead of the ``initialize`` option, initialization is sometimes done
with a Python assignment statement as in

.. literalinclude:: ../script_spy_files/spy4Variables_Assign_value.spy
   :language: python

For indexed variables, bounds and initial values are often specified by
a rule (a Python function) that itself may make reference to parameters
or other data. The formal arguments to these rules begins with the model
followed by the indexes. This is illustrated in the following code
snippet that makes use of Python dictionaries declared as lb and ub that
are used by a function to provide bounds:

.. literalinclude:: ../script_spy_files/spy4Variables_Declare_bounds.spy
   :language: python

.. note::

   Many of the pre-defined virtual sets that are used as domains imply
   bounds. A strong example is the set ``Boolean`` that implies bounds
   of zero and one.
