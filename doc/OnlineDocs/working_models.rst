Working with Pyomo Models
=========================

This section gives an overview of commonly used scripting commands when
working with Pyomo models. These commands must be applied to a concrete
model instance or in other words an instantiated model.


Repeated Solves
---------------

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory
   >>> model = pyo.ConcreteModel()
   >>> model.nVars = pyo.Param(initialize=4)
   >>> model.N = pyo.RangeSet(model.nVars)
   >>> model.x = pyo.Var(model.N, within=pyo.Binary)
   >>> model.obj = pyo.Objective(expr=pyo.summation(model.x))
   >>> model.cuts = pyo.ConstraintList()
   >>> opt = SolverFactory('glpk')
   >>> opt.solve(model) # doctest: +SKIP

   >>> # Iterate, adding a cut to exclude the previously found solution
   >>> for i in range(5):
   ...    expr = 0
   ...    for j in model.x:
   ...        if pyo.value(model.x[j]) < 0.5:
   ...            expr += model.x[j]
   ...        else:
   ...            expr += (1 - model.x[j])
   ...    model.cuts.add( expr >= 1 )
   ...    results = opt.solve(model)
   ...    print ("\n===== iteration",i)
   ...    model.display() # doctest: +SKIP

To illustrate Python scripts for Pyomo we consider an example that is in
the file ``iterative1.py`` and is executed using the command

::

  python iterative1.py

.. note::

   This is a Python script that contains elements of Pyomo, so it is
   executed using the ``python`` command.  The ``pyomo`` command can be
   used, but then there will be some strange messages at the end when
   Pyomo finishes the script and attempts to send the results to a
   solver, which is what the ``pyomo`` command does.

This script creates a model, solves it, and then adds a constraint to
preclude the solution just found. This process is repeated, so the
script finds and prints multiple solutions.  The particular model it
creates is just the sum of four binary variables. One does not need a
computer to solve the problem or even to iterate over solutions. This
example is provided just to illustrate some elementary aspects of
scripting.

.. literalinclude:: script_spy_files/iterative1.spy
   :language: python

Let us now analyze this script. The first line is a comment that happens
to give the name of the file. This is followed by two lines that import
symbols for Pyomo. The pyomo namespace is imported as
``pyo``. Therefore, ``pyo.`` must precede each use of a Pyomo name.

.. literalinclude:: script_spy_files/iterative1_Import_symbols_for_pyomo.spy
   :language: python

An object to perform optimization is created by calling
``SolverFactory`` with an argument giving the name of the solver. The
argument would be ``'gurobi'`` if, e.g., Gurobi was desired instead of
glpk:

.. literalinclude:: script_spy_files/iterative1_Call_SolverFactory_with_argument.spy
   :language: python

The next lines after a comment create a model. For our discussion here,
we will refer to this as the base model because it will be extended by
adding constraints later. (The words "base model" are not reserved
words, they are just being introduced for the discussion of this
example).  There are no constraints in the base model, but that is just
to keep it simple.  Constraints could be present in the base model.
Even though it is an abstract model, the base model is fully specified
by these commands because it requires no external data:

.. literalinclude:: script_spy_files/iterative1_Call_SolverFactory_with_argument.spy
   :language: python

The next line is not part of the base model specification. It creates an
empty constraint list that the script will use to add constraints.

.. literalinclude:: script_spy_files/iterative1_Create_empty_constraint_list.spy
   :language: python

The next non-comment line creates the instantiated model and refers to
the instance object with a Python variable ``instance``.  Models run
using the ``pyomo`` script do not typically contain this line because
model instantiation is done by the ``pyomo`` script. In this example,
the ``create`` function is called without arguments because none are
needed; however, the name of a file with data commands is given as an
argument in many scripts.

.. literalinclude:: script_spy_files/iterative1_Create_instantiated_model.spy
   :language: python

The next line invokes the solver and refers to the object contain
results with the Python variable ``results``.

.. literalinclude:: script_spy_files/iterative1_Solve_and_refer_to_results.spy
   :language: python

The solve function loads the results into the instance, so the next line
writes out the updated values.

.. literalinclude:: script_spy_files/iterative1_Display_updated_value.spy
   :language: python

The next non-comment line is a Python iteration command that will
successively assign the integers from 0 to 4 to the Python variable
``i``, although that variable is not used in script. This loop is what
causes the script to generate five more solutions:

.. literalinclude:: script_spy_files/iterative1_Assign_integers.spy
   :language: python

An expression is built up in the Python variable named ``expr``.  The
Python variable ``j`` will be iteratively assigned all of the indexes of
the variable ``x``. For each index, the value of the variable (which was
loaded by the ``load`` method just described) is tested to see if it is
zero and the expression in ``expr`` is augmented accordingly.  Although
``expr`` is initialized to 0 (an integer), its type will change to be a
Pyomo expression when it is assigned expressions involving Pyomo
variable objects:

.. literalinclude:: script_spy_files/iterative1_Iteratively_assign_and_test.spy
   :language: python

During the first iteration (when ``i`` is 0), we know that all values of
``x`` will be 0, so we can anticipate what the expression will look
like. We know that ``x`` is indexed by the integers from 1 to 4 so we
know that ``j`` will take on the values from 1 to 4 and we also know
that all value of ``x`` will be zero for all indexes so we know that the
value of ``expr`` will be something like

::

  0 + instance.x[1] + instance.x[2] + instance.x[3] + instance.x[4]

The value of ``j`` will be evaluated because it is a Python variable;
however, because it is a Pyomo variable, the value of ``instance.x[j]``
not be used, instead the variable object will appear in the
expression. That is exactly what we want in this case. When we wanted to
use the current value in the ``if`` statement, we used the ``value``
function to get it.

The next line adds to the constaint list called ``c`` the requirement
that the expression be greater than or equal to one:

.. literalinclude:: script_spy_files/iterative1_Add_expression_constraint.spy
   :language: python

The proof that this precludes the last solution is left as an exerise
for the reader.

The final lines in the outer for loop find a solution and display it:

.. literalinclude:: script_spy_files/iterative1_Find_and_display_solution.spy
   :language: python

.. note::
   
   The assignment of the solve output to a results object is somewhat
   anachronistic. Many scripts just use

   >>> opt.solve(instance) # doctest: +SKIP

   since the results are moved to the instance by default, leaving
   the results object with little of interest. If, for some reason,
   you want the results to stay in the results object and *not* be
   moved to the instance, you would use

   >>> results = opt.solve(instance, load_solutions=False) # doctest: +SKIP
   
   This approach can be usefull if there is a concern that the solver
   did not terminate with an optimal solution. For example,
   
   >>> results = opt.solve(instance, load_solutions=False) # doctest: +SKIP
   >>> if results.solver.termination_condition == TerminationCondition.optimal: # doctest: +SKIP
   ...     instance.solutions.load_from(results) # doctest: +SKIP

Changing the Model or Data and Re-solving
-----------------------------------------

The ``iterative1.py`` example above illustrates how a model can be changed and
then re-solved. In that example, the model is changed by adding a
constraint, but the model could also be changed by altering the values
of parameters. Note, however, that in these examples, we make the
changes to the concrete model instances.  This is particularly important
for ``AbstractModel`` users, as this implies working with the
``instance`` object rather than the ``model`` object, which allows us to
avoid creating a new ``model`` object for each solve. Here is the basic
idea for users of an ``AbstractModel``:

#. Create an ``AbstractModel`` (suppose it is called ``model``)
#. Call ``model.create_instance()`` to create an instance (suppose it is called ``instance``)
#. Solve ``instance``
#. Change someting in ``instance``
#. Solve ``instance`` again

.. note::

   Users of ``ConcreteModel`` typically name their models ``model``, which
   can cause confusion to novice readers of documentation. Examples based on
   an ``AbstractModel`` will refer to ``instance`` where users of a
   ``ConcreteModel`` would typically use the name ``model``.

..
   NOTE: the tests in this file are fragile right now because some
   code has been brought inline (particularly from spy4cripts.spy)
   DO NOT redefine instance until you are ready to delete this entire comment.
   (and note that model is redefined all over the place).
   
.. testsetup:: *

    # code from spy4scripts
    import pyomo.environ as pyo

    instance = pyo.ConcreteModel()
    instance.I = pyo.Set(initialize=[1,2,3])
    instance.sigma = pyo.Param(mutable=True, initialize=2.3)
    instance.Theta = pyo.Param(instance.I, mutable=True)
    for i in instance.I:
        instance.Theta[i] = i
    idx = 1
    NewVal = 1134
    
    instance.x = pyo.Var([1,2,3], initialize=0)
    instance.y = pyo.Var()
    instance.iVar = pyo.Var([1,2,3], initialize=1, domain=pyo.Boolean)
    instance.sVar = pyo.Var(initialize=1, domain=pyo.Boolean)

If ``instance`` has a parameter whose name is ``Theta`` that was
declared to be ``mutable`` (i.e., ``mutable=True``) with an
index that contains ``idx``, then the value in ``NewVal`` can be assigned to
it using

   >>> instance.Theta[idx] = NewVal	       

For a singleton parameter named ``sigma`` (i.e., if it is not
indexed), the assignment can be made using

   >>> instance.sigma = NewVal

.. note::

   If the ``Param`` is not declared to be mutable, an error will occur if an assignment to it is attempted.
    
For more information about access to Pyomo parameters, see the section
in this document on ``Param`` access :ref:`ParmAccess`. Note that for
concrete models, the model is the instance.

Fixing Variables and Re-solving
-------------------------------

Instead of changing model data, scripts are often used to fix variable
values. The following example illustrates this.

.. literalinclude:: script_spy_files/iterative2.spy
   :language: python

In this example, the variables are binary. The model is solved and then
the value of ``model.x[2]`` is flipped to the opposite value before
solving the model again. The main lines of interest are:

.. literalinclude:: script_spy_files/iterative2_Flip_value_before_solve_again.spy
   :language: python

This could also have been accomplished by setting the upper and lower
bounds:

   >>> if instance.x[2] == 0:
   ...     instance.x[2].setlb(1)
   ...     instance.x[2].setub(1)
   ... else:
   ...     instance.x[2].setlb(0)
   ...     instance.x[2].setub(0)
   
Notice that when using the bounds, we do not set ``fixed`` to ``True``
because that would fix the variable at whatever value it presently has
and then the bounds would be ignored by the solver.

For more information about access to Pyomo variables, see the section in
this document on ``Var`` access :ref:`VarAccess`.

Note that

   >>> instance.x.fix(2)

is equivalent to

   >>> instance.y.value = 2
   >>> instance.y.fixed = True

and
   >>> instance.x.fix()

is equivalent to

   >>> instance.x.fixed = True

Extending the Objective Function
--------------------------------

One can add terms to an objective function of a ``ConcreteModel`` (or
and instantiated ``AbstractModel``) using the ``expr`` attribute
of the objective function object. Here is a simple example:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.opt import SolverFactory

   >>> model = pyo.ConcreteModel()

   >>> model.x = pyo.Var(within=pyo.PositiveReals)
   >>> model.y = pyo.Var(within=pyo.PositiveReals)

   >>> model.sillybound = pyo.Constraint(expr = model.x + model.y <= 2)

   >>> model.obj = pyo.Objective(expr = 20 * model.x)

   >>> opt = SolverFactory('glpk') # doctest: +SKIP
   >>> opt.solve(model) # doctest: +SKIP

   >>> model.pprint() # doctest: +SKIP

   >>> print ("------------- extend obj --------------") # doctest: +SKIP
   >>> model.obj.expr += 10 * model.y

   >>> opt = SolverFactory('cplex') # doctest: +SKIP
   >>> opt.solve(model) # doctest: +SKIP
   >>> model.pprint() # doctest: +SKIP

Activating and Deactivating Objectives
--------------------------------------

Multiple objectives can be declared, but only one can be active at a
time (at present, Pyomo does not support any solvers that can be given
more than one objective). If both ``model.obj1`` and ``model.obj2`` have
been declared using ``Objective``, then one can ensure that
``model.obj2`` is passed to the solver as shown in this simple example:

   >>> model = pyo.ConcreteModel()
   >>> model.obj1 = pyo.Objective(expr = 0)
   >>> model.obj2 = pyo.Objective(expr = 0)

   >>> model.obj1.deactivate()
   >>> model.obj2.activate()

For abstract models this would be done prior to instantiation or else
the ``activate`` and ``deactivate`` calls would be on the instance
rather than the model.

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
been loded back into the instance object, then we can make use of the
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

.. literalinclude:: script_spy_files/noiteration1.py
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
results have been loded back into the instance object. Here is a code
snippet for fixing all integers at their current value:

    >>> for var in instance.component_data_objects(pyo.Var, active=True):
    ...     if not var.is_continuous():
    ...         print ("fixing "+str(v))  # doctest: +SKIP
    ...         var.fixed = True # fix the current value


Another way to access all of the variables (particularly if there are
blocks) is as follows (this particular snippet assumes that instead of
`import pyomo.environ as pyo` `from pyo.environ import *` was used):

.. literalinclude:: script_spy_files/block_iter_example_compprintloop.spy
   :language: python

.. _ParmAccess:

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

.. literalinclude:: script_spy_files/driveabs2_Create_dual_suffix_component.spy
   :language: python

See the section on Suffixes :ref:`Suffixes` for more information on
Pyomo's Suffix component. After the results are obtained and loaded into
an instance, duals can be accessed in the following fashion.

.. literalinclude:: script_spy_files/driveabs2_Access_all_dual.spy
   :language: python

The following snippet will only work, of course, if there is a
constraint with the name ``AxbConstraint`` that has and index, which is
the string ``Film``.

.. literalinclude:: script_spy_files/driveabs2_Access_one_dual.spy
   :language: python

Here is a complete example that relies on the file ``abstract2.py`` to
provide the model and the file ``abstract2.dat`` to provide the
data. Note that the model in ``abstract2.py`` does contain a constraint
named ``AxbConstraint`` and ``abstract2.dat`` does specify an index for
it named ``Film``.

.. literalinclude:: script_spy_files/driveabs2.spy
   :language: python

Concrete models are slightly different because the model is the
instance. Here is a complete example that relies on the file
``concrete1.py`` to provide the model and instantiate it.

.. literalinclude:: script_spy_files/driveconc1.py
   :language: python

Accessing Slacks
----------------

The functions ``lslack()`` and ``uslack()`` return the upper and lower
slacks, respectively, for a constraint.


Accessing Solver Status
-----------------------

After a solve, the results object has a member ``Solution.Status`` that
contains the solver status. The following snippet shows an example of
access via a ``print`` statement:

.. literalinclude:: script_spy_files/spy4scripts_Print_solver_status.spy
   :language: python

The use of the Python ``str`` function to cast the value to a be string
makes it easy to test it. In particular, the value 'optimal' indicates
that the solver succeeded. It is also possible to access Pyomo data that
can be compared with the solver status as in the following code snippet:

.. literalinclude:: script_spy_files/spy4scripts_Pyomo_data_comparedwith_solver_status_1.spy
   :language: python

Alternatively,

.. literalinclude:: script_spy_files/spy4scripts_Pyomo_data_comparedwith_solver_status_2.spy
   :language: python

.. _TeeTrue:

Display of Solver Output
------------------------


To see the output of the solver, use the option ``tee=True`` as in

.. literalinclude:: script_spy_files/spy4scripts_See_solver_output.spy
   :language: python

This can be useful for troubleshooting solver difficulties.

.. _SolverOpts:

Sending Options to the Solver
-----------------------------

Most solvers accept options and Pyomo can pass options through to a
solver. In scripts or callbacks, the options can be attached to the
solver object by adding to its options dictionary as illustrated by this
snippet:

.. literalinclude:: script_spy_files/spy4scripts_Add_option_to_solver.spy
   :language: python

If multiple options are needed, then multiple dictionary entries should
be added.

Sometime it is desirable to pass options as part of the call to the
solve function as in this snippet:

.. literalinclude:: script_spy_files/spy4scripts_Add_multiple_options_to_solver.spy
   :language: python

The quoted string is passed directly to the solver. If multiple options
need to be passed to the solver in this way, they should be separated by
a space within the quoted string. Notice that ``tee`` is a Pyomo option
and is solver-independent, while the string argument to ``options`` is
passed to the solver without very little processing by Pyomo. If the
solver does not have a "threads" option, it will probably complain, but
Pyomo will not.

There are no default values for options on a ``SolverFactory``
object. If you directly modify its options dictionary, as was done
above, those options will persist across every call to
``optimizer.solve(…)`` unless you delete them from the options
dictionary. You can also pass a dictionary of options into the
``opt.solve(…)`` method using the ``options`` keyword. Those options
will only persist within that solve and temporarily override any
matching options in the options dictionary on the solver object.

Specifying the Path to a Solver
-------------------------------

Often, the executables for solvers are in the path; however, for
situations where they are not, the SolverFactory function accepts the
keyword ``executable``, which you can use to set an absolute or relative
path to a solver executable. E.g.,

.. literalinclude:: script_spy_files/spy4scripts_Set_path_to_solver_executable.spy
   :language: python

Warm Starts
-----------

Some solvers support a warm start based on current values of
variables. To use this feature, set the values of variables in the
instance and pass ``warmstart=True`` to the ``solve()`` method. E.g.,

.. literalinclude:: script_spy_files/spy4scripts_Pass_warmstart_to_solver.spy
   :language: python

.. note::

   The Cplex and Gurobi LP file (and Python) interfaces will generate an
   MST file with the variable data and hand this off to the solver in
   addition to the LP file.

.. warning::

   Solvers using the NL file interface (e.g., "gurobi_ampl", "cplexamp")
   do not accept warmstart as a keyword to the solve() method as the NL
   file format, by default, includes variable initialization data (drawn
   from the current value of all variables).


Solving Multiple Instances in Parallel
--------------------------------------

Use of parallel solvers for PySP is discussed in the section on parallel
PySP :ref:`ParallelPySP`.

Solvers are controlled by solver servers. The pyro mip solver server is
launched with the command ``pyro_mip_server``. This command may be
repeated to launch as many solvers as are desired. A name server and a
dispatch server must be running and accessible to the process that runs
the script that will use the mip servers as well as to the mip
servers. The name server is launched using the command ``pyomo_ns`` and
then the dispatch server is launched with ``dispatch_srvr``. Note that
both commands contain an underscore. Both programs keep running until
terminated by an external signal, so it is common to pipe their output
to a file.  The commands are:

- Once: ``pyomo_ns``
- Once: ``dispatch_srvr``
- Multiple times: ``pyro_mip_server``


This example demonstrates how to use these services to solve two
instances in parallel.

.. literalinclude:: script_spy_files/parallel.py
   :language: python

This example creates two instances that are very similar and then sends
them to be dispatched to solvers. If there are two solvers, then these
problems could be solved in parallel (we say "could" because for such
trivial problems to be actually solved in parallel, the solvers would
have to be very, very slow). This example is non-sensical; the goal is
simply to show ``solver_manager.queue`` to submit jobs to a name server
for dispatch to solver servers and ``solver_manager.wait_any`` to
recover the results. The ``wait_all`` function is similar, but it takes
a list of action handles (returned by ``queue``) as an argument and
returns all of the results at once.

Changing the temporary directory
--------------------------------

A "temporary" directory is used for many intermediate files. Normally,
the name of the directory for temporary files is provided by the
operating system, but the user can specify their own directory name.
The pyomo command-line ``--tempdir`` option propagates through to the
TempFileManager service. One can accomplish the same through the
following few lines of code in a script:

.. literalinclude:: script_spy_files/spy4scripts_Specify_temporary_directory_name.spy
   :language: python
