Solver Recipes
==============


Accessing Solver Status
-----------------------

After a solve, the results object has a member ``Solution.Status`` that
contains the solver status. The following snippet shows an example of
access via a ``print`` statement:

.. literalinclude:: /src/scripting/spy4scripts_Print_solver_status.spy
   :language: python

The use of the Python ``str`` function to cast the value to a be string
makes it easy to test it. In particular, the value 'optimal' indicates
that the solver succeeded. It is also possible to access Pyomo data that
can be compared with the solver status as in the following code snippet:

.. literalinclude:: /src/scripting/spy4scripts_Pyomo_data_comparedwith_solver_status_1.spy
   :language: python

Alternatively,

.. literalinclude:: /src/scripting/spy4scripts_Pyomo_data_comparedwith_solver_status_2.spy
   :language: python

.. _TeeTrue:

Display of Solver Output
------------------------


To see the output of the solver, use the option ``tee=True`` as in

.. literalinclude:: /src/scripting/spy4scripts_See_solver_output.spy
   :language: python

This can be useful for troubleshooting solver difficulties.

.. _SolverOpts:

Sending Options to the Solver
-----------------------------

Most solvers accept options and Pyomo can pass options through to a
solver. In scripts or callbacks, the options can be attached to the
solver object by adding to its options dictionary as illustrated by this
snippet:

.. literalinclude:: /src/scripting/spy4scripts_Add_option_to_solver.spy
   :language: python

If multiple options are needed, then multiple dictionary entries should
be added.

Sometimes it is desirable to pass options as part of the call to the
solve function as in this snippet:

.. literalinclude:: /src/scripting/spy4scripts_Add_multiple_options_to_solver.spy
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

.. literalinclude:: /src/scripting/spy4scripts_Set_path_to_solver_executable.spy
   :language: python

Warm Starts
-----------

Some solvers support a warm start based on current values of
variables. To use this feature, set the values of variables in the
instance and pass ``warmstart=True`` to the ``solve()`` method. E.g.,

.. literalinclude:: /src/scripting/spy4scripts_Pass_warmstart_to_solver.spy
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

Building and solving Pyomo models in parallel is a common requirement
for many applications. We recommend using MPI for Python (mpi4py) for
this purpose. For more information on mpi4py, see the mpi4py
documentation (https://mpi4py.readthedocs.io/en/stable/). The example
below demonstrates how to use mpi4py to solve two pyomo models in
parallel. The example can be run with the following command:

.. code-block::

   mpirun -np 2 python -m mpi4py parallel.py


.. literalinclude:: /src/scripting/parallel.py
   :language: python


Changing the temporary directory
--------------------------------

A "temporary" directory is used for many intermediate files. Normally,
the name of the directory for temporary files is provided by the
operating system, but the user can specify their own directory name.
The pyomo command-line ``--tempdir`` option propagates through to the
TempFileManager service. One can accomplish the same through the
following few lines of code in a script:

.. literalinclude:: /src/scripting/spy4scripts_Specify_temporary_directory_name.spy
   :language: python
