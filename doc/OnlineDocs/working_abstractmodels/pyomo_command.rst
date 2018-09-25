The ``pyomo`` Command
=====================

The ``pyomo`` command is issued to the DOS prompt or a Unix shell.  To
see a list of Pyomo command line options, use:

::

   pyomo solve --help

.. note::

   There are two dashes before ``help``.

In this section we will detail some of the options.

Passing Options to a Solver
---------------------------

To pass arguments to a solver when using the ``pyomo solve`` command,
appned the Pyomo command line with the argument ``--solver-options=``
followed by an argument that is a string to be sent to the solver
(perhaps with dashes added by Pyomo).  So for most MIP solvers, the mip
gap can be set using

::

   --solver-options= "mipgap=0.01 "

Multiple options are separated by a space.  Options that do not take an
argument should be specified with the equals sign followed by either a
space or the end of the string.

For example, to specify that the solver is GLPK, then to specify a
mipgap of two percent and the GLPK cuts option, use

::

   solver=glpk --solver-options="mipgap=0.02 cuts="

If there are multiple "levels" to the keyword, as is the case for some
Gurobi and CPLEX options, the tokens are separated by underscore.  For
example, ``mip cuts all`` would be specified as ``mip_cuts_all``.  For
another example, to set the solver to be CPLEX, then to set a mip gap of
one percent and to specify 'y' for the sub-option ``numerical`` to the
option ``emphasis`` use

::

   --solver=cplex --solver-options="mipgap=0.001 emphasis_numerical=y"

See :ref:`SolverOpts` for a discussion of passing options in a script.

Troubleshooting
---------------

Many of things that can go wrong are covered by error messages, but
sometimes they can be confusing or do not provide enough
information. Depending on what the troubles are, there might be ways to
get a little additional information.

If there are syntax errors in the model file, for example, it can
occasionally be helpful to get error messages directly from the Python
interpreter rather than through Pyomo. Suppose the name of the model
file is scuc.py, then

::

   python scuc.py

can sometimes give useful information for fixing syntax errors.

When there are no syntax errors, but there troubles reading the data or
generating the information to pass to a solver, then the ``--verbose``
option provides a trace of the execution of Pyomo. The user should be
aware that for some models this option can generate a lot of output.

If there are troubles with solver (i.e., after Pyomo has output
"Applying Solver"), it is often helpful to use the option
``--stream-solver`` that causes the solver output to be displayed rather
than trapped. (See <<TeeTrue>> for information about getting this output
in a script). Advanced users may wish to examine the files that are
generated to be passed to a solver. The type of file generated is
controlled by the ``--solver-io`` option and the ``--keepfiles`` option
instructs pyomo to keep the files and output their names. However, the
``--symbolic-solver-labels`` option should usually also be specified so
that meaningful names are used in these files.

When there seem to be troubles expressing the model, it is often useful
to embed print commands in the model in places that will yield helpful
information.  Consider the following snippet:

.. literalinclude:: ../script_spy_files/spy4PyomoCommand_Troubleshooting_printed_command.spy
   :language: python

The effect will be to output every member of the set ``model.I`` at the
time the constraint named ``model.AxbConstraint`` is constructed.

Direct Interfaces to Solvers
----------------------------

In many applications, the default solver interface works well. However,
in some cases it is useful to specify the interface using the
``solver-io`` option. For example, if the solver supports a direct
Python interface, then the option would be specified on the command line
as

::

   --solver-io=python

Here are some of the choices:

- lp: generate a standard linear programming format file with filename
  extension ``lp``
- nlp: generate a file with a standard format that supports linear and
  nonlinear optimization with filename extension ``n1lp``
- os: generate an OSiL format XML file.
- python: use the direct Python interface.

.. note::

   Not all solvers support all interfaces.
