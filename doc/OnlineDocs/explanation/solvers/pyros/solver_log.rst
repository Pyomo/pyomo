.. _pyros_solver_log:

=======================
PyROS Solver Output Log
=======================

.. contents:: Table of Contents
   :depth: 1
   :local:


.. _pyros_solver_log_appearance:

Default Format
==============

When the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
is called to solve a robust optimization problem,
your console output will, by default, look like this:

.. _solver-log-snippet:

.. code-block:: text
   :caption: PyROS solver output log for the :ref:`Quickstart example <pyros_quickstart_example-two-stg>`.
   :linenos:

   ==============================================================================
   PyROS: The Pyomo Robust Optimization Solver, v1.3.11.
          Pyomo version: 6.9.5
          Commit hash: unknown
          Invoked at UTC 2025-10-18T00:00:00.000000+00:00
   
   Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),
                 John D. Siirola (2), Chrysanthos E. Gounaris (1)
   (1) Carnegie Mellon University, Department of Chemical Engineering
   (2) Sandia National Laboratories, Center for Computing Research
   
   The developers gratefully acknowledge support from the U.S. Department
   of Energy's Institute for the Design of Advanced Energy Systems (IDAES).
   ==============================================================================
   ================================= DISCLAIMER =================================
   PyROS is currently under active development. 
   Please provide feedback and/or report any issues by creating a ticket at
   https://github.com/Pyomo/pyomo/issues/new/choose
   ==============================================================================
   User-provided solver options:
    objective_focus=<ObjectiveType.worst_case: 1>
    decision_rule_order=1
    solve_master_globally=True
   ------------------------------------------------------------------------------
   Model Statistics (before preprocessing):
     Number of variables : 4
       First-stage variables : 1
       Second-stage variables : 1
       State variables : 2
     Number of uncertain parameters : 2
     Number of constraints : 4
       Equality constraints : 2
       Inequality constraints : 2
   ------------------------------------------------------------------------------
   Preprocessing...
   Done preprocessing; required wall time of 0.004s.
   ------------------------------------------------------------------------------
   Itn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)
   ------------------------------------------------------------------------------
   0     5.4079e+03  -            -            3       7.9226e+00   0.209        
   1     5.4079e+03  6.0451e-10   1.0717e-10   2       1.0250e-01   0.476        
   2     6.5403e+03  1.0018e-01   7.4564e-03   1       1.0249e-02   0.786        
   3     6.5403e+03  1.9372e-16   2.0321e-05   2       8.7074e-03   1.132        
   4     6.5403e+03  0.0000e+00   2.0311e-05   0       1.2310e-06g  1.956        
   ------------------------------------------------------------------------------
   Robust optimal solution identified.
   ------------------------------------------------------------------------------
   Termination stats:
    Iterations            : 5
    Solve time (wall s)   : 1.956
    Final objective value : 6.5403e+03
    Termination condition : pyrosTerminationCondition.robust_optimal
   ------------------------------------------------------------------------------
   All done. Exiting PyROS.
   ==============================================================================



Observe that the log contains the following information
(listed in order of appearance):


* **Introductory information and disclaimer** (lines 1--19):
  Includes the version number, author
  information, (UTC) time at which the solver was invoked,
  and, if available, information on the local Git branch and
  commit hash.
* **Summary of solver options** (lines 20--24): Enumeration of
  specifications for optional arguments to the solver.
* **Model component statistics** (lines 25--34):
  Breakdown of component statistics for the user-provided model
  and variable selection (before preprocessing).
* **Preprocessing information** (lines 35--37):
  Wall time required for preprocessing
  the deterministic model and associated components,
  i.e., standardizing model components and adding the decision rule
  variables and equations.
* **Iteration log table** (lines 38--45):
  Summary information on the problem iterates and subproblem outcomes.
  The constituent columns are defined in detail in
  :ref:`the table that follows <table-iteration-log-columns>`.
* **Termination message** (lines 46--47): One-line message briefly summarizing
  the reason the solver has terminated.
* **Final result** (lines 48--53):
  A printout of the
  :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`
  object that is finally returned.
* **Exit message** (lines 54--55): Confirmation that the
  solver has been exited properly.

The iteration log table is designed to provide, in a concise manner,
important information about the progress of the iterative algorithm for
the problem of interest.
The constituent columns are defined in the
table below.

.. _pyros-table-iteration-log-columns:

.. list-table:: PyROS iteration log table columns.
   :widths: 10 50
   :header-rows: 1

   * - Column Name
     - Definition
   * - Itn
     - Iteration number, equal to one less than the total number of elapsed
       iterations.
   * - Objective
     - Master solution objective function value.
       If the objective of the deterministic model provided
       has a maximization sense,
       then the negative of the objective function value is displayed.
       Expect this value to trend upward as the iteration number
       increases.
       A dash ("-") is produced in lieu of a value if the master
       problem of the current iteration is not solved successfully.
   * - 1-Stg Shift
     - Infinity norm of the relative difference between the first-stage
       variable vectors of the master solutions of the current
       and previous iterations. Expect this value to trend
       downward as the iteration number increases.
       A dash ("-") is produced in lieu of a value
       if the current iteration number is 0,
       there are no first-stage variables,
       or the master problem of the current iteration is not solved successfully.
   * - 2-Stg Shift
     - Infinity norm of the relative difference between the second-stage
       variable vectors (evaluated subject to the nominal uncertain
       parameter realization) of the master solutions of the current
       and previous iterations. Expect this value to trend
       downward as the iteration number increases.
       A dash ("-") is produced in lieu of a value
       if the current iteration number is 0,
       there are no second-stage variables,
       or the master problem of the current iteration is not solved successfully.
       An asterisk ("*") is appended to the value if decision rule
       polishing was unsuccessful.
   * - #CViol
     - Number of second-stage inequality constraints found to be violated during
       the separation step of the current iteration.
       Unless a custom prioritization of the model's second-stage inequality
       constraints is specified (through the ``separation_priority_order`` argument),
       expect this number to trend downward as the iteration number increases.
       A "+" is appended if not all of the separation problems
       were solved successfully, either due to custom prioritization, a time out,
       or an issue encountered by the subordinate optimizers.
       A dash ("-") is produced in lieu of a value if the separation
       routine is not invoked during the current iteration.
   * - Max Viol
     - Maximum scaled second-stage inequality constraint violation.
       Expect this value to trend downward as the iteration number increases.
       A 'g' is appended to the value if the separation problems were solved
       globally during the current iteration.
       A dash ("-") is produced in lieu of a value if the separation
       routine is not invoked during the current iteration, or if there are
       no second-stage inequality constraints.
   * - Wall time (s)
     - Total time elapsed by the solver, in seconds, up to the end of the
       current iteration.


.. _pyros_solver_log_verbosity:

Configuring the Output Log
==========================

The PyROS solver output log is produced by the
Python logger (:py:class:`logging.Logger`) object
derived from the optional argument ``progress_logger``
to the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.
By default, the PyROS solver argument ``progress_logger``
is taken to be the :py:obj:`logging.INFO`-level
logger with name ``"pyomo.contrib.pyros"``.
The verbosity level of the output log can be adjusted by setting the
:py:mod:`logging` level of the progress logger.
For example, the level of the default logger can be adjusted to
:py:obj:`logging.DEBUG` as follows:

.. code-block::

   import logging
   logging.getLogger("pyomo.contrib.pyros").setLevel(logging.DEBUG)


We refer the reader to the
:doc:`official Python logging library documentation <python:library/logging>`
for further guidance on (customization of) Python logger objects.


The :ref:`following table <pyros-table-logging-levels>`
describes the information logged by PyROS at the various :py:mod:`logging` levels.
Messages of a lower logging level than that of the progress logger
are excluded from the solver log.


.. _pyros-table-logging-levels:

.. list-table:: PyROS solver log output at the various standard Python :py:mod:`logging` levels.
   :widths: 10 50
   :header-rows: 1

   * - Logging Level
     - Output Messages
   * - :py:obj:`logging.ERROR`
     - * Elaborations of exceptions stemming from expression
         evaluation errors or issues encountered by the subordinate solvers
   * - :py:obj:`logging.WARNING`
     - * Elaboration of unacceptable subproblem termination statuses
         for critical subproblems
       * Caution about solution robustness guarantees in event that
         user passes ``bypass_global_separation=True``
   * - :py:obj:`logging.INFO`
     - * PyROS version, author, and disclaimer information
       * Summary of user options
       * Model component statistics (before preprocessing)
       * Summary of preprocessing outcome
       * Iteration log table
       * Termination message and summary statistics
       * Exit message
   * - :py:obj:`logging.DEBUG`
     - * Detailed progress through the various preprocessing subroutines
       * Detailed component statistics for the preprocessed model
       * Termination outcomes, backup solver invocation statements,
         and summaries of results for all subproblems
       * Summary of separation subroutine overall outcomes:
         second-stage inequality constraints violated and
         uncertain parameter realization(s) added to the master problem
       * Solve time profiling statistics
