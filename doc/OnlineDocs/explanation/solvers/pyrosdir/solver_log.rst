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
is invoked on a robust optimization problem,
your console output will, by default, look like this
(line numbers added for reference):


.. _solver-log-snippet:

.. code-block:: text
   :caption: PyROS solver output log for the :ref:`two-stage problem example <example-two-stg>`.
   :linenos:

   ==============================================================================
   PyROS: The Pyomo Robust Optimization Solver, v1.3.4.
          Pyomo version: 6.9.0
          Commit hash: unknown
          Invoked at UTC 2025-02-13T00:00:00.000000
   
   Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),
                 John D. Siirola (2), Chrysanthos E. Gounaris (1)
   (1) Carnegie Mellon University, Department of Chemical Engineering
   (2) Sandia National Laboratories, Center for Computing Research
   
   The developers gratefully acknowledge support from the U.S. Department
   of Energy's Institute for the Design of Advanced Energy Systems (IDAES).
   ==============================================================================
   ================================= DISCLAIMER =================================
   PyROS is still under development.
   Please provide feedback and/or report any issues by creating a ticket at
   https://github.com/Pyomo/pyomo/issues/new/choose
   ==============================================================================
   Solver options:
    time_limit=None
    keepfiles=False
    tee=False
    load_solution=True
    symbolic_solver_labels=False
    objective_focus=<ObjectiveType.worst_case: 1>
    nominal_uncertain_param_vals=[0.13248000000000001, 4.97, 4.97, 1800]
    decision_rule_order=1
    solve_master_globally=True
    max_iter=-1
    robust_feasibility_tolerance=0.0001
    separation_priority_order={}
    progress_logger=<PreformattedLogger pyomo.contrib.pyros (INFO)>
    backup_local_solvers=[]
    backup_global_solvers=[]
    subproblem_file_directory=None
    bypass_local_separation=False
    bypass_global_separation=False
    p_robustness={}
   ------------------------------------------------------------------------------
   Preprocessing...
   Done preprocessing; required wall time of 0.009s.
   ------------------------------------------------------------------------------
   Model Statistics:
     Number of variables : 62
       Epigraph variable : 1
       First-stage variables : 7
       Second-stage variables : 6 (6 adj.)
       State variables : 18 (7 adj.)
       Decision rule variables : 30
     Number of uncertain parameters : 4
     Number of constraints : 52
       Equality constraints : 24
         Coefficient matching constraints : 0
         Other first-stage equations : 10
         Second-stage equations : 8
         Decision rule equations : 6
       Inequality constraints : 28
         First-stage inequalities : 1
         Second-stage inequalities : 27
   ------------------------------------------------------------------------------
   Itn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)
   ------------------------------------------------------------------------------
   0     3.5838e+07  -            -            5       1.8832e+04   0.412
   1     3.5838e+07  1.2289e-09   1.5886e-12   5       2.8919e+02   0.992
   2     3.6269e+07  3.1647e-01   1.0432e-01   4       2.9020e+02   1.865
   3     3.6285e+07  7.6526e-01   2.2258e-01   0       2.3874e-12g  3.508
   ------------------------------------------------------------------------------
   Robust optimal solution identified.
   ------------------------------------------------------------------------------
   Timing breakdown:
   
   Identifier                ncalls   cumtime   percall      %
   -----------------------------------------------------------
   main                           1     3.509     3.509  100.0
        ------------------------------------------------------
        dr_polishing              3     0.209     0.070    6.0
        global_separation        27     0.590     0.022   16.8
        local_separation        108     1.569     0.015   44.7
        master                    4     0.654     0.163   18.6
        master_feasibility        3     0.083     0.028    2.4
        preprocessing             1     0.009     0.009    0.3
        other                   n/a     0.394       n/a   11.2
        ======================================================
   ===========================================================
   
   ------------------------------------------------------------------------------
   Termination stats:
    Iterations            : 4
    Solve time (wall s)   : 3.509
    Final objective value : 3.6285e+07
    Termination condition : pyrosTerminationCondition.robust_optimal
   ------------------------------------------------------------------------------
   All done. Exiting PyROS.
   ==============================================================================

Observe that the log contains the following information:


* **Introductory information** (lines 1--18).
  Includes the version number, author
  information, (UTC) time at which the solver was invoked,
  and, if available, information on the local Git branch and
  commit hash.
* **Summary of solver options** (lines 19--40).
* **Preprocessing information** (lines 41--43).
  Wall time required for preprocessing
  the deterministic model and associated components,
  i.e., standardizing model components and adding the decision rule
  variables and equations.
* **Model component statistics** (lines 44--61).
  Breakdown of model component statistics.
  Includes components added by PyROS, such as the decision rule variables
  and equations.
  The preprocessor may find that some second-stage variables
  and state variables are mathematically
  not adjustable to the uncertain parameters.
  To this end, in the logs, the numbers of
  adjustable second-stage variables and state variables
  are included in parentheses, next to the total numbers
  of second-stage variables and state variables, respectively;
  note that "adjustable" has been abbreviated as "adj."
* **Iteration log table** (lines 62--68).
  Summary information on the problem iterates and subproblem outcomes.
  The constituent columns are defined in detail in
  :ref:`the table that follows <table-iteration-log-columns>`.
* **Termination message** (lines 69--70). Very brief summary of the termination outcome.
* **Timing statistics** (lines 71--87).
  Tabulated breakdown of the solver timing statistics, based on a
  :class:`pyomo.common.timing.HierarchicalTimer` printout.
  The identifiers are as follows:

  * ``main``: Time elapsed by the solver.
  * ``main.dr_polishing``: Time spent by the subordinate solvers
    on polishing of the decision rules.
  * ``main.global_separation``: Time spent by the subordinate solvers
    on global separation subproblems.
  * ``main.local_separation``: Time spent by the subordinate solvers
    on local separation subproblems.
  * ``main.master``: Time spent by the subordinate solvers on
    the master problems.
  * ``main.master_feasibility``: Time spent by the subordinate solvers
    on the master feasibility problems.
  * ``main.preprocessing``: Preprocessing time.
  * ``main.other``: Overhead time.

* **Final result** (lines 88--93).
  A printout of the
  :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`
  object that is finally returned
* **Exit message** (lines 94--95).

The iteration log table (lines 62--68) is designed to provide, in a concise manner,
important information about the progress of the iterative algorithm for
the problem of interest.
The constituent columns are defined in the
table below:

.. _table-iteration-log-columns:

.. list-table:: PyROS iteration log table columns.
   :widths: 10 50
   :header-rows: 1

   * - Column Name
     - Definition
   * - Itn
     - Iteration number.
   * - Objective
     - Master solution objective function value.
       If the objective of the deterministic model provided
       has a maximization sense,
       then the negative of the objective function value is displayed.
       Expect this value to trend upward as the iteration number
       increases.
       If the master problems are solved globally
       (by passing ``solve_master_globally=True``),
       then after the iteration number exceeds the number of uncertain parameters,
       this value should be monotonically nondecreasing
       as the iteration number is increased.
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

For a given call to the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method,
the solver log output is produced by the
Python logger (:py:class:`logging.Logger`) object
derived from the optional argument ``progress_logger``.
By default, ``progress_logger``
is taken to be the logger with name ``"pyomo.contrib.pyros"``.
The level of the default progress logger is originally set to
:py:obj:`logging.INFO` and, for example, can be set to
:py:obj:`logging.DEBUG` with:

.. doctest::

   >>> import logging
   >>> logging.getLogger("pyomo.contrib.pyros").setLevel(logging.DEBUG)

 
The verbosity of the output log can be adjusted by setting the
:py:mod:`logging` level of the progress logger.
PyROS logs output messages at different :py:mod:`logging` levels,
according to the following table, in which the levels are
arranged in decreasing order of severity.
Messages with a lower level than that of ``progress_logger``
are excluded from the solver log.

.. _table-logging-levels:

.. list-table:: PyROS solver log output at the various standard Python :py:mod:`logging` levels.
   :widths: 10 50
   :header-rows: 1

   * - Logging Level
     - Output Messages
   * - :py:obj:`logging.ERROR`
     - * Information on the subproblem for which an exception was raised
         by a subordinate solver
       * Details about failure of the PyROS coefficient matching routine
   * - :py:obj:`logging.WARNING`
     - * Information about a subproblem not solved to an acceptable status
         by the user-provided subordinate optimizers
       * Invocation of a backup solver for a particular subproblem
       * Caution about solution robustness guarantees in event that
         user passes ``bypass_global_separation=True``
   * - :py:obj:`logging.INFO`
     - * PyROS version, author, and disclaimer information
       * Summary of user options
       * Breakdown of model component statistics
       * Iteration log table
       * Termination details: message, timing breakdown, summary of statistics
   * - :py:obj:`logging.DEBUG`
     - * Progress through the various preprocessing subroutines
       * Termination outcomes and summary of statistics for
         every master feasility, master, and DR polishing problem
       * Progress updates for the separation procedure
       * Separation subproblem initial point infeasibilities
       * Summary of separation loop outcomes: second-stage inequality constraints
         violated, uncertain parameter scenario added to the
         master problem
       * Uncertain parameter scenarios added to the master problem
         thus far

We refer the reader to the
:doc:`official Python logging library documentation <python:library/logging>`
for further guidance on (customization of) Python logger objects;
for a basic tutorial, see the :doc:`logging HOWTO <python:howto/logging>`.
