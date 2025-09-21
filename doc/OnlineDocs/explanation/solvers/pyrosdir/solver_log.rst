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
is called to solve a robust optimzation problem,
your console output will, by default, look like this:


.. _solver-log-snippet:

.. code-block:: text
   :caption: PyROS solver output log for the :ref:`two-stage problem example <example-two-stg>`.
   :linenos:

   ==============================================================================
   PyROS: The Pyomo Robust Optimization Solver, v1.3.9.
          Pyomo version: 6.9.5.dev0 (devel {main})
          Commit hash: unknown
          Invoked at UTC 2025-09-21T00:00:00.000000+00:00
   
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
    nominal_uncertain_param_vals=[1, 1]
    decision_rule_order=1
    solve_master_globally=True
    max_iter=-1
    robust_feasibility_tolerance=0.0001
    separation_priority_order={}
    progress_logger=<PreformattedLogger pyomo.contrib.pyros (INFO)>
    backup_local_solvers=[]
    backup_global_solvers=[]
    subproblem_file_directory=None
    subproblem_format_options={'bar': {'symbolic_solver_labels': True}}
    bypass_local_separation=False
    bypass_global_separation=False
    p_robustness={}
   ------------------------------------------------------------------------------
   Preprocessing...
   Done preprocessing; required wall time of 0.587s.
   ------------------------------------------------------------------------------
   Model Statistics:
     Number of variables : 8
       Epigraph variable : 1
       First-stage variables : 1
       Second-stage variables : 1 (1 adj.)
       State variables : 2 (2 adj.)
       Decision rule variables : 3
     Number of uncertain parameters : 2 (2 eff.)
     Number of constraints : 12
       Equality constraints : 3
         Coefficient matching constraints : 0
         Other first-stage equations : 0
         Second-stage equations : 2
         Decision rule equations : 1
       Inequality constraints : 9
         First-stage inequalities : 0
         Second-stage inequalities : 9
   ------------------------------------------------------------------------------
   Itn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)
   ------------------------------------------------------------------------------
   0     5.4079e+03  -            -            3       7.9226e+00   0.815        
   1     5.4079e+03  6.0451e-10   1.0717e-10   2       1.0250e-01   1.194        
   2     6.5403e+03  1.0018e-01   7.4564e-03   1       1.7142e-03   1.604        
   3     6.5403e+03  1.9372e-16   3.6853e-06   2       1.6673e-03   1.993        
   4     6.5403e+03  0.0000e+00   2.9067e-06   0       9.8487e-05g  2.969        
   ------------------------------------------------------------------------------
   Robust optimal solution identified.
   ------------------------------------------------------------------------------
   Timing breakdown:
   
   Identifier                ncalls   cumtime   percall      %
   -----------------------------------------------------------
   main                           1     2.970     2.970  100.0
        ------------------------------------------------------
        dr_polishing              4     0.227     0.057    7.6
        global_separation         9     0.486     0.054   16.4
        local_separation         45     0.739     0.016   24.9
        master                    5     0.672     0.134   22.6
        master_feasibility        4     0.095     0.024    3.2
        preprocessing             1     0.587     0.587   19.8
        other                   n/a     0.164       n/a    5.5
        ======================================================
   ===========================================================
   
   ------------------------------------------------------------------------------
   Termination stats:
    Iterations            : 5
    Solve time (wall s)   : 2.970
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
* **Summary of solver options** (lines 20--41): Enumeration of
  specifications for optional arguments to the solver.
* **Preprocessing information** (lines 42--44):
  Wall time required for preprocessing
  the deterministic model and associated components,
  i.e., standardizing model components and adding the decision rule
  variables and equations.
* **Model component statistics** (lines 45--62):
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
* **Iteration log table** (lines 63--70):
  Summary information on the problem iterates and subproblem outcomes.
  The constituent columns are defined in detail in
  :ref:`the table that follows <table-iteration-log-columns>`.
* **Termination message** (lines 71--72): One-line message briefly summarizing
  the reason the solver has terminated.
* **Timing statistics** (lines 73--89):
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

* **Final result** (lines 90--95):
  A printout of the
  :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`
  object that is finally returned.
* **Exit message** (lines 96--97): Confirmation that the
  solver has been exited properly.

The iteration log table is designed to provide, in a concise manner,
important information about the progress of the iterative algorithm for
the problem of interest.
The constituent columns are defined in the
table below.

.. _table-iteration-log-columns:

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
For example, the level of the default logger can be set to
:py:obj:`logging.DEBUG` with:

.. doctest::

   >>> import logging
   >>> logging.getLogger("pyomo.contrib.pyros").setLevel(logging.DEBUG)

We refer the reader to the
:doc:`official Python logging library documentation <python:library/logging>`
for further guidance on (customization of) Python logger objects.
