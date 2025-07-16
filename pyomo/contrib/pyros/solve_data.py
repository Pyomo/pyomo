#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Containers for PyROS subproblem solve results.
"""


class ROSolveResults(object):
    """
    PyROS solver results object.

    Parameters
    ----------
    config : ConfigDict, optional
        User-specified solver settings.
    iterations : int, optional
        Number of iterations required.
    time : float, optional
        Total elapsed time (or wall time), in seconds.
    final_objective_value : float, optional
        Final objective function value to report.
    pyros_termination_condition : pyrosTerminationCondition, optional
        PyROS-specific termination condition.

    Attributes
    ----------
    config : ConfigDict
        User-specified solver settings.
    iterations : int
        Number of iterations required by PyROS.
    time : float
        Total elapsed time (or wall time), in seconds.
    final_objective_value : float
        Final objective function value to report.
        If a nominal objective focus was elected, then the
        value of the nominal objective function is reported.
        If a worst-case objective focus was elected, then
        the value of the worst-case objective function is reported.
    pyros_termination_condition : pyrosTerminationCondition
        Indicator of the manner of termination.
    """

    def __init__(
        self,
        config=None,
        iterations=None,
        time=None,
        final_objective_value=None,
        pyros_termination_condition=None,
    ):
        """Initialize self (see class docstring)."""
        self.config = config
        self.iterations = iterations
        self.time = time
        self.final_objective_value = final_objective_value
        self.pyros_termination_condition = pyros_termination_condition

    def __str__(self):
        """
        Generate string representation of self.
        Does not include any information about `self.config`.
        """
        lines = ["Termination stats:"]
        attr_name_format_dict = {
            "iterations": ("Iterations", "f'{val}'"),
            "time": ("Solve time (wall s)", "f'{val:.3f}'"),
            "final_objective_value": ("Final objective value", "f'{val:.4e}'"),
            "pyros_termination_condition": ("Termination condition", "f'{val}'"),
        }
        attr_desc_pad_length = max(
            len(desc) for desc, _ in attr_name_format_dict.values()
        )
        for attr_name, (attr_desc, fmt_str) in attr_name_format_dict.items():
            val = getattr(self, attr_name)
            val_str = eval(fmt_str) if val is not None else str(val)
            lines.append(f" {attr_desc:<{attr_desc_pad_length}s} : {val_str}")

        return "\n".join(lines)


class MasterResults:
    """
    Result of solving the master problem in a single PyROS iteration.

    Attributes
    ----------
    master_model : ConcreteModel
        Master model.
    feasibility_problem_results : SolverResults
        Feasibility problem subsolver results.
    master_results_list : list of SolverResults
        List of subsolver results for the master problem.
    pyros_termination_condition : None or pyrosTerminationCondition
        PyROS termination status established via solution of
        the master problem.
        If `None`, then no termination status has been established.
    """

    def __init__(
        self,
        master_model=None,
        feasibility_problem_results=None,
        master_results_list=None,
        pyros_termination_condition=None,
    ):
        """Initialize self (see class docstring)."""
        self.master_model = master_model
        self.feasibility_problem_results = feasibility_problem_results
        if master_results_list is None:
            self.master_results_list = []
        else:
            self.master_results_list = list(master_results_list)
        self.pyros_termination_condition = pyros_termination_condition


class SeparationSolveCallResults:
    """
    Container for results of solve attempt for single separation
    problem.

    Parameters
    ----------
    solved_globally : bool
        True if separation problem was solved globally,
        False otherwise.
    results_list : list of pyomo.opt.results.SolverResults, optional
        Pyomo solver results for each subordinate optimizer invoked on
        the separation problem.
        For problems with non-discrete uncertainty set types,
        each entry corresponds to a single subordinate solver.
        For problems with discrete set types, the list may
        be empty (didn't need to use a subordinate solver to
        evaluate optimal separation solution), or the number
        of entries may be as high as the product of the number of
        subordinate local/global solvers provided (including backup)
        and the number of scenarios in the uncertainty set.
    scaled_violations : ComponentMap, optional
        Mapping from second-stage inequality constraints to floats equal
        to their scaled violations by separation problem solution
        stored in this result.
    violating_param_realization : list of float, optional
        Uncertain parameter realization for reported separation
        problem solution.
    auxiliary_param_values : list of float, optional
        Auxiliary parameter values corresponding to the
        uncertain parameter realization `violating_param_realization`.
    variable_values : ComponentMap, optional
        Second-stage DOF and state variable values for reported
        separation problem solution.
    found_violation : bool, optional
        True if violation of second-stage inequality constraint
        (i.e. constraint expression value) by reported separation
        solution was found to exceed tolerance, False otherwise.
    time_out : bool, optional
        True if PyROS time limit reached attempting to solve the
        separation problem, False otherwise.
    subsolver_error : bool, optional
        True if subsolvers found to be unable to solve separation
        problem of interest, False otherwise.
    discrete_set_scenario_index : None or int, optional
        If discrete set used to solve the problem, index of
        `violating_param_realization` as listed in the
        `scenarios` attribute of a ``DiscreteScenarioSet``
        instance. If discrete set not used, pass None.

    Attributes
    ----------
    solved_globally
    results_list
    scaled_violations
    violating_param_realizations
    auxiliary_param_values
    variable_values
    found_violation
    time_out
    subsolver_error
    discrete_set_scenario_index
    """

    def __init__(
        self,
        solved_globally,
        results_list=None,
        scaled_violations=None,
        violating_param_realization=None,
        auxiliary_param_values=None,
        variable_values=None,
        found_violation=None,
        time_out=None,
        subsolver_error=None,
        discrete_set_scenario_index=None,
    ):
        """Initialize self (see class docstring)."""
        self.results_list = results_list
        self.solved_globally = solved_globally
        self.scaled_violations = scaled_violations
        self.violating_param_realization = violating_param_realization
        self.auxiliary_param_values = auxiliary_param_values
        self.variable_values = variable_values
        self.found_violation = found_violation
        self.time_out = time_out
        self.subsolver_error = subsolver_error
        self.discrete_set_scenario_index = discrete_set_scenario_index

    def termination_acceptable(self, acceptable_terminations):
        """
        Return True if termination condition for at least
        one result in `self.results_list` is in list
        of pre-specified acceptable terminations, False otherwise.

        Parameters
        ----------
        acceptable_terminations : set of pyomo.opt.TerminationCondition
            Acceptable termination conditions.

        Returns
        -------
        bool
        """
        return any(
            res.solver.termination_condition in acceptable_terminations
            for res in self.results_list
        )


class DiscreteSeparationSolveCallResults:
    """
    Container for results of solve attempt for single separation
    problem.

    Parameters
    ----------
    solved_globally : bool
        True if separation problems solved to global optimality,
        False otherwise.
    solver_call_results : dict
        Mapping from discrete uncertainty set scenario list
        indexes to solver call results for separation problems
        subject to the scenarios.
    second_stage_ineq_con : Constraint
        Separation problem second-stage inequality constraint for which
        `self` was generated.

    Attributes
    ----------
    solved_globally
    solver_call_results
    second_stage_ineq_con
    """

    def __init__(
        self, solved_globally, solver_call_results=None, second_stage_ineq_con=None
    ):
        """Initialize self (see class docstring)."""
        self.solved_globally = solved_globally
        self.solver_call_results = solver_call_results
        self.second_stage_ineq_con = second_stage_ineq_con

    @property
    def time_out(self):
        """
        bool : True if there is a time out status for at least one of
        the ``SeparationSolveCallResults`` objects listed in `self`,
        False otherwise.
        """
        return any(res.time_out for res in self.solver_call_results.values())

    @property
    def subsolver_error(self):
        """
        bool : True if there is a subsolver error status for all
        of the ``SeparationSolveCallResults`` objects listed
        in `self`, False otherwise.
        """
        return all(res.subsolver_error for res in self.solver_call_results.values())


class SeparationLoopResults:
    """
    Container for results of all separation problems solved
    to a single desired optimality target (local or global).

    Parameters
    ----------
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.
    solver_call_results : ComponentMap
        Mapping from second-stage inequality constraints to corresponding
        ``SeparationSolveCallResults`` objects.
    worst_case_ss_ineq_con : None or Constraint
        Second-stage inequality constraint mapped to
        ``SeparationSolveCallResults``
        object in `self` corresponding to maximally violating
        separation problem solution.
    all_discrete_scenarios_exhausted : bool, optional
        For problems with discrete uncertainty sets,
        True if all scenarios were explicitly accounted for in master
        (which occurs if there have been
        as many PyROS iterations as there are scenarios in the set)
        False otherwise.

    Attributes
    ----------
    solved_globally : bool
        True if global solver was used, False otherwise.
    solver_call_results : ComponentMap
        Mapping from second-stage inequality constraints to corresponding
        ``SeparationSolveCallResults`` objects.
    worst_case_ss_ineq_con : None or ConstraintData
        Worst-case second-stage inequality constraint.
    all_discrete_scenarios_exhausted : bool
        True if all scenarios of the discrete set were exhausted
        already explicitly accounted for in the master problems,
        False otherwise.
    """

    def __init__(
        self,
        solved_globally,
        solver_call_results,
        worst_case_ss_ineq_con,
        all_discrete_scenarios_exhausted=False,
    ):
        """Initialize self (see class docstring)."""
        self.solver_call_results = solver_call_results
        self.solved_globally = solved_globally
        self.worst_case_ss_ineq_con = worst_case_ss_ineq_con
        self.all_discrete_scenarios_exhausted = all_discrete_scenarios_exhausted

    @property
    def found_violation(self):
        """
        bool : True if separation solution for at least one
        ``SeparationSolveCallResults`` object listed in self
        was reported to violate its corresponding second-stage
        inequality constraint, False otherwise.
        """
        return any(
            solver_call_res.found_violation
            for solver_call_res in self.solver_call_results.values()
        )

    @property
    def violating_param_realization(self):
        """
        None or list of float : Uncertain parameter values for
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_ss_ineq_con``.
        If ``self.worst_case_ss_ineq_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_ss_ineq_con is not None:
            return self.solver_call_results[
                self.worst_case_ss_ineq_con
            ].violating_param_realization
        else:
            return None

    @property
    def auxiliary_param_values(self):
        """
        None or list of float : Auxiliary parameter values for the
        maximially violating separation problem solution.
        """
        if self.worst_case_ss_ineq_con is not None:
            return self.solver_call_results[
                self.worst_case_ss_ineq_con
            ].auxiliary_param_values
        else:
            return None

    @property
    def scaled_violations(self):
        """
        None or ComponentMap : Scaled second-stage inequality
        constraint violations
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_ss_ineq_con``.
        If ``self.worst_case_ss_ineq_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_ss_ineq_con is not None:
            return self.solver_call_results[
                self.worst_case_ss_ineq_con
            ].scaled_violations
        else:
            return None

    @property
    def violating_separation_variable_values(self):
        """
        None or ComponentMap : Second-stage and state variable values
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_ss_ineq_con``.
        If ``self.worst_case_ss_ineq_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_ss_ineq_con is not None:
            return self.solver_call_results[self.worst_case_ss_ineq_con].variable_values
        else:
            return None

    @property
    def violated_second_stage_ineq_cons(self):
        """
        list of Constraint : Second-stage inequality constraints
        for which violation found.
        """
        return [
            con
            for con, solver_call_results in self.solver_call_results.items()
            if solver_call_results.found_violation
        ]

    @property
    def subsolver_error(self):
        """
        bool : Return True if subsolver error reported for
        at least one ``SeparationSolveCallResults`` stored in
        `self` and no violations are found, False otherwise.
        """
        return (
            any(
                solver_call_res.subsolver_error
                for solver_call_res in self.solver_call_results.values()
            )
            and not self.found_violation
        )

    @property
    def time_out(self):
        """
        bool : Return True if time out reported for
        at least one ``SeparationSolveCallResults`` stored in
        `self`, False otherwise.
        """
        return any(
            solver_call_res.time_out
            for solver_call_res in self.solver_call_results.values()
        )


class SeparationResults:
    """
    Container for results of PyROS separation problem routine.

    Parameters
    ----------
    local_separation_loop_results : None or SeparationLoopResults
        Local separation problem loop results.
    global_separation_loop_results : None or SeparationLoopResults
        Global separation problem loop results.

    Attributes
    ----------
    local_separation_loop_results : None or SeparationLoopResults
        Local separation results. If separation problems
        were not solved locally, then this attribute is set
        to None.
    global_separation_loop_results : None or SeparationLoopResults
        Global separation results. If separation problems
        were not solved globally, then this attribute is set
        to None.
    """

    def __init__(self, local_separation_loop_results, global_separation_loop_results):
        """Initialize self (see class docstring)."""
        self.local_separation_loop_results = local_separation_loop_results
        self.global_separation_loop_results = global_separation_loop_results

    @property
    def time_out(self):
        """
        bool : True if time out found for local or global
        separation loop, False otherwise.
        """
        local_time_out = (
            self.solved_locally and self.local_separation_loop_results.time_out
        )
        global_time_out = (
            self.solved_globally and self.global_separation_loop_results.time_out
        )
        return local_time_out or global_time_out

    @property
    def subsolver_error(self):
        """
        bool : True if subsolver error found for local or global
        separation loop, False otherwise.
        """
        local_subsolver_error = (
            self.solved_locally and self.local_separation_loop_results.subsolver_error
        )
        global_subsolver_error = (
            self.solved_globally and self.global_separation_loop_results.subsolver_error
        )
        return local_subsolver_error or global_subsolver_error

    @property
    def solved_locally(self):
        """
        bool : true if local separation loop was invoked,
        False otherwise.
        """
        return self.local_separation_loop_results is not None

    @property
    def solved_globally(self):
        """
        bool : True if global separation loop was invoked,
        False otherwise.
        """
        return self.global_separation_loop_results is not None

    def get_violating_attr(self, attr_name):
        """
        If separation problems solved globally, returns
        value of attribute of global separation loop results.

        Otherwise, if separation problems solved locally,
        returns value of attribute of local separation loop results.
        If local separation loop results specified, return
        value of attribute of local separation loop results.

        Otherwise, if global separation loop results specified,
        return value of attribute of global separation loop
        results.

        Otherwise, return None.

        Parameters
        ----------
        attr_name : str
            Name of attribute to be retrieved. Should be
            valid attribute name for object of type
            ``SeparationLoopResults``.

        Returns
        -------
        object
            Attribute value.
        """
        return getattr(self.main_loop_results, attr_name, None)

    @property
    def all_discrete_scenarios_exhausted(self):
        """
        bool : For problems where the uncertainty set is of type
        DiscreteScenarioSet,
        True if last master problem solved explicitly
        accounts for all scenarios in the uncertainty set,
        False otherwise.
        """
        return self.get_violating_attr("all_discrete_scenarios_exhausted")

    @property
    def worst_case_ss_ineq_con(self):
        """
        ConstraintData : Second-stage inequality constraint
        corresponding to the
        separation solution chosen for the next master problem.
        """
        return self.get_violating_attr("worst_case_ss_ineq_con")

    @property
    def main_loop_results(self):
        """
        SeparationLoopResults : Main separation loop results.
        In particular, this is considered to be the global
        loop result if solved globally, and the local loop
        results otherwise.
        """
        if self.solved_globally:
            return self.global_separation_loop_results
        return self.local_separation_loop_results

    @property
    def found_violation(self):
        """
        bool : True if ``found_violation`` attribute for
        main separation loop results is True, False otherwise.
        """
        found_viol = self.get_violating_attr("found_violation")
        if found_viol is None:
            found_viol = False
        return found_viol

    @property
    def violating_param_realization(self):
        """
        None or list of float : Uncertain parameter values
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_ss_ineq_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr("violating_param_realization")

    @property
    def auxiliary_param_values(self):
        """
        None or list of float: Auxiliary parameter values accompanying
        `self.violating_param_realization`.
        """
        return self.get_violating_attr("auxiliary_param_values")

    @property
    def scaled_violations(self):
        """
        None or ComponentMap :
        Scaled second-stage inequality constraint violations
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_ss_ineq_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr("scaled_violations")

    @property
    def violating_separation_variable_values(self):
        """
        None or ComponentMap : Second-stage and state variable values
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_ss_ineq_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr("violating_separation_variable_values")

    @property
    def violated_second_stage_ineq_cons(self):
        """
        Return list of violated second-stage inequality constraints.
        """
        return self.get_violating_attr("violated_second_stage_ineq_cons")

    @property
    def robustness_certified(self):
        """
        bool : Return True if separation results certify that
        first-stage solution is robust, False otherwise.
        """
        assert self.solved_locally or self.solved_globally

        if self.time_out or self.subsolver_error:
            return False

        if self.solved_locally:
            heuristically_robust = (
                not self.local_separation_loop_results.found_violation
            )
        else:
            heuristically_robust = None

        if self.solved_globally:
            is_robust = not self.global_separation_loop_results.found_violation
        else:
            # global separation bypassed, either
            # because uncertainty set is discrete
            # or user opted to bypass global separation
            is_robust = heuristically_robust

        return is_robust
