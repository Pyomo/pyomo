'''
Objects to contain all model data and solve results for the ROSolver
'''


class ROSolveResults(object):
    '''
    Container for solve-instance data returned to the user after solving with PyROS.

    Attributes:
        :pyros_termination_condition: termination condition of the PyROS algorithm
        :config: the config block for this solve instance
        :time: Total solver CPU time
        :iterations: total iterations done by PyROS solver
        :final_objective_value: objective function value at termination
    '''

    pass


class MasterProblemData(object):
    '''
    Container for the grcs master problem

    Attributes:
        :master_model: master problem model object
        :base_model: block representing the original model object
        :iteration: current iteration of the algorithm
    '''


class SeparationProblemData(object):
    '''
    Container for the grcs separation problem

    Attributes:
        :separation_model: separation problem model object
        :points_added_to_master: list of parameter violations added to the master problem over the course of the algorithm
        :separation_problem_subsolver_statuses: list of subordinate sub-solver statuses throughout separations
        :total_global_separation_solvers: Counter for number of times global solvers were employed in separation
        :constraint_violations: List of constraint violations identified in separation
    '''

    pass


class MasterResult(object):
    """Data class for master problem results data.

    Attributes:
         - termination_condition: Solver termination condition
         - fsv_values: list of design variable values
         - ssv_values: list of control variable values
         - first_stage_objective: objective contribution due to first-stage degrees of freedom
         - second_stage_objective: objective contribution due to second-stage degrees of freedom
         - grcs_termination_condition: the conditions under which the grcs terminated
                                       (max_iter, robust_optimal, error)
         - pyomo_results: results object from solve() statement

    """


class SeparationResult(object):
    """Data class for master problem results data.

    Attributes:
         - termination_condition: Solver termination condition
         - violation_found: True if a violating parameter realization was identified in separation. For a given
            separation objective function, it is considered a violation only if the parameter realization led to a
            violation of the corresponding ineq. constraint used to define that objective
         - is_global: True if separation problem differed to global solver, False if local solver
         - separation_model: Pyomo model for separation problem at optimal solution
         - control_var_values: list of control variable values
         - violating_param_realization: list for the values of the uncertain_params identified as a violation
         - list_of_violations: value of constraints violation for each ineq. constraint considered
            in separation against the violation in violating_param_realizations
         - pyomo_results: results object from solve() statement

    """


class SeparationSolveCallResults:
    """
    Container for results of solve attempt for single separation
    problem.

    Parameters
    ----------
    solved_globally : bool
        True if separation problem was solved globally,
        False otherwise.
    results_list : list of pyomo.opt.results.SolveResults, optional
        Solve results for each subordinate optimizer invoked on the
        separation problem.
    list_of_scaled_violations : list of float, optional
        Normalized violation of each performance constraint considered
        by the separation problem solution.
    violating_param_realization : list of float, optional
        Uncertain parameter realization for which maximum constraint
        violation was found.
    found_violation : bool, optional
        True if uncertain parameter realization for which performance
        corresponding to separation problem of interest is violated
        was found, False otherwise.
    time_out : bool, optional
        True if PyROS time limit reached attempting to solve the
        separation problem, False otherwise.

    Attributes
    ----------
    solved_globally : bool
        True if separation problem was solved globally,
        False otherwise.
    results_list : list of pyomo.opt.results.SolveResults
        Solve results for each subordinate optimizer invoked on the
        separation problem.
    list_of_scaled_violations : list of float
        Normalized violation of each performance constraint considered
        by the separation problem solution.
    violating_param_realization : list of float
        Uncertain parameter realization for which maximum constraint
        violation was found.
    found_violation : bool, optional
        True if uncertain parameter realization for which performance
        corresponding to separation problem of interest is violated
        was found, False otherwise.
    time_out : bool
        True if PyROS time limit reached attempting to solve the
        separation problem, False otherwise.
    """

    def __init__(
            self,
            solved_globally,
            results_list=None,
            list_of_scaled_violations=None,
            violating_param_realization=None,
            found_violation=None,
            time_out=None,
            ):
        """Initialize self (see class docstring).

        """
        self.results_list = results_list
        self.solved_globally = solved_globally
        self.list_of_scaled_violations = list_of_scaled_violations
        self.violating_param_realization = violating_param_realization
        self.found_violation = found_violation
        self.time_out = time_out

    def termination_acceptable(self, acceptable_terminations):
        """
        Return True if termination condition for at least
        one result in `self.results_list` is in list
        of pre-specified acceptable terminations, False otherwise.

        Parameters
        ----------
        acceptable_terminations : set of pyomo.opt.TerminationCondition
            Acceptable termination conditions.
        """
        return any(
            res.solver.termination_condition in acceptable_terminations
            for res in self.results_list
        )


class SeparationLoopResults:
    """
    Container for results of all separation problems solved
    to a single desired optimality target (local or global).

    Parameters
    ----------
    solve_data_list : list of SeparationSolveCallResult
        Solver call results for each separation problem
        (i.e. each performance constraint).
    violating_param_realization : None or list of float
        Uncertain parameter realization corresponding to a separation
        problem solution found to have violated a performance
        constraint.
        If no such realization found, pass None.
    violations : None or list of float
        Value of violation (performance constraint function) at
        separation solution corresponding to
        `violating_param_realization`.
    solve_time : float
        Total time spent by subsolvers for problems addressed
        in the loop, in seconds.
    subsolver_error : bool
        True if subordinate optimizers failed to solve at least one
        separation problem to desired optimality target,
        False otherwise.
    time_out : bool
        True if PyROS time limit reached during course of solving a
        separation subproblem, False otherwise.
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.

    Attributes
    ----------
    solve_data_list : list of SeparationSolveCallResult
        Solver call results for each separation problem
        (i.e. each performance constraint).
    violating_param_realization : None or list of float
        Uncertain parameter realization corresponding to a separation
        problem solution found to have violated a performance
        constraint.
        If no such realization found, pass None.
    violations : None or list of float
        Value of violation (performance constraint function) at
        separation solution corresponding to
        `violating_param_realization`.
    solve_time : float
        Total time spent by subsolvers for problems addressed
        in the loop, in seconds.
    subsolver_error : bool
        True if subordinate optimizers failed to solve at least one
        separation problem to desired optimality target,
        False otherwise.
    time_out : bool
        True if PyROS time limit reached during course of solving a
        separation subproblem, False otherwise.
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.
    found_violation
    """

    def __init__(
            self,
            solve_data_list,
            violating_param_realization,
            violations,
            solve_time,
            subsolver_error,
            time_out,
            solved_globally,
            ):
        """Initialize self (see class docstring).

        """
        self.solve_data_list = solve_data_list
        self.violating_param_realization = violating_param_realization
        self.violations = violations
        self.solve_time = solve_time
        self.subsolver_error = subsolver_error
        self.time_out = time_out
        self.solved_globally = solved_globally

    @property
    def found_violation(self):
        """
        bool : True if at least one performance constraint was found
        to be violated, False otherwise.
        """
        return self.violating_param_realization is not None


class SeparationResults:
    """
    Container for results of PyROS separation problem routine.

    Parameters
    ----------
    solve_data_list : list of SeparationSolveCallResult
        Solver call results for each separation problem
        (i.e. each performance constraint).
    violating_param_realization : None or list of float
        Uncertain parameter realization corresponding to a separation
        problem solution found to have violated a performance
        constraint.
        If no such realization found, pass None.
    violations : list of float
        Value of violation (performance constraint function) at
        separation solution corresponding to
        `violating_param_realization`.
    local_solve_time : float
        Total time required by local subsolvers, in seconds.
    global_solve_time : float
        Total time required by global subsolvers, in seconds.
    subsolver_error : bool
        True if subordinate optimizers failed to solve at least one
        separation problem to desired optimality target,
        False otherwise.
    time_out : bool
        True if PyROS time limit reached during course of solving
        separation problems, False otherwise.
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.
    robustness_certified : bool
        True if robustness of candidate first-stage solution assessed
        is certified, False otherwise.

    Attributes
    ----------
    solve_data_list : list of SeparationSolveCallResult
        Solver call results for each separation problem
        (i.e. each performance constraint).
    violating_param_realization : None or list of float
        Uncertain parameter realization corresponding to a separation
        problem solution found to have violated a performance
        constraint.
        If no such realization found, pass None.
    violations : list of float
        Value of violation (performance constraint function) at
        separation solution corresponding to
        `violating_param_realization`.
    local_solve_time : float
        Total time required by local subsolvers, in seconds.
    global_solve_time : float
        Total time required by global subsolvers, in seconds.
    subsolver_error : bool
        True if subordinate optimizers failed to solve at least one
        separation problem to desired optimality target,
        False otherwise.
    time_out : bool
        True if PyROS time limit reached during course of solving
        separation problems, False otherwise.
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.
    robustness_certified : bool
        True if robustness of candidate first-stage solution assessed
        is certified, False otherwise.
    """

    def __init__(
            self,
            solve_data_list,
            violating_param_realization,
            violations,
            local_solve_time,
            global_solve_time,
            subsolver_error,
            time_out,
            solved_globally,
            robustness_certified,
            ):
        """Initialize self (see class docstring).

        """
        self.solve_data_list = solve_data_list
        self.violating_param_realization = violating_param_realization
        self.violations = violations
        self.local_solve_time = local_solve_time
        self.global_solve_time = global_solve_time
        self.subsolver_error = subsolver_error
        self.time_out = time_out
        self.solved_globally = solved_globally
        self.robustness_certified = robustness_certified

    @property
    def violation_found(self):
        """
        bool: True if at least one performance constraint
        was found to be violated, False otherwise.
        """
        return self.violating_param_realization is not None
