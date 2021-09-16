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