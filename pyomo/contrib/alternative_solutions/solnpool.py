from pyomo.contrib import appsi
from pyomo.contrib.alternative_solutions import aos_utils, var_utils, solution

def gurobi_generate_solutions(model, max_solutions=10, rel_opt_gap=0.0, 
                              abs_opt_gap=0.0, search_mode=2, 
                              round_discrete_vars=True, solver_options={}):
    '''
    Finds alternative optimal solutions for discrete variables using Gurobi's 
    built-in Solution Pool capability. See the Gurobi Solution Pool
    documentation for additional details. This function uses the Gurobi
    Auto-Persistent Pyomo Solver interface (appsi).

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model
        max_solutions : int or None
            The maximum number of solutions to generate. None indictes no upper
            limit. Note, using None could lead to a large number of solutions.
            This parameter maps to the PoolSolutions parameter in Gurobi.
        rel_opt_gap : non-negative float
            The relative optimality gap for allowable alternative solutions.
            This parameter maps to the PoolGap parameter in Gurobi.
        abs_opt_gap : non-negative float
            The absolute optimality gap for allowable alternative solutions.
            This parameter maps to the PoolGapAbs parameter in Gurobi.
        search_mode : 0, 1, or 2
            Indicates the mode that is used to generate alternative solutions.
            Mode 2 should typically be used as it finds the best n solutions. 
            Mode 0 finds a single optimal solution. Mode 1 will generate n 
            solutions without providing guarantees on their quality. This 
            parameter maps to the PoolSearchMode in Gurobi.
        round_discrete_vars : boolean
            Boolean indicating that discrete values should be rounded to the 
            nearest integer in the solutions results.
        solver_options : dict
            Solver option-value pairs to be passed to the solver.
            
        Returns
        -------
        solutions
            A list of solution dictionaries.
            [solution]
    '''

    # Validate inputs
    aos_utils._is_concrete_model(model)
    num_solutions = aos_utils._get_max_solutions(max_solutions)
    assert isinstance(rel_opt_gap, float) and rel_opt_gap >= 0, \
                          'rel_opt_gap must be a non-negative float'
    assert isinstance(abs_opt_gap, float) and abs_opt_gap >= 0, \
                          'abs_opt_gap must be a non-negative float'
    assert search_mode in [0, 1, 2], 'search_mode must be 0, 1, or 2'
    assert isinstance(round_discrete_vars, bool), \
        'round_discrete_vars must be a Boolean'
    
    # Configure solver and solve model
    opt = appsi.solvers.Gurobi()
    opt.config.stream_solver = True
    opt.set_instance(model)
    opt.set_gurobi_param('PoolSolutions', num_solutions)
    opt.set_gurobi_param('PoolSearchMode', search_mode)
    opt.set_gurobi_param('PoolGap', rel_opt_gap)
    opt.set_gurobi_param('PoolGapAbs', abs_opt_gap)
    results = opt.solve(model)
    assert results.termination_condition == \
        appsi.base.TerminationCondition.optimal, \
        'Solver terminated with conditions {}.'.format(
            results.termination_condition)

    # Get model solutions
    solution_count = opt.get_model_attr('SolCount')
    print("Gurobi found {} solutions".format(solution_count))
    variables = var_utils.get_model_variables(model, 'all', include_fixed=True)
    solutions = []
    for i in range(solution_count):
        results.solution_loader.load_vars(solution_number=i)
        solutions.append(solution.get_model_solution(model, variables))

    return solutions