#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib import appsi
from pyomo.contrib.alternative_solutions import aos_utils, var_utils, solution

def gurobi_generate_solutions(model, max_solutions=10, rel_opt_gap=None, 
                              abs_opt_gap=None, search_mode=2, 
                              solver_options={}):
    '''
    Finds alternative optimal solutions for discrete variables using Gurobi's 
    built-in Solution Pool capability. See the Gurobi Solution Pool
    documentation for additional details. 
        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model.
        max_solutions : int or None
            The maximum number of solutions to generate. None indictes no upper
            limit. Note, using None could lead to a large number of solutions.
            This parameter maps to the PoolSolutions parameter in Gurobi.
        rel_opt_gap : non-negative float or None
            The relative optimality gap for allowable alternative solutions.
            None implies that there is no limit on the relative optimality gap 
            (i.e. that any feasible solution can be considered by Gurobi).
            This parameter maps to the PoolGap parameter in Gurobi.
        abs_opt_gap : non-negative float or None
            The absolute optimality gap for allowable alternative solutions.
            None implies that there is no limit on the absolute optimality gap 
            (i.e. that any feasible solution can be considered by Gurobi).
            This parameter maps to the PoolGapAbs parameter in Gurobi.
        search_mode : 0, 1, or 2
            Indicates the SolutionPool mode that is used to generate 
            alternative solutions in Gurobi. Mode 2 should typically be used as 
            it finds the top n solutions. Mode 0 finds a single optimal 
            solution (i.e. the standard mode in Gurobi). Mode 1 will generate n 
            solutions without providing guarantees on their quality. This 
            parameter maps to the PoolSearchMode in Gurobi.
        solver_options : dict
            Solver option-value pairs to be passed to the Gurobi solver.
            
        Returns
        -------
        solutions
            A list of Solution objects.
            [Solution]
    '''

    # Input validation
    num_solutions = aos_utils._get_max_solutions(max_solutions)
    assert (isinstance(rel_opt_gap, (float, int)) and rel_opt_gap >= 0) or \
        isinstance(rel_opt_gap, type(None)), \
            'rel_opt_gap must be a non-negative float or None'
    assert (isinstance(abs_opt_gap, (float, int)) and abs_opt_gap >= 0) or \
        isinstance(abs_opt_gap, type(None)), \
            'abs_opt_gap must be a non-negative float or None'
    assert search_mode in [0, 1, 2], 'search_mode must be 0, 1, or 2'
    
    # Configure solver and solve model
    opt = appsi.solvers.Gurobi()
    opt.config.stream_solver = True
    opt.set_instance(model)
    opt.set_gurobi_param('PoolSolutions', num_solutions)
    opt.set_gurobi_param('PoolSearchMode', search_mode)
    if rel_opt_gap is not None:
        opt.set_gurobi_param('PoolGap', rel_opt_gap)
    if abs_opt_gap is not None:
        opt.set_gurobi_param('PoolGapAbs', abs_opt_gap)
    for parameter, value in solver_options.items():
        opt.set_gurobi_param(parameter, abs_opt_gap)
    results = opt.solve(model)
    assert results.termination_condition == \
        appsi.base.TerminationCondition.optimal, \
        'Solver terminated with conditions {}.'.format(
            results.termination_condition)

    # Get model solutions
    solution_count = opt.get_model_attr('SolCount')
    print("Gurobi found {} solutions.".format(solution_count))
    variables = var_utils.get_model_variables(model, 'all', include_fixed=True)
    solutions = []
    for i in range(solution_count):
        results.solution_loader.load_vars(solution_number=i)
        solutions.append(solution.Solution(model, variables))

    return solutions