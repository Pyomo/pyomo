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

import pyomo.environ as pe
from pyomo.contrib.alternative_solutions import aos_utils, solution

def gurobi_generate_solutions(model, num_solutions=10, rel_opt_gap=None, 
                              abs_opt_gap=None, search_mode=2, 
                              solver_options={}, tee=True):
    '''
    Finds alternative optimal solutions for discrete variables using Gurobi's 
    built-in Solution Pool capability. See the Gurobi Solution Pool
    documentation for additional details. 
        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model.
        num_solutions : int
            The maximum number of solutions to generate. This parameter maps to 
            the PoolSolutions parameter in Gurobi.
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
        tee : boolean
            Boolean indicating that the solver output should be displayed.
            
        Returns
        -------
        solutions
            A list of Solution objects.
            [Solution]
    '''
    
    opt = pe.SolverFactory('appsi_gurobi')
    
    for parameter, value in solver_options.items():
        opt.options[parameter] = value
    opt.options['PoolSolutions'] = num_solutions
    opt.options['PoolSearchMode'] = search_mode
    if rel_opt_gap is not None:
        opt.options['PoolGap'] = rel_opt_gap
    if abs_opt_gap is not None:
        opt.options['PoolGapAbs'] = abs_opt_gap
        
    results = opt.solve(model, tee=tee)
    status = results.solver.status
    condition = results.solver.termination_condition

    solutions = []
    if condition == pe.TerminationCondition.optimal:
        solution_count = opt.get_model_attr('SolCount')
        print("{} solutions found.".format(solution_count))
        variables = aos_utils.get_model_variables(model, 'all', 
                                                  include_fixed=True)
        for i in range(solution_count):
            results.solution_loader.load_vars(solution_number=i)
            solutions.append(solution.Solution(model, variables))
    else:
        print(('Model cannot be solved, SolverStatus = {}, '
               'TerminationCondition = {}').format(status.value, 
                                                   condition.value))

    return solutions