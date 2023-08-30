# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:49:54 2022

@author: jlgearh

"""

from numpy import dot

from pyomo.core.base.PyomoModel import ConcreteModel
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.contrib.alternative_solutions import aos_utils, var_utils

def enumerate_binary_solutions(model, max_solutions=10, variables='all', 
                               rel_opt_gap=None, abs_gap=None,
                               search_mode='optimal', already_solved=False,
                               solver='gurobi', solver_options={}, tee=False):
    '''Finds alternative optimal solutions for a binary problem.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model
        max_solutions : int or None
            The maximum number of solutions to generate. None indictes no upper
            limit. Note, using None could lead to a large number of solutions.
        variables: 'all', None, Block, or a Collection of Pyomo components
            The binary variables for which alternative solutions will be 
            generated. 'all' or None indicates that all binary variables will 
            be included.
        rel_opt_gap : float or None
            The relative optimality gap for allowable alternative solutions.
            None indicates that a relative gap constraint will not be added to
            the model.
        abs_gap : float or None
            The absolute optimality gap for allowable alternative solutions.
            None indicates that an absolute gap constraint will not be added to
            the model.
        search_mode : 'optimal', 'random', or 'hamming'
            Indicates the mode that is used to generate alternative solutions.
            The optimal mode finds the next best solution. The random mode
            finds an alternative solution in the direction of a random ray. The
            hamming mode iteratively finds solution that maximize the hamming 
            distance from previously discovered solutions.
        already_solved : boolean
            Indicates that the model has already been solved and that the 
            alternative solution search can start from the current solution.
        solver : string
            The solver to be used for alternative solution search.
        solver_options : dict
            Solver option-value pairs to be passed to the solver.
        tee : boolean
            Boolean indicating if the solver output should be displayed.
            
        Returns
        -------
        solutions
            A dictionary of alternative optimal solutions.
            {solution_id: (objective_value,[variable, variable_value])}
    '''

    #assert isinstance(model, ConcreteModel), \
    #    'model parameter must be an instance of a Pyomo Concrete Model'

    # Find the maximum number of solutions to generate
    num_solutions = aos_utils._get_max_solutions(max_solutions)
    if variables == 'all':
        binary_variables = var_utils.get_model_variables(model, 'all',
                                                         include_binary=True)
    else:
        variable_list = var_utils.check_variables(model, variables)
    all_variables = var_utils.get_model_variables(model, 'all')
    orig_objective = aos_utils._get_active_objective(model)

    aos_block = aos_utils._add_aos_block(model)
    aos_block.no_good_cuts = pe.ConstraintList()

    opt = aos_utils._get_solver(solver, solver_options)

    # Repeat until all solutions are found
    solution_number = 0
    solutions = {}
    while solution_number < num_solutions:
    
        # Solve the model unless this is the first solution and the model was 
        # not already solved
        if solution_number > 0 or not already_solved:
            results = opt.solve(model, tee=tee)
    
        if (((results.solver.status == SolverStatus.ok) and
        (results.solver.termination_condition == TerminationCondition.optimal))
        or (already_solved and solution_number == 0)):
            objective_value = pe.value(orig_objective)
            hamming_value = 0
            if solution_number > 0:
                hamming_value = pe.value(aos_block.hamming_objective/solution_number)
            print("Found solution #{}, objective = {}".format(solution_number, 
                                                              hamming_value))
            
            solutions[solution_number] = (objective_value, 
                                          aos_utils.get_solution(model, 
                                                                 all_variables))
            
            if solution_number == 0:
                aos_utils._add_objective_constraint(aos_block, orig_objective, 
                                                    objective_value, 
                                                    rel_opt_gap, abs_gap)
                                
                if search_mode in ['random', 'hamming']:
                    orig_objective.deactivate()
    
            # Add the new solution to the list of previous solutions
            expr = 0
            for var in binary_variables:
                if var.value > 0.5:
                    expr += 1 - var
                else:
                    expr += var
                    
            aos_block.no_good_cuts.add(expr= expr >= 1)
    
            # TODO: Maybe rescale these
            if search_mode == 'hamming':
                if hasattr(aos_block, 'hamming_objective'):
                    aos_block.hamming_objective.expr += expr
                else:
                    aos_block.hamming_objective = pe.Objective(expr=expr,
                                                           sense=pe.maximize)
                
            if search_mode == 'random':
                if hasattr(aos_block, 'random_objective'):
                    aos_block.del_component('random_objective')
                vector = aos_utils._get_random_direction(len(binary_variables))
                idx = 0
                expr = 0
                for var in binary_variables:
                    expr += vector[idx] * var
                    idx += 1
                aos_block.random_objective = \
                    pe.Objective(expr=expr, sense=pe.maximize)
                    
            solution_number += 1
        else:
            print('Algorithm Stopped. Solver Status: {}. Solver Condition: {}.'\
                  .format(results.solver.status,
                          results.solver.termination_condition))
            break
        
    aos_block.deactivate()
    orig_objective.activate()
    
    return solutions

