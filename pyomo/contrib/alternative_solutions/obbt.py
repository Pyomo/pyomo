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
from pyomo.contrib.alternative_solutions import  aos_utils, var_utils

def obbt_analysis(model, variables='all', rel_opt_gap=None, abs_opt_gap=None, 
                  refine_discrete_bounds=True, warmstart=True, solver='gurobi', 
                  solver_options={}, tee=False):
    '''
    Calculates the bounds on each variable by solving a series of min and max 
    optimization problems where each variable is used as the objective function
    This can be applied to any class of problem supported by the selected 
    solver.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model.
        variables: 'all' or a collection of Pyomo _GenereralVarData variables
            The variables for which bounds will be generated. 'all' indicates 
            that all variables will be included. Alternatively, a collection of
            _GenereralVarData variables can be provided.
        rel_opt_gap : float or None
            The relative optimality gap for the original objective for which 
            variable bounds will be found. None indicates that a relative gap 
            constraint will not be added to the model.
        abs_opt_gap : float or None
            The absolute optimality gap for the original objective for which 
            variable bounds will be found. None indicates that an absolute gap 
            constraint will not be added to the model.
        refine_discrete_bounds : boolean
            Boolean indicating that new constraints should be added to the 
            model at each iteration to tighten the bounds for discrete
            variables.
        warmstart : boolean
            Boolean indicating that the solver should be warmstarted from the 
            best previously discovered solution.
        solver : string
            The solver to be used.
        solver_options : dict
            Solver option-value pairs to be passed to the solver.
        use_appsi : boolean
            Boolean indicating if the the APPSI persistent solver interface
            should be used. To use APPSI pass the base solver name and set this
            input to true. E.g., passing 'gurobi' as the solver and a value of 
            true will create an instance of an 'appsi_gurobi' solver.
        tee : boolean
            Boolean indicating that the solver output should be displayed.
            
        Returns
        -------
        variable_ranges
            A Pyomo ComponentMap containing the bounds for each variable.
            {variable: (lower_bound, upper_bound)}. A None value indicates
            the solver encountered an issue.
    '''
    
    print('STARTING OBBT ANALYSIS')
    if variables == 'all' or warmstart:
        all_variables = var_utils.get_model_variables(model, 'all',
                                                      include_fixed=False)
        variable_list = all_variables
    if warmstart:
        solutions = pe.ComponentMap()
        for var in all_variables:
            solutions[var] = []
    
    num_vars = len(variable_list)
    print('Analyzing {} variables ({} total solves).'.format(num_vars, 
                                                            2 * num_vars))
    orig_objective = aos_utils._get_active_objective(model)
    
    opt = pe.SolverFactory(solver)
    for parameter, value in solver_options.items():
        opt.options[parameter] = value
    use_appsi = False
    if 'appsi' in solver:
        use_appsi = True    
    print('Peforming initial solve of model.')
    results = opt.solve(model, warmstart=warmstart, tee=tee)
    status = results.solver.status
    condition = results.solver.termination_condition
    if condition != pe.TerminationCondition.optimal:
        raise Exception(('OBBT cannot be applied, SolverStatus = {}, '
                         'TerminationCondition = {}').format(status.value, 
                                                             condition.value))
    if warmstart:
        _add_solution(solutions)
    orig_objective_value = pe.value(orig_objective)
    print('Found optimal solution, value = {}.'.format(orig_objective_value))
    aos_block = aos_utils._add_aos_block(model, name='_obbt')
    print('Added block {} to the model.'.format(aos_block))
    obj_constraints = aos_utils._add_objective_constraint(aos_block, 
                                                          orig_objective, 
                                                          orig_objective_value, 
                                                          rel_opt_gap, 
                                                          abs_opt_gap)   
    new_constraint = False
    if len(obj_constraints) > 0:
        new_constraint = True
    orig_objective.deactivate()
    
    if use_appsi:
        opt.update_config.check_for_new_or_removed_constraints = new_constraint
        opt.update_config.check_for_new_or_removed_vars = False
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.check_for_new_objective = True
        opt.update_config.update_constraints = False
        opt.update_config.update_vars = False
        opt.update_config.update_params = False
        opt.update_config.update_named_expressions = False
        opt.update_config.update_objective = False
        opt.update_config.treat_fixed_vars_as_params = False
        
    variable_bounds = pe.ComponentMap()
 
    senses = [(pe.minimize, 'LB'), (pe.maximize, 'UB')]
    
    iteration = 1
    total_iterations = len(senses) *  num_vars
    for idx in range(len(senses)):
        sense = senses[idx][0]
        bound_dir = senses[idx][1]
            
        for var in variable_list:
            if idx == 0:
                variable_bounds[var] = [None, None]
        
            if hasattr(aos_block, 'var_objective'):
                aos_block.del_component('var_objective')
            
            aos_block.var_objective = pe.Objective(expr=var, sense=sense)  
            
            if warmstart:
                _update_values(var, bound_dir, solutions)
            
            if use_appsi:
                opt.update_config.check_for_new_or_removed_constraints = \
                    new_constraint
            results = opt.solve(model, warmstart=warmstart, tee=tee)
            new_constraint = False
            status = results.solver.status
            condition = results.solver.termination_condition
            if condition == pe.TerminationCondition.optimal:
                if warmstart:
                    _add_solution(solutions)
                obj_val = pe.value(var)
                variable_bounds[var][idx] = obj_val

                if refine_discrete_bounds and not var.is_continuous():
                    if sense == pe.minimize and var.lb < obj_val:
                        bound_name = var.name + '_' + str.lower(bound_dir)
                        bound = pe.Constraint(expr= var >= obj_val)
                        setattr(aos_block, bound_name, bound)
                        new_constraint = True
                    
                    if sense == pe.maximize and var.ub > obj_val:
                        bound_name = var.name + '_' + str.lower(bound_dir)
                        bound = pe.Constraint(expr= var <= obj_val)
                        setattr(aos_block, bound_name, bound)
                        new_constraint = True
                    
            # An infeasibleOrUnbounded status code will imply the problem is
            # unbounded since feasibility has been established previously
            elif (condition == pe.TerminationCondition.infeasibleOrUnbounded or 
                  condition == pe.TerminationCondition.unbounded):
                if sense == pe.minimize:
                    variable_bounds[var][idx] = float('-inf')
                else:
                    variable_bounds[var][idx] = float('inf')
            else:
                print(('Unexpected condition for the variable {} {} problem.'
                       'SolverStatus = {}, TerminationCondition = {}').\
                      format(var.name, bound_dir, status.value, 
                             condition.value))
            
            var_value = variable_bounds[var][idx]
            print('Iteration {}/{}: {}_{} = {}'.format(iteration, 
                                                       total_iterations,
                                                       var.name, 
                                                       bound_dir, 
                                                       var_value))
            
            if idx == 1:
                variable_bounds[var] = tuple(variable_bounds[var])
                
            iteration += 1

        
    aos_block.deactivate()
    orig_objective.activate()
    
    print('COMPLETED OBBT ANALYSIS')
    
    return variable_bounds, solutions

def _add_solution(solutions):
    '''Add the current variable values to the solution list.'''
    for var in solutions:
        solutions[var].append(pe.value(var))

def _update_values(var, bound_dir, solutions):
    '''
    Set the values of all variables to the best solution seen previously for
    the current objective function.
    '''
    if bound_dir == 'LB':
        value = min(solutions[var])
    else:
        value = max(solutions[var])
    idx = solutions[var].index(value)
    for variable in solutions:
        variable.set_value(solutions[variable][idx])
    