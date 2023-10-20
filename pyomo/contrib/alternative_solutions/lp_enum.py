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
from pyomo.contrib.alternative_solutions import aos_utils, shifted_lp, solution
    
def enumerate_linear_solutions(model, num_solutions=10, variables='all', 
                               rel_opt_gap=None, abs_opt_gap=None,
                               search_mode='optimal', solver='cplex', 
                               solver_options={}, tee=False):
    '''
    Finds alternative optimal solutions for a binary problem.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model
        num_solutions : int
            The maximum number of solutions to generate.
        variables: 'all' or a collection of Pyomo _GeneralVarData variables
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
        search_mode : 'optimal', 'random', or 'norm'
            Indicates the mode that is used to generate alternative solutions.
            The optimal mode finds the next best solution. The random mode
            finds an alternative solution in the direction of a random ray. The
            norm mode iteratively finds solution that maximize the L2 distance 
            from previously discovered solutions.
        solver : string
            The solver to be used.
        solver_options : dict
            Solver option-value pairs to be passed to the solver.
        tee : boolean
            Boolean indicating that the solver output should be displayed.
            
        Returns
        -------
        solutions
            A list of Solution objects.
            [Solution]
    '''
    # TODO: Set this intelligently
    zero_threshold = 1e-5
    print('STARTING LP ENUMERATION ANALYSIS')
    
    # For now keeping things simple
    # TODO: Relax this
    assert variables == 'all'
    
    assert search_mode in ['optimal', 'random', 'norm'], \
        'search mode must be "optimal", "random", or "norm".'
        
    if variables == 'all':
        all_variables = aos_utils.get_model_variables(model, 'all')
    # else:
    #     binary_variables = ComponentSet()
    #     non_binary_variables = []
    #     for var in variables:
    #         if var.is_binary():
    #             binary_variables.append(var)
    #         else:
    #             non_binary_variables.append(var.name)
    #     if len(non_binary_variables) > 0:
    #         print(('Warning: The following non-binary variables were included'
    #                'in the variable list and will be ignored:'))
    #         print(", ".join(non_binary_variables))
    # all_variables = aos_utils.get_model_variables(model, 'all', 
    #                                               include_fixed=True)
    
    # TODO: Relax this if possible
    for var in all_variables:
        assert var.is_continuous(), 'Model must be an LP'

    opt = pe.SolverFactory(solver)
    for parameter, value in solver_options.items():
        opt.options[parameter] = value
        
    use_appsi = False
    # TODO Check all this once implemented
    if 'appsi' in solver:
        use_appsi = True
        opt.update_config.check_for_new_or_removed_constraints = True
        opt.update_config.update_constraints = False
        opt.update_config.check_for_new_or_removed_vars = True
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.update_vars = False
        opt.update_config.update_params = False
        opt.update_config.update_named_expressions = False
        opt.update_config.treat_fixed_vars_as_params = False
        
        if search_mode == 'norm':
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = True
        elif search_mode == 'random':
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = False   
        else:
            opt.update_config.check_for_new_objective = False
            opt.update_config.update_objective = False
        
    print('Peforming initial solve of model.')
    results = opt.solve(model, tee=tee)
    status = results.solver.status
    condition = results.solver.termination_condition
    if condition != pe.TerminationCondition.optimal:
        raise Exception(('Model could not be solve. LP enumeration analysis '
                         'cannot be applied, SolverStatus = {}, '
                         'TerminationCondition = {}').format(status.value, 
                                                             condition.value))
    
    orig_objective = aos_utils._get_active_objective(model)
    orig_objective_value = pe.value(orig_objective)
    print('Found optimal solution, value = {}.'.format(orig_objective_value))
    
    aos_block = aos_utils._add_aos_block(model, name='_lp_enum')
    print('Added block {} to the model.'.format(aos_block))
    aos_utils._add_objective_constraint(aos_block, orig_objective, 
                                        orig_objective_value, rel_opt_gap, 
                                        abs_opt_gap)   
    
    canon_block = shifted_lp.get_shifted_linear_model(model)
    cb = canon_block
        
    # Set K
    cb.iteration = pe.Set(pe.PositiveIntegers)

    # w variables
    cb.basic_lower = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    cb.basic_upper = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    cb.basic_slack = pe.Var(pe.Any, domain=pe.Binary, dense=False)
  
    # w upper bounds constraints
    cb.bound_lower = pe.Constraint(pe.Any)
    cb.bound_upper = pe.Constraint(pe.Any)
    cb.bound_slack = pe.Constraint(pe.Any)
    
    # non-zero basic variable no-good cut set
    cb.cut_set = pe.Constraint(pe.PositiveIntegers)
    
    variable_groups = [(cb.var_lower, cb.basic_lower, cb.bound_lower),
                       (cb.var_upper, cb.basic_upper, cb.bound_upper),
                       (cb.slack_vars, cb.basic_slack, cb.bound_slack)]

    solution_number = 1
    solutions = []    
    while solution_number <= num_solutions:
        print('Solving Iteration {}: '.format(solution_number), end='')
        results = opt.solve(cb, tee=tee)
        status = results.solver.status
        condition = results.solver.termination_condition
        if condition == pe.TerminationCondition.optimal:
            for var, index in cb.var_map.items():
                var.set_value(var.lb + cb.var_lower[index].value)
            sol = solution.Solution(model, all_variables, 
                                     objective=orig_objective)
            solutions.append(sol)
            orig_objective_value = sol.objective[1]
            print('Solved, objective = {}'.format(orig_objective_value))
            for var, index in cb.var_map.items():
                print('{} = {}'.format(var.name, var.lb + cb.var_lower[index].value))
            if hasattr(cb, 'force_out'):
                cb.del_component('force_out')
            if hasattr(cb, 'link_in_out'):
                cb.del_component('link_in_out')

            if hasattr(cb, 'basic_last_lower'):
                cb.del_component('basic_last_lower')       
            if hasattr(cb, 'basic_last_upper'):
                cb.del_component('basic_last_upper')
            if hasattr(cb, 'basic_last_slack'):
                cb.del_component('basic_last_slack')
                
            cb.link_in_out = pe.Constraint(pe.Any)
            cb.basic_last_lower = pe.Var(pe.Any, domain=pe.Binary, dense=False)
            cb.basic_last_upper = pe.Var(pe.Any, domain=pe.Binary, dense=False)
            cb.basic_last_slack = pe.Var(pe.Any, domain=pe.Binary, dense=False)
            basic_last_list = [cb.basic_last_lower, cb.basic_last_upper, 
                               cb.basic_last_slack]
            
            num_non_zero = 0
            force_out_expr = -1
            non_zero_basic_expr = 1
            for idx in range(len(variable_groups)):
                continuous_var, binary_var, constraint = variable_groups[idx]
                for var in continuous_var:
                    if continuous_var[var].value > zero_threshold:
                        num_non_zero += 1
                        if var not in binary_var:
                            binary_var[var]
                            constraint[var] = continuous_var[var] <= \
                                continuous_var[var].ub * binary_var[var]
                        non_zero_basic_expr += binary_var[var]
                        basic_var = basic_last_list[idx][var]
                        force_out_expr += basic_var
                        cb.link_in_out[var] = basic_var + binary_var[var] <= 1
            cb.cut_set[solution_number] = non_zero_basic_expr <= num_non_zero
            
            solution_number += 1
        elif (condition == pe.TerminationCondition.infeasibleOrUnbounded or 
              condition == pe.TerminationCondition.infeasible):
            print("Infeasible, all alternative solutions have been found.")
            break
        else:
            print(("Unexpected solver condition. Stopping LP enumeration. "
                   "SolverStatus = {}, TerminationCondition = {}").format(
                       status.value, condition.value))
            break

    aos_block.deactivate()
    print('COMPLETED LP ENUMERATION ANALYSIS')
    
    return solutions