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
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.gdp.util import clone_without_expression_components
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import aos_utils
    
def enumerate_linear_solutions(model, num_solutions=10, variables='all', 
                               rel_opt_gap=None, abs_gap=None,
                               search_mode='optimal', solver='cplex', 
                               solver_options={}, tee=False):
    '''Finds alternative optimal solutions for a binary problem.

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
    print('STARTING LP ENUMERATION ANALYSIS')
    
    # For now keeping things simple
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
    
    for var in all_variables:
        assert var.is_continuous(), 'Model must be an LP'
    
    orig_objective = aos_utils._get_active_objective(model)
    
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
        raise Exception(('LP enumeration analysis cannot be applied, '
                         'SolverStatus = {}, '
                         'TerminationCondition = {}').format(status.value, 
                                                             condition.value))
    
    orig_objective_value = pe.value(orig_objective)
    print('Found optimal solution, value = {}.'.format(orig_objective_value))
    
    aos_block = aos_utils._add_aos_block(model, name='_lp_enum')
    print('Added block {} to the model.'.format(aos_block))
    aos_utils._add_objective_constraint(aos_block, orig_objective, 
                                        orig_objective_value, rel_opt_gap, 
                                        abs_opt_gap)   
    
    canon_block = get_canonical_lp(model)    
    
   
    solution_number = 2
    
    orig_objective.deactivate()
    solutions = [solution.Solution(model, all_variables)]
    
    while solution_number <= num_solutions:
    model.iteration = pe.Set(dimen=1)

    model.basic_lower = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    model.basic_upper = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    model.basic_slack = pe.Var(pe.Any, domain=pe.Binary, dense=False)
  
    model.bound_lower = pe.Constraint(pe.Any)
    model.bound_upper = pe.Constraint(pe.Any)
    model.bound_slack = pe.Constraint(pe.Any)
    model.cut_set = pe.Constraint(pe.Any)
    
    variable_groups = [(model.var_lower, model.basic_lower, model.bound_lower),
                       (model.var_upper, model.basic_upper, model.bound_upper),
                       (model.slack_vars, model.basic_slack, model.bound_slack)]
    
    # Repeat until all solutions are found
    solution_number = 1
    solutions = {}
    while solution_number < num_solutions:
    
        # Solve the model unless this is the first solution and the model was 
        # not already solved
        if solution_number > 1 or not already_solved:
            print('Iteration: {}'.format(solution_number))
            results = opt.solve(model, tee=tee)
    
        if (((results.solver.status == SolverStatus.ok) and
        (results.solver.termination_condition == TerminationCondition.optimal))
        or (already_solved and solution_number == 0)):
            #objective_value = pe.value(orig_objective)

            for variable in model.var_lower:
                print('Var {} = {}'.format(variable, 
                                         pe.value(model.var_lower[variable])))

            expr = 1
            num_non_zeros = 0
            
            for continuous_var, binary_var, constraint in variable_groups:
                for variable in continuous_var:
                    if pe.value(continuous_var[variable]) > 1e-5:
                        if variable not in binary_var:
                            model.basic_upper[variable]
                        constraint[variable] = continuous_var[variable] <= \
                            continuous_var[variable].ub * binary_var[variable]
                        expr += binary_var[variable]
                        num_non_zeros += 1
            model.cut_set[solution_number] = expr <= num_non_zeros
            solution_number += 1
            
        else:
            print('Algorithm Stopped. Solver Status: {}. Solver Condition: {}.'\
                  .format(results.solver.status,
                          results.solver.termination_condition))
            break