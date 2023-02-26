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

#import pandas as pd

import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.collections import ComponentMap

import pyomo.contrib.alternative_solutions.aos_utils as aos_utils
import pyomo.contrib.alternative_solutions.variables as var_utils

def obbt_analysis(model, variables='all', rel_opt_gap=None, abs_gap=None, 
                  refine_bounds=False, warmstart=False, already_solved=False, 
                  solver='gurobi', solver_options={}, 
                  use_persistent_solver=False, tee=False):
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
        abs_gap : float or None
            The absolute optimality gap for the original objective for which 
            variable bounds will be found. None indicates that an absolute gap 
            constraint will not be added to the model.
        refine_bounds : boolean
            Boolean indicating that new constraints should be added to the 
            model at each iteration to tighten the bounds for varaibles.
        warmstart : boolean
            Boolean indicating that previous solutions should be passed to the
            solver as warmstart solutions.
        already_solved : boolean
            Indicates that the model has already been solved and that the 
            variable bound search can start from the current solution.
        solver : string
            The solver to be used.
        solver_options : dict
            Solver option-value pairs to be passed to the solver.
        use_persistent_solver : boolean
            Boolean indicating if the the APPSI persistent solver interface
            should be used. Currently, only supported Gurobi is supported for
            variable bound analysis with the persistent solver.
        tee : boolean
            Boolean indicating that the solver output should be displayed.
            
        Returns
        -------
        variable_ranges
            A Pyomo ComponentMap containing the bounds for each variable.
            {variable: (lower_bound, upper_bound)}
    '''

    aos_utils._is_concrete_model(model)
    assert isinstance(refine_bounds, bool), 'refine_bounds must be a Boolean'
    assert isinstance(warmstart, bool), 'warmstart must be a Boolean'
    assert isinstance(already_solved, bool), 'already_solved must be a Boolean'
    assert isinstance(use_persistent_solver, bool), \
        'use_persistent_solver must be a Boolean'
    assert isinstance(tee, bool), 'tee must be a Boolean'

    if variables == 'all':
        variable_list = var_utils.get_model_variables(model, variables,
                                                      include_fixed=False)
    else:
        variable_list = var_utils.check_variables(model, variables)
    
    orig_objective = aos_utils._get_active_objective(model)
    aos_block = aos_utils._add_aos_block(model)
    new_constraint = False

    opt = aos_utils._get_solver(solver, solver_options, use_persistent_solver)

    if not already_solved:
        results = opt.solve(model)#, tee=tee)
        status = results.solver.status
        condition = results.solver.termination_condition
        assert (status == SolverStatus.ok and 
            condition == TerminationCondition.optimal), \
            ('Model cannot be solved, SolverStatus = {}, '
             'TerminationCondition = {}').format(status.value, 
                                                 condition.value)
        
    orig_objective_value = pe.value(orig_objective)
    aos_utils._add_objective_constraint(aos_block, orig_objective, 
                                        orig_objective_value, rel_opt_gap, 
                                        abs_gap)    
    if rel_opt_gap is not None or abs_gap is not None:
        new_constraint = True
    
    orig_objective.deactivate()
    
    if use_persistent_solver:
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
        
    variable_bounds = ComponentMap()
 
    senses = [pe.minimize, pe.maximize]
    
    iteration = 1
    total_iterations = len(senses) *  len(variable_list)
    for idx in range(2):
        sense = senses[idx]
        sense_name = 'min'
        bound_dir = 'LB'
        if sense == pe.maximize:
            sense_name = 'max'
            bound_dir = 'UB'
            
        for var in variable_list:
            if idx == 0:
                variable_bounds[var] = [None, None]
        
            if hasattr(aos_block, 'var_objective'):
                aos_block.del_component('var_objective')
            
            aos_block.var_objective = pe.Objective(expr=var, sense=sense)  
            
            # TODO: Updated solution pool
            
            if use_persistent_solver:
                opt.update_config.check_for_new_or_removed_constraints = \
                    new_constraint
            results = opt.solve(model)#, tee=tee)
            new_constraint = False
            status = results.solver.status
            condition = results.solver.termination_condition
            if (status == SolverStatus.ok and 
                    condition == TerminationCondition.optimal):
                obj_val = pe.value(var)
                variable_bounds[var][idx] = obj_val

                if refine_bounds and sense == pe.minimize and var.lb < obj_val:
                    bound_name = var.name + '_lb'
                    bound = pe.Constraint(expr= var >= obj_val)
                    setattr(aos_block, bound_name, bound)
                    new_constraint = True
                    
                if refine_bounds and sense == pe.maximize and var.ub > obj_val:
                    bound_name = var.name + '_ub'
                    bound = pe.Constraint(expr= var <= obj_val)
                    setattr(aos_block, bound_name, bound)
                    new_constraint = True
            # An infeasibleOrUnbounded status code will imply the problem is
            # unbounded since feasibility has be established previously
            elif (status == SolverStatus.ok and (
                    condition == TerminationCondition.infeasibleOrUnbounded or
                    condition == TerminationCondition.unbounded)):
                if sense == pe.minimize:
                    variable_bounds[var][idx] = float('-inf')
                else:
                    variable_bounds[var][idx] = float('inf')
            else:
                print(('Unexpected solver status for variable {} {} problem.'
                       'SolverStatus = {}, TerminationCondition = {}').\
                      format(var.name, sense_name, status.value, 
                             condition.value))
            
            
            print('It. {}/{}: {}_{} = {}'.format(iteration, total_iterations,
                                                var.name, bound_dir, 
                                                variable_bounds[var][idx]))
            
            if idx == 1:
                variable_bounds[var] = tuple(variable_bounds[var])
                
            iteration += 1

        
    aos_block.deactivate
    orig_objective.active
    
    return variable_bounds

# def get_var_bound_dataframe(variable_bounds):
#     '''Get a pandas DataFrame displaying the variable bound results.'''
#     return pd.DataFrame.from_dict(variable_bounds,orient='index', 
#                                   columns=['LB','UB'])
