# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:18:04 2022

@author: jlgearh
"""

import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.gdp.util import clone_without_expression_components
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import aos_utils
    
# TODO set the variable values at the end

model = pe.ConcreteModel()

model.x = pe.Var(within=pe.PercentFraction)
model.y = pe.Var(within=pe.PercentFraction)


model.obj = pe.Objective(expr=model.x+model.y, sense=pe.maximize)

model.wx_limit = pe.Constraint(expr=model.x+model.y<=2)

# model = pe.ConcreteModel()

# model.w = pe.Var(within=pe.NonNegativeReals)
# model.x = pe.Var(within=pe.Reals)
# model.y = pe.Var(within=pe.PercentFraction)
# model.z = pe.Var(within=pe.Reals, bounds=(0,1))


# model.obj = pe.Objective(expr=model.w+model.x+model.y+model.z, sense=pe.maximize)

# model.wx_limit = pe.Constraint(expr=model.w+model.x<=2)
# model.wu_limit = pe.Constraint(expr=model.w<=1)
# model.xl_limit = pe.Constraint(expr=model.x>=0)
# model.xu_limit = pe.Constraint(expr=model.x<=1)

# model.b = pe.Block()
# model.b.yz_limit = pe.Constraint(expr=-model.y-model.z>=-2)
# model.b.wy = pe.Constraint(expr=model.w+model.y==1)

# model = pe.ConcreteModel()

# model.w = pe.Var(within=pe.PercentFraction)
# model.x = pe.Var(within=pe.PercentFraction)
# model.y = pe.Var(within=pe.PercentFraction)
# model.z = pe.Var(within=pe.PercentFraction)


# model.obj = pe.Objective(expr=model.w+model.x+model.y+model.z, sense=pe.maximize)

# model.wx_limit = pe.Constraint(expr=model.w+model.x<=2)


# model.b = pe.Block()
# model.b.yz_limit = pe.Constraint(expr=-model.y-model.z>=-2)
# model.b.wy = pe.Constraint(expr=model.w+model.y==1)

# Get a Pyomo concrete model


# Get all continuous variables in the model and check that they have finite
# bounds
# TODO handle fixed variables
model_vars = aos_utils.get_model_variables(model, 'all')
model_var_names = {}
model_var_names_bounds = {}
for mv in model_vars:
    assert mv.is_continuous, 'Variable {} is not continuous'.format(mv.name)
    assert not (mv.lb is None and mv.ub is None)
    var_name = mv.name
    model_var_names[id(mv)] = var_name
    model_var_names_bounds[var_name] = (0,mv.ub - mv.lb)

canon_lp = aos_utils._add_aos_block(model, name='canon_lp')

# Replace original variables with shifted lower and upper bound "s" variables 
# TODO use unique names

canon_lp.var_index = pe.Set(initialize=model_var_names_bounds.keys())

canon_lp.var_lower = pe.Var(canon_lp.var_index, domain=pe.NonNegativeReals, 
                            bounds=model_var_names_bounds)
canon_lp.var_upper = pe.Var(canon_lp.var_index, domain=pe.NonNegativeReals, 
                            bounds=model_var_names_bounds)

def link_vars_rule(model, var_index):
    return model.var_lower[var_index] + model.var_upper[var_index] == \
        model.var_upper[var_index].ub
canon_lp.link_vars = pe.Constraint(canon_lp.var_index, rule=link_vars_rule)

var_lower_map = {}
var_lower_bounds = {}
for mv in model_vars:
    var_lower_map[id(mv)] = canon_lp.var_lower[model_var_names[id(mv)]]
    var_lower_bounds[id(mv)] = mv.lb

# Substitue the new s variables into the objective function
orig_objective = aos_utils._get_active_objective(model)
c_var_lower = clone_without_expression_components(orig_objective.expr, 
                                                  substitute=var_lower_map)
c_fix_lower = clone_without_expression_components(orig_objective.expr, 
                                                  substitute=var_lower_bounds)
canon_lp.objective = pe.Objective(expr=c_var_lower + c_fix_lower,
                                  name=orig_objective.name + '_shifted',
                                  sense=orig_objective.sense)

new_constraints = {}
slacks = []
for constraint in model.component_data_objects(pe.Constraint, active=None,
                                               sort=False, 
                                               descend_into=pe.Block,
                                               descent_order=None):
    if constraint.parent_block() == canon_lp:
        continue
    if constraint.equality:
        constraint_name = constraint.name + '_equal'
        new_constraints[constraint_name] = (constraint,0)
    else:
        if constraint.lb is not None:
            constraint_name = constraint.name + '_lower'
            new_constraints[constraint_name] = (constraint,-1)
            slacks.append(constraint_name)
        if constraint.ub is not None:
            constraint_name = constraint.name + '_upper'
            new_constraints[constraint_name] = (constraint,1)
            slacks.append(constraint_name)
canon_lp.constraint_index = pe.Set(initialize=new_constraints.keys())
canon_lp.slack_index = pe.Set(initialize=slacks)
canon_lp.slack_vars = pe.Var(canon_lp.slack_index, domain=pe.NonNegativeReals)
canon_lp.constraints = pe.Constraint(canon_lp.constraint_index)

constraint_map = {}
constraint_bounds = {}

def set_slack_ub(expression, slack_var):
    slack_lb, slack_ub = compute_bounds_on_expr(expression)
    assert slack_lb == 0 and slack_ub >= 0
    slack_var.setub(slack_ub)
                    
for constraint_name, (constraint, constraint_type) in new_constraints.items():
    
    a_sub_var_lower = clone_without_expression_components(constraint.body, 
                                                  substitute=var_lower_map)
    a_sub_fix_lower = clone_without_expression_components(constraint.body, 
                                                  substitute=var_lower_bounds)
    b_lower = constraint.lb
    b_upper = constraint.ub
    if constraint_type == 0:
        expression = a_sub_var_lower + a_sub_fix_lower - b_lower == 0     
    elif constraint_type == -1:
        expression_rhs = a_sub_var_lower + a_sub_fix_lower - b_lower
        expression = canon_lp.slack_vars[constraint_name] == expression_rhs
        set_slack_ub(expression_rhs, canon_lp.slack_vars[constraint_name])
    elif constraint_type == 1:
        expression_rhs = b_upper - a_sub_var_lower - a_sub_fix_lower
        expression = canon_lp.slack_vars[constraint_name] == expression_rhs
        set_slack_ub(expression_rhs, canon_lp.slack_vars[constraint_name])
    canon_lp.constraints[constraint_name] = expression


def enumerate_linear_solutions(model, max_solutions=10, variables='all', 
                                rel_opt_gap=None, abs_gap=None,
                                search_mode='optimal', already_solved=False,
                                solver='cplex', solver_options={}, tee=False):
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


    # Find the maximum number of solutions to generate
    num_solutions = aos_utils._get_max_solutions(max_solutions)
    opt = aos_utils._get_solver(solver, solver_options)

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