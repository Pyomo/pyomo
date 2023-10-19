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

from pyomo.common.collections import ComponentMap
from pyomo.gdp.util import clone_without_expression_components
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import aos_utils
    
m = pe.ConcreteModel()

m.x = pe.Var(within=pe.Reals, bounds=(-1,3))
m.y = pe.Var(within=pe.Reals, bounds=(-3,2))

m.obj = pe.Objective(expr=m.x+2*m.y, sense=pe.maximize)

m.con1 = pe.Constraint(expr=m.x+m.y<=3)
m.con2 = pe.Constraint(expr=m.x+2*m.y<=5)

model = m

def _set_slack_ub(expression, slack_var):
    slack_lb, slack_ub = compute_bounds_on_expr(expression)
    assert slack_ub >= 0
    slack_var.setub(slack_ub)

# def get_canonical_lp(model, block):
block = None
if block is None:
    block = model

# Gather all variable and confirm the model is a bounded LP
all_variables = aos_utils.get_model_variables(model, 'all')
var_names = {}
var_name_bounds = {}
var_map = ComponentMap()
for var in all_variables:
    assert var.is_continuous(), ('Variable {} is not continuous. Model must be'
                                 ' a linear program.'.format(var.name))
    assert var.lb is not None , ('Variable {} does not have a lower bound. '
                                 'Variables must be bounded.'.format(var.name))
    assert var.ub is not None , ('Variable {} does not have an upper bound. '
                                 'Variables must be bounded.'.format(var.name))
    var_name = var.name
    #TODO: Need to make names unique
    var_names[id(var)] = var_name
    var_name_bounds[var_name] = (0,var.ub - var.lb)

canon_lp = aos_utils._add_aos_block(block, name='_canon_lp')

# Replace original variables with shifted lower and upper bound "s" variables 
canon_lp.var_index = pe.Set(initialize=var_name_bounds.keys())
canon_lp.var_lower = pe.Var(canon_lp.var_index, domain=pe.NonNegativeReals, 
                            bounds=var_name_bounds)
canon_lp.var_upper = pe.Var(canon_lp.var_index, domain=pe.NonNegativeReals, 
                            bounds=var_name_bounds)

# Link the shifted lower and upper bound "s" variables 
def link_vars_rule(m, var_index):
    return m.var_lower[var_index] + m.var_upper[var_index] == \
        m.var_upper[var_index].ub
canon_lp.link_vars = pe.Constraint(canon_lp.var_index, rule=link_vars_rule)


# Link the original and shifted lower bound variables, and get the original
# lower bound
var_lower_map = {}
var_lower_bounds = {}
for var in all_variables:
    var_lower_map[id(var)] = canon_lp.var_lower[var_names[id(var)]]
    var_lower_bounds[id(var)] = var.lb

# Substitute the new s variables into the objective function
active_objective = aos_utils._get_active_objective(model)
c_var_lower = clone_without_expression_components(active_objective.expr, 
                                                  substitute=var_lower_map)
c_fix_lower = clone_without_expression_components(active_objective.expr, 
                                                  substitute=var_lower_bounds)
canon_lp.objective = pe.Objective(expr=c_var_lower + c_fix_lower,
                                  name=active_objective.name + '_shifted',
                                  sense=active_objective.sense)

# Identify all of the shifted constraints and associated slack variables 
# that will need to be created
new_constraints = {}
slacks = []
for constraint in model.component_data_objects(pe.Constraint, active=True):
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
        _set_slack_ub(expression_rhs, canon_lp.slack_vars[constraint_name])
    elif constraint_type == 1:
        expression_rhs = b_upper - a_sub_var_lower - a_sub_fix_lower
        expression = canon_lp.slack_vars[constraint_name] == expression_rhs
        _set_slack_ub(expression_rhs, canon_lp.slack_vars[constraint_name])
    canon_lp.constraints[constraint_name] = expression