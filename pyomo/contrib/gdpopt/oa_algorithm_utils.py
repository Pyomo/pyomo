#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.gdpopt.initialize_subproblems import (
    initialize_master_problem, get_subproblem, add_util_block, 
    add_disjunct_list, add_algebraic_variable_list, add_discrete_variable_list, 
    add_boolean_variable_lists, add_constraint_list, save_initial_values, 
    add_transformed_boolean_variable_list)
from pyomo.contrib.gdpopt.util import (move_nonlinear_objective_to_constraints)

from pyomo.core import Objective, Expression

def _get_master_and_subproblem(original_model, config, solver,
                               constraint_list=True):
    # Make a block where we will store some component lists so that after we
    # clone we know who's who
    util_block = solver.original_util_block = add_util_block(original_model)
    # Needed for finding indicator_vars mainly
    add_disjunct_list(util_block)
    add_boolean_variable_lists(util_block)
    # To transfer solutions between MILP and NLP
    add_algebraic_variable_list(util_block)
    # We'll need these to get dual info after solving subproblems
    if constraint_list:
        add_constraint_list(util_block)
    if config.force_subproblem_nlp:
        # We'll need to fix these too
        add_discrete_variable_list(util_block)
    move_nonlinear_objective_to_constraints(util_block, config.logger)

    # create model to hold the subproblems: We create this first because
    # certain initialization strategies for the master problem need it.
    subproblem = get_subproblem(original_model)
    # TODO: use getname and a bufffer!
    subproblem_util_block = subproblem.component(util_block.name)
    save_initial_values(subproblem_util_block)
    add_transformed_boolean_variable_list(subproblem_util_block)
    # TODO, not completely sure if this is what I should do
    subproblem_obj = next(subproblem.component_data_objects(
        Objective, active=True, descend_into=True))
    subproblem_util_block.obj = Expression(expr=subproblem_obj.expr)

    # create master MILP
    master_util_block = initialize_master_problem(util_block,
                                                  subproblem_util_block,
                                                  config, solver)

    return (master_util_block, subproblem_util_block)
