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

from pyomo.core import (
    SortComponents,
    Constraint,
    Objective,
    LogicalConstraint,
    Expression,
)
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
    get_main_elapsed_time,
    move_nonlinear_objective_to_constraints,
)
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components


def _get_discrete_problem_and_subproblem(solver, config):
    util_block = solver.original_util_block
    original_model = util_block.parent_block()
    if config.force_subproblem_nlp:
        # We'll need to fix these too
        add_discrete_variable_list(util_block)
    original_obj = move_nonlinear_objective_to_constraints(util_block, config.logger)
    solver.original_obj = original_obj

    # create model to hold the subproblems: We create this first because
    # certain initialization strategies for the discrete problem need it.
    subproblem, subproblem_util_block = get_subproblem(original_model, util_block)

    # create discrete problem--the MILP relaxation
    start = get_main_elapsed_time(solver.timing)
    discrete_problem_util_block = initialize_discrete_problem(
        util_block, subproblem_util_block, config, solver
    )

    config.logger.info(
        'Finished discrete problem initialization in {:.2f}s '
        'and {} iterations \n'.format(
            get_main_elapsed_time(solver.timing) - start,
            solver.initialization_iteration,
        )
    )

    return (discrete_problem_util_block, subproblem_util_block)


def initialize_discrete_problem(util_block, subprob_util_block, config, solver):
    """
    Calls the specified transformation (by default bigm) on the original
    model and removes nonlinear constraints to create a MILP discrete problem.
    """
    config.logger.info("---Starting GDPopt initialization---")
    # clone the original model
    discrete = util_block.parent_block().clone()
    discrete.name = discrete.name + ": discrete problem"

    discrete_problem_util_block = discrete.component(util_block.local_name)
    discrete_problem_util_block.no_good_cuts = ConstraintList()
    discrete_problem_util_block.no_good_disjunctions = Disjunction(Integers)

    # deactivate nonlinear constraints
    for c in discrete.component_data_objects(
        Constraint, active=True, descend_into=(Block, Disjunct)
    ):
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()

    # Transform to a MILP
    TransformationFactory(config.discrete_problem_transformation).apply_to(discrete)
    add_transformed_boolean_variable_list(discrete_problem_util_block)
    add_algebraic_variable_list(discrete_problem_util_block, name='all_mip_variables')

    # Call the specified initialization strategy. (We've already validated the
    # input in the config logic, so we know this is okay.)
    init_algorithm = valid_init_strategies.get(config.init_algorithm)
    init_algorithm(
        util_block, discrete_problem_util_block, subprob_util_block, config, solver
    )

    return discrete_problem_util_block


def add_util_block(discrete):
    # create a block to store the cuts
    name = unique_component_name(discrete, '_gdpopt_cuts')
    block = Block()
    discrete.add_component(name, block)

    return block


def add_disjunct_list(util_block):
    model = util_block.parent_block()
    util_block.disjunct_list = list(
        model.component_data_objects(
            ctype=Disjunct,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        )
    )


def add_disjunction_list(util_block):
    model = util_block.parent_block()
    util_block.disjunction_list = list(
        model.component_data_objects(
            ctype=Disjunction,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        )
    )


def add_constraint_list(util_block):
    model = util_block.parent_block()
    util_block.constraint_list = list(
        model.component_data_objects(
            ctype=Constraint,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        )
    )


def add_global_constraint_list(util_block):
    model = util_block.parent_block()
    util_block.global_constraint_list = list(
        model.component_data_objects(
            ctype=Constraint,
            active=True,
            descend_into=Block,
            sort=SortComponents.deterministic,
        )
    )


def add_constraints_by_disjunct(util_block):
    constraints_by_disjunct = util_block.constraints_by_disjunct = {}
    for disj in util_block.disjunct_list:
        cons_list = constraints_by_disjunct[disj] = []
        for cons in disj.component_data_objects(
            Constraint,
            active=True,
            descend_into=Block,
            sort=SortComponents.deterministic,
        ):
            cons_list.append(cons)


def add_algebraic_variable_list(util_block, name=None):
    """
    This collects variables from active Constraints and Objectives. It descends
    into Disjuncts, but does not collect any indicator variables that do not
    appear in algebraic constraints pre-transformation.
    """
    model = util_block.parent_block()
    if name is None:
        name = "algebraic_variable_list"
    setattr(
        util_block,
        name,
        list(
            get_vars_from_components(
                model,
                ctype=(Constraint, Objective),
                descend_into=(Block, Disjunct),
                active=True,
                sort=SortComponents.deterministic,
            )
        ),
    )


def add_discrete_variable_list(util_block):
    lst = util_block.discrete_variable_list = []
    for v in util_block.algebraic_variable_list:
        if v.is_integer():
            lst.append(v)


# Must be collected after list of Disjuncts
def add_boolean_variable_lists(util_block):
    util_block.boolean_variable_list = []
    util_block.non_indicator_boolean_variable_list = []
    for disjunct in util_block.disjunct_list:
        util_block.boolean_variable_list.append(disjunct.indicator_var)
    ind_var_set = ComponentSet(util_block.boolean_variable_list)
    # This will not necessarily include the indicator_vars if it is called
    # before the GDP is transformed to a MIP.
    for v in get_vars_from_components(
        util_block.parent_block(),
        ctype=LogicalConstraint,
        descend_into=(Block, Disjunct),
        active=True,
        sort=SortComponents.deterministic,
    ):
        if v not in ind_var_set:
            util_block.boolean_variable_list.append(v)
            util_block.non_indicator_boolean_variable_list.append(v)


# For the discrete problem, we want the corresponding binaries for all of the
# BooleanVars. This must be called after logical_to_disjunctive has been called.
def add_transformed_boolean_variable_list(util_block):
    util_block.transformed_boolean_variable_list = [
        v.get_associated_binary() for v in util_block.boolean_variable_list
    ]


def get_subproblem(original_model, util_block):
    """Clone the original, and reclassify all the Disjuncts to Blocks.
    We'll also call logical_to_disjunctive and bigm the disjunctive parts in
    case any of the indicator_vars are used in logical constraints and to make
    sure that the rest of the model is algebraic (assuming it was a proper
    GDP to begin with).
    """
    subproblem = original_model.clone()
    subproblem.name = subproblem.name + ": subproblem"

    # Set up dual value reporting
    if not hasattr(subproblem, 'dual'):
        subproblem.dual = Suffix(direction=Suffix.IMPORT)
    elif not isinstance(subproblem.dual, Suffix):
        raise ValueError(
            "The model contains a component called 'dual' that "
            "is not a Suffix. It is of type %s. Please rename "
            "this component, as GDPopt needs dual information to "
            "create cuts." % type(subproblem.dual)
        )
    subproblem.dual.activate()

    # reclassify all the Disjuncts as Blocks and deactivate the Disjunctions. We
    # don't need to add the xor constraints because we're not going to pass
    # infeasible integer solutions to this model.
    for disjunction in subproblem.component_data_objects(
        Disjunction,
        descend_into=(Block, Disjunct),
        descent_order=TraversalStrategy.PostfixDFS,
    ):
        for disjunct in disjunction.disjuncts:
            if disjunct.indicator_var.fixed:
                if not disjunct.indicator_var.value:
                    disjunct.deactivate()
            disjunct.parent_block().reclassify_component_type(disjunct, Block)

        disjunction.deactivate()

    TransformationFactory('contrib.logical_to_disjunctive').apply_to(subproblem)
    # transform any of the Disjuncts we created above with bigm.
    TransformationFactory('gdp.bigm').apply_to(subproblem)

    subproblem_util_block = subproblem.component(util_block.local_name)
    save_initial_values(subproblem_util_block)
    add_transformed_boolean_variable_list(subproblem_util_block)
    subproblem_obj = next(
        subproblem.component_data_objects(Objective, active=True, descend_into=True)
    )
    subproblem_util_block.obj = Expression(expr=subproblem_obj.expr)

    return subproblem, subproblem_util_block


def save_initial_values(subproblem_util_block):
    initial_values = subproblem_util_block.initial_var_values = ComponentMap()
    for v in subproblem_util_block.algebraic_variable_list:
        initial_values[v] = v.value
