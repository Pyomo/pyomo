#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.core import SortComponents, Constraint, Objective
from pyomo.core.base import TransformationFactory, Suffix
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentMap
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.contrib.gdpopt.master_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import move_nonlinear_objective_to_constraints

def initialize_master_problem(util_block, subprob_util_block, config, solver):
    """
    Calls the specified transformation (by default bigm) on the original
    model and removes nonlinear constraints to create a MILP master problem.
    """
    config.logger.info("---Starting GDPopt initialization---")
    # clone the original model
    master = util_block.model().clone()

    # TODO: switch to getname and set up a name buffer
    master_util_block = master.component(util_block.name)
    # TODO: Should we do this for subproblem too?? I think I have this wrong.
    move_nonlinear_objective_to_constraints(master_util_block, config.logger)

    # deactivate nonlinear constraints
    for c in master.component_data_objects(Constraint, active=True,
                                           descend_into=(Block, Disjunct)):
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()

    # Transform to a MILP
    TransformationFactory(config.master_problem_transformation).apply_to(master)

    # Call the specified initialization strategy
    init_strategy = valid_init_strategies.get(config.init_strategy, None)
    if init_strategy is not None:
        init_strategy(util_block, master_util_block, subprob_util_block, config,
                      solver)
    else:
        raise ValueError('Unknown initialization strategy: %s. '
                         'Valid strategies include: %s'
                         % (config.init_strategy,
                            ", ".join(k for (k, v) in 
                                      valid_init_strategies.items()
                                      if v is not None)))

    return master_util_block

def add_util_block(master):
    # create a block to store the cuts
    name = unique_component_name(master, '_gdpopt_cuts')
    block = Block()
    master.add_component(name, block)

    return block

def add_disjunct_list(util_block):
    model = util_block.model()
    util_block.disjunct_list = list(model.component_data_objects(
        ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
        sort=SortComponents.deterministic))

def add_constraint_list(util_block):
    model = util_block.model()
    util_block.constraint_list = list(model.component_data_objects(
        ctype=Constraint, active=True, descend_into=(Block, Disjunct),
        sort=SortComponents.deterministic))

def add_variable_list(util_block):
    model = util_block.model()
    util_block.variable_list = list(get_vars_from_components(
        model, ctype=(Constraint, Objective), descend_into=(Block, Disjunct),
        active=True, sort=SortComponents.deterministic))

def get_subproblem(original_model):
    """Clone the original, and reclassify all the Disjuncts to Blocks.
    We'll also call logical_to_linear in case any of the indicator_vars are
    used in logical constraints and to make sure that the rest of the model is
    algebraic (assuming it was a proper GDP to begin with).
    """
    ## Debatably this could be a transformation in gdp...

    subproblem = original_model.clone()

    # Set up dual value reporting
    if not hasattr(subproblem, 'dual'):
        subproblem.dual = Suffix(direction=Suffix.IMPORT)
    elif not isinstance(subproblem.dual, Suffix):
        raise ValueError("The model containts a component called 'dual' which "
                         "is not a Suffix. It is of type %s. Please rename "
                         "this component, as GDPopt needs dual information to "
                         "create cuts." % type(subproblem.dual))
    subproblem.dual.activate()

    # reclassify all the Disjuncts as Blocks and deactivate the Disjunctions. We
    # don't need to add the xor constraints because we're not going to pass
    # infeasible integer solutions to this model.  
    # ESJ TODO: Perhaps this should
    # rely on pre-constructed ordered lists, but I suspect if can't because it
    # should store some original states?...
    for disjunction in subproblem.component_data_objects(
            Disjunction, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.PostfixDFS):
        for disjunct in disjunction.disjuncts:
            if disjunct.indicator_var.fixed:
                if not disjunct.indicator_var.value:
                    disjunct.deactivate()
            disjunct.parent_block().reclassify_component_type(disjunct, Block)

        disjunction.deactivate()

    TransformationFactory('core.logical_to_linear').apply_to(subproblem)

    return subproblem

def save_initial_values(subproblem_util_block):
    initial_values = subproblem_util_block.initial_var_values = ComponentMap()
    for v in subproblem_util_block.variable_list:
        initial_values[v] = v.value
