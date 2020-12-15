#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""This module contains functions to interrogate the size of a Pyomo model."""
import logging

from pyomo.common.collections import ComponentSet, Container
from pyomo.core import Block, Constraint, Var
from pyomo.core.expr import current as EXPR
from pyomo.gdp import Disjunct, Disjunction


default_logger = logging.getLogger('pyomo.util.model_size')
default_logger.setLevel(logging.INFO)


class ModelSizeReport(Container):
    """Stores model size information.

    Activated blocks are those who have an active flag of True and whose
    parent, if exists, is an activated block or an activated Disjunct.

    Activated constraints are those with an active flag of True and: are
    reachable via an activated Block, are on an activated Disjunct, or are on a
    disjunct with indicator_var fixed to 1 with active flag True.

    Activated variables refer to the presence of the variable on an activated
    constraint, or that the variable is an indicator_var for an activated
    Disjunct.

    Activated disjuncts refer to disjuncts with an active flag of True, have an
    unfixed indicator_var, and who participate in an activated Disjunction.

    Activated disjunctions follow the same rules as activated constraints.

    """
    pass


def build_model_size_report(model):
    """Build a model size report object."""
    report = ModelSizeReport()
    activated_disjunctions = ComponentSet()
    activated_disjuncts = ComponentSet()
    fixed_true_disjuncts = ComponentSet()
    activated_constraints = ComponentSet()
    activated_vars = ComponentSet()
    new_containers = (model,)

    while new_containers:
        new_activated_disjunctions = ComponentSet()
        new_activated_disjuncts = ComponentSet()
        new_fixed_true_disjuncts = ComponentSet()
        new_activated_constraints = ComponentSet()

        for container in new_containers:
            (next_activated_disjunctions,
             next_fixed_true_disjuncts,
             next_activated_disjuncts,
             next_activated_constraints
             ) = _process_activated_container(container)
            new_activated_disjunctions.update(next_activated_disjunctions)
            new_activated_disjuncts.update(next_activated_disjuncts)
            new_fixed_true_disjuncts.update(next_fixed_true_disjuncts)
            new_activated_constraints.update(next_activated_constraints)

        new_containers = ((new_activated_disjuncts - activated_disjuncts) |
                          (new_fixed_true_disjuncts - fixed_true_disjuncts))

        activated_disjunctions.update(new_activated_disjunctions)
        activated_disjuncts.update(new_activated_disjuncts)
        fixed_true_disjuncts.update(new_fixed_true_disjuncts)
        activated_constraints.update(new_activated_constraints)

    activated_vars.update(
        var for constr in activated_constraints
        for var in EXPR.identify_variables(
            constr.body, include_fixed=False))
    activated_vars.update(
        disj.indicator_var for disj in activated_disjuncts)

    report.activated = Container()
    report.activated.variables = len(activated_vars)
    report.activated.binary_variables = sum(
        1 for v in activated_vars if v.is_binary())
    report.activated.integer_variables = sum(
        1 for v in activated_vars if v.is_integer() and not v.is_binary())
    report.activated.continuous_variables = sum(
        1 for v in activated_vars if v.is_continuous())
    report.activated.disjunctions = len(activated_disjunctions)
    report.activated.disjuncts = len(activated_disjuncts)
    report.activated.constraints = len(activated_constraints)
    report.activated.nonlinear_constraints = sum(
        1 for c in activated_constraints
        if c.body.polynomial_degree() not in (1, 0))

    report.overall = Container()
    block_like = (Block, Disjunct)
    all_vars = ComponentSet(
        model.component_data_objects(Var, descend_into=block_like))
    report.overall.variables = len(all_vars)
    report.overall.binary_variables = sum(1 for v in all_vars if v.is_binary())
    report.overall.integer_variables = sum(
        1 for v in all_vars if v.is_integer() and not v.is_binary())
    report.overall.continuous_variables = sum(
        1 for v in all_vars if v.is_continuous())
    report.overall.disjunctions = sum(
        1 for d in model.component_data_objects(
            Disjunction, descend_into=block_like))
    report.overall.disjuncts = sum(
        1 for d in model.component_data_objects(
            Disjunct, descend_into=block_like))
    report.overall.constraints = sum(
        1 for c in model.component_data_objects(
            Constraint, descend_into=block_like))
    report.overall.nonlinear_constraints = sum(
        1 for c in model.component_data_objects(
            Constraint, descend_into=block_like)
        if c.body.polynomial_degree() not in (1, 0))

    report.warning = Container()
    report.warning.unassociated_disjuncts = sum(
        1 for d in model.component_data_objects(
            Disjunct, descend_into=block_like)
        if not d.indicator_var.fixed and d not in activated_disjuncts)

    return report


def log_model_size_report(model, logger=default_logger):
    """Generate a report logging the model size."""
    logger.info(build_model_size_report(model))


def _process_activated_container(blk):
    """Process a container object, returning the new components found."""
    new_fixed_true_disjuncts = ComponentSet(
        disj for disj in blk.component_data_objects(Disjunct, active=True)
        if disj.indicator_var.value == 1 and disj.indicator_var.fixed)
    new_activated_disjunctions = ComponentSet(
        blk.component_data_objects(Disjunction, active=True))
    new_activated_disjuncts = ComponentSet(
        disj for disjtn in new_activated_disjunctions
        for disj in _activated_disjuncts_in_disjunction(disjtn))
    new_activated_constraints = ComponentSet(
        blk.component_data_objects(Constraint, active=True))
    return (
        new_activated_disjunctions,
        new_fixed_true_disjuncts,
        new_activated_disjuncts,
        new_activated_constraints
    )


def _activated_disjuncts_in_disjunction(disjtn):
    """Retrieve generator of activated disjuncts on disjunction."""
    return (disj for disj in disjtn.disjuncts
            if disj.active and not disj.indicator_var.fixed)
