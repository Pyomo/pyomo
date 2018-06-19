"""This module contains functions to interrogate the size of a Pyomo model."""
import logging

from pyomo.core import Block, Constraint, Var
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyutilib.misc import Container


default_logger = logging.getLogger('pyomo.util.model_size')
default_logger.setLevel(logging.INFO)


class ModelSizeReport(Container):
    """Stores model size information.

    Active blocks are those who have an active flag of True and whose parent,
    if exists, is an active block or an active Disjunct.

    Active constraints are those with an active flag of True and: are reachable
    via an active Block, are on an active Disjunct, or are on a disjunct with
    indicator_var fixed to 1 with active flag True.

    Active variables refer to the presence of the variable on an active
    constraint, or that the variable is an indicator_var for an active
    Disjunct.

    Active disjuncts refer to disjuncts with an active flag of True, have an
    unfixed indicator_var, and who participate in an active Disjunction.

    Active disjunctions follow the same rules as active constraints.

    """
    pass


def build_model_size_report(model):
    """Build a model size report object."""
    report = ModelSizeReport()
    active_disjunctions = ComponentSet()
    active_disjuncts = ComponentSet()
    fixed_true_disjuncts = ComponentSet()
    active_constraints = ComponentSet()
    active_vars = ComponentSet()
    new_containers = (model,)

    while new_containers:
        new_active_disjunctions = ComponentSet()
        new_active_disjuncts = ComponentSet()
        new_fixed_true_disjuncts = ComponentSet()
        new_active_constraints = ComponentSet()

        for container in new_containers:
            (next_active_disjunctions,
             next_fixed_true_disjuncts,
             next_active_disjuncts,
             next_active_constraints) = _process_active_container(model)
            new_active_disjunctions.update(next_active_disjunctions)
            new_active_disjuncts.update(next_active_disjuncts)
            new_fixed_true_disjuncts.update(next_fixed_true_disjuncts)
            new_active_constraints.update(next_active_constraints)

        new_containers = ((new_active_disjuncts - active_disjuncts) |
                          (new_fixed_true_disjuncts - fixed_true_disjuncts))

        active_disjunctions.update(new_active_disjunctions)
        active_disjuncts.update(new_active_disjuncts)
        fixed_true_disjuncts.update(new_fixed_true_disjuncts)
        active_constraints.update(new_active_constraints)

    active_vars.update(
        var for constr in new_active_constraints
        for var in EXPR.identify_variables(
            constr.body, include_fixed=False))
    active_vars.update(
        disj.indicator_var for disj in active_disjuncts)

    report.active = Container()
    report.active.variables = len(active_vars)
    report.active.binary_variables = sum(
        1 for v in active_vars if v.is_binary())
    report.active.integer_variables = sum(
        1 for v in active_vars if v.is_integer())
    report.active.continuous_variables = sum(
        1 for v in active_vars if v.is_continuous())
    report.active.disjunctions = len(active_disjunctions)
    report.active.disjuncts = len(active_disjuncts)

    report.overall = Container()
    block_like = (Block, Disjunct)
    all_vars = ComponentSet(
        model.component_data_objects(Var, descend_into=block_like))
    report.overall.variables = len(all_vars)
    report.overall.binary_variables = sum(1 for v in all_vars if v.is_binary())
    report.overall.integer_variables = sum(
        1 for v in all_vars if v.is_integer())
    report.overall.continuous_variables = sum(
        1 for v in all_vars if v.is_continuous())
    report.overall.disjunctions = sum(
        1 for d in model.component_data_objects(
            Disjunction, descend_into=block_like))
    report.overall.disjuncts = sum(
        1 for d in model.component_data_objects(
            Disjunct, descend_into=block_like))

    report.warn = Container()
    report.warn.unassociated_disjuncts = sum(
        1 for d in model.component_data_objects(
            Disjunct, descend_into=block_like)
        if not d.indicator_var.fixed and d not in active_disjuncts)

    return report


def log_model_size_report(model, logger=default_logger):
    """Generate a report logging the model size."""
    logger.info(build_model_size_report(model))


def _process_active_container(blk):
    """Process a container object, returning the new components found."""
    new_fixed_true_disjuncts = ComponentSet(
        disj for disj in blk.component_data_objects(Disjunct, active=True)
        if disj.indicator_var.value == 1 and disj.indicator_var.fixed)
    new_active_disjunctions = ComponentSet(
        blk.component_data_objects(Disjunction, active=True))
    new_active_disjuncts = ComponentSet(
        disj for disjtn in new_active_disjunctions
        for disj in _active_disjuncts_in_disjunction(disjtn))
    new_active_constraints = ComponentSet(
        blk.component_data_objects(Constraint, active=True))
    return (
        new_active_disjunctions,
        new_fixed_true_disjuncts,
        new_active_disjuncts,
        new_active_constraints
    )


def _active_disjuncts_in_disjunction(disjtn):
    """Retrieve generator of active disjuncts on disjunction."""
    return (disj for disj in disjtn.disjuncts
            if disj.active and not disj.indicator_var.fixed)
