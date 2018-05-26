# -*- coding: UTF-8 -*-
"""Module with diagnostic utilities for infeasible models."""
from pyomo.core import Constraint, Var, value, TraversalStrategy
from math import fabs
import logging

logger = logging.getLogger('pyomo.util.infeasible')
logger.setLevel(logging.INFO)


def log_infeasible_constraints(m, tol=1E-6, logger=logger):
    """Print the infeasible constraints in the model.

    Uses the current model state. Uses pyomo.util.infeasible logger unless one
    is provided.

    Args:
        m (Block): Pyomo block or model to check
        tol (float): feasibility tolerance

    """
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        # if constraint is an equality, handle differently
        if constr.equality and fabs(value(constr.lower - constr.body)) >= tol:
            logger.info('CONSTR {}: {} != {}'.format(
                constr.name, value(constr.body), value(constr.lower)))
            continue
        # otherwise, check LB and UB, if they exist
        if constr.has_lb() and value(constr.lower - constr.body) >= tol:
            logger.info('CONSTR {}: {} < {}'.format(
                constr.name, value(constr.body), value(constr.lower)))
        if constr.has_ub() and value(constr.body - constr.upper) >= tol:
            logger.info('CONSTR {}: {} > {}'.format(
                constr.name, value(constr.body), value(constr.upper)))


def log_infeasible_bounds(m, tol=1E-6, logger=logger):
    """Print the infeasible variable bounds in the model.

    Args:
        m (Block): Pyomo block or model to check
        tol (float): feasibility tolerance

    """
    for var in m.component_data_objects(
            ctype=Var, descend_into=True):
        if var.has_lb() and value(var.lb - var) >= tol:
            logger.info('VAR {}: {} < LB {}'.format(
                var.name, value(var), value(var.lb)))
        if var.has_ub() and value(var - var.ub) >= tol:
            logger.info('VAR {}: {} > UB {}'.format(
                var.name, value(var), value(var.ub)))


def log_close_to_bounds(m, tol=1E-6, logger=logger):
    """Print the variables and constraints that are near their bounds.

    Fixed variables and equality constraints are excluded from this analysis.

    Args:
        m (Block): Pyomo block or model to check
        tol (float): bound tolerance
    """
    for var in m.component_data_objects(
            ctype=Var, descend_into=True):
        if var.fixed:
            continue
        if (var.has_lb() and var.has_ub() and
                fabs(value(var.ub - var.lb)) <= 2 * tol):
            continue  # if the bounds are too close, skip.
        if var.has_lb() and fabs(value(var.lb - var)) <= tol:
            logger.info('{} near LB of {}'.format(var.name, value(var.lb)))
        elif var.has_ub() and fabs(value(var.ub - var)) <= tol:
            logger.info('{} near UB of {}'.format(var.name, value(var.ub)))

    for constr in m.component_data_objects(
            ctype=Constraint, descend_into=True, active=True):
        if not constr.equality:
            if (constr.has_ub() and
                    fabs(value(constr.body - constr.upper)) <= tol):
                logger.info('{} near UB'.format(constr.name))
            if (constr.has_lb() and
                    fabs(value(constr.body - constr.lower)) <= tol):
                logger.info('{} near LB'.format(constr.name))


def log_active_constraints(m, logger=logger):
    """Prints the active constraints in the model."""
    for constr in m.component_data_objects(
        ctype=Constraint, active=True, descend_into=True,
        descent_order=TraversalStrategy.PrefixDepthFirstSearch
    ):
        logger.info("%s active" % constr.name)
