# -*- coding: utf-8 -*-
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

"""Module with diagnostic utilities for infeasible models."""
from pyomo.core import Constraint, Var, value
from math import fabs
import logging

from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints

logger = logging.getLogger(__name__)

def find_infeasible_constraints(m, tol=1E-6):
    """Find the infeasible constraints in the model.

    Uses the current model state.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    Yields
    ------
    constr: ConstraintData
        The infeasible constraint object

    lb_value: float or None
        The numeric value of the constraint lower bound (or None)

    body_value: float or None
        The numeric value of the constraint body (or None if there was an
        error evaluating the expression)

    ub_value: float or None
        The numeric value of the constraint upper bound (or None)

    infeasible: int
        A bitmask indicating which bound was infeasible (1 for the lower
        bound or 2 for the upper bound)

    """
    # Iterate through all active constraints on the model
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        body_value = value(constr.body, exception=False)
        lb_value = value(constr.lower, exception=False)
        ub_value = value(constr.upper, exception=False)

        if body_value is None:
            # Undefined constraint body value due to missing variable value
            pass
        else:
            # Check for infeasibilities
            if constr.equality:
                if fabs(lb_value - body_value) < tol:
                    continue
                infeasible = 0
            else:
                infeasible = 0
                if constr.has_lb() and lb_value - body_value >= tol:
                    infeasible |= 1
                if constr.has_ub() and body_value - ub_value >= tol:
                    infeasible |= 2
                if not infeasible:
                    continue

        yield constr, lb_value, body_value, ub_value, infeasible


def log_infeasible_constraints(
        m, tol=1E-6, logger=logger,
        log_expression=False, log_variables=False
):
    """Logs the infeasible constraints in the model.

    Uses the current model state.  Messages are logged at the INFO level.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    log_expression: bool
        If true, prints the constraint expression

    log_variables: bool
        If true, prints the constraint variable names and values

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning(
            'log_infeasible_constraints() called with a logger whose '
            'effective level is higher than logging.INFO: no output '
            'will be logged reguardless of constraint feasibility'
        )

    for constr, lb, body, ub, infeas in find_infeasible_constraints(m, tol):
        if constr.equality:
            lb = lb_expr = lb_op = ""
            ub_expr = constr.upper
            if body is None:
                ub_op = " =?= "
            else:
                ub_op = " =/= "
        else:
            if constr.has_lb():
                lb_expr = constr.lower
                if body is None:
                    lb_op = " <?= "
                elif infeas & 1:
                    lb_op = " </= "
                else:
                    lb_op = " <= "
            else:
                lb = lb_expr = lb_op = ""

            if constr.has_ub():
                ub_expr = constr.upper
                if body is None:
                    ub_op = " <?= "
                elif infeas & 2:
                    ub_op = " </= "
                else:
                    ub_op = " <= "
            else:
                ub = ub_expr = ub_op = ""
        if body is None:
            body = "evaluation error"

        line = f"CONSTR {constr.name}: {lb}{lb_op}{body}{ub_op}{ub}"
        if log_expression:
            line += (
                f"\n  - EXPR: {lb_expr}{lb_op}{constr.body}{ub_op}{ub_expr}"
            )
        if log_variables:
            line += ''.join(
                f"\n  - VAR {v.name}: {v.value}"
                for v in identify_variables(constr.body, include_fixed=True)
            )

        logger.info(line)


def find_infeasible_bounds(m, tol=1E-6):
    """Find variables whose values are outside their bounds

    Uses the current model state. Variables with no values are returned
    as if they were infeasible.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    Yields
    ------
    var: VarData
        The variable that is outside its bounds

    infeasible: int
        A bitmask indicating which bound was infeasible (1 for the lower
        bound or 2 for the upper bound; 0 indicates the variable had no
        value)

    """
    for var in m.component_data_objects(
            ctype=Var, descend_into=True):
        val = var.value
        infeasible = 0
        if val is None:
            yield var, infeasible
        else:
            if var.has_lb() and var.lb - val >= tol:
                infeasible |= 1
            if var.has_ub() and val - var.ub >= tol:
                infeasible |= 2
            if infeasible:
                yield var, infeasible


def log_infeasible_bounds(m, tol=1E-6, logger=logger):
    """Logs the infeasible variable bounds in the model.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning(
            'log_infeasible_bounds() called with a logger whose '
            'effective level is higher than logging.INFO: no output '
            'will be logged reguardless of bound feasibility'
        )

    for var, infeas in find_infeasible_bounds(m, tol):
        if not infeas:
            logger.debug("Skipping VAR {} with no assigned value.")
            continue
        if infeas & 1:
            logger.info(f'VAR {var.name}: {var.value} >/= LB {var.lb}')
        if infeas & 2:
            logger.info(f'VAR {var.name}: {var.value} </= UB {var.ub}')


def find_close_to_bounds(m, tol=1E-6):
    """Find variables and constraints whose values are close to their bounds.

    Uses the current model state. Variables with no values and
    Constraints with evaluation errors are returned as if they were
    close to their bounds.

    Note
    ----
    This will omit variables and constraints in several situations:
      - Equality constraints are omitted (as they should always
        be close to their bounds!).
      - Range constraints where both the upper and lower bounds are
        close are omitted (these are basically equality constriants).
      - Fixed variables are omitted (this is analogous to an equality
        constriant).
      - Variables where both the upper and lower bounds are close are
        omitted (these are basically fixed variables).

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance: values within tol of the bound
        will be returned.

    Yields
    ------
    var: ComponentData
        The variable or Constraint that is close to its bounds

    val: float
        The value of the variable or constraint body

    close: int
        A bitmask indicating which bound(s) the value was close to (1
        for the lower bound or 2 for the upper bound; 0 indicates the
        variable or constraint had no value or evaluating the constraint
        generated an error)

    """
    for var in m.component_data_objects(ctype=Var, descend_into=True):
        if var.fixed:
            continue
        val = var.value
        close = 0
        if val is None:
            yield var, val, close
        else:
            if var.has_lb() and fabs(var.lb - val) <= tol:
                close |= 1
            if var.has_ub() and fabs(val - var.ub) <= tol:
                close |= 2
            if close == 3:
                # The bounds are too close: skip this Var (it is
                # effectively fixed)
                continue
            if close:
                yield var, val, close

    for con in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        if con.equality:
            continue
        val = value(con.body, exception=False)
        close = 0
        if val is None:
            yield con, val, close
        else:
            if con.has_lb() and fabs(con.lb - val) <= tol:
                close |= 1
            if con.has_ub() and fabs(val - con.ub) <= tol:
                close |= 2
            if close == 3:
                # The bounds are too close: skip this Constraint (it is
                # effectively an equality)
                continue
            if close:
                yield con, val, close


def log_close_to_bounds(m, tol=1E-6, logger=logger):
    """Print the variables and constraints that are near their bounds.

    See :py:func:`find_close_to_bounds()` for a description of the
    variables and constraints that are returned (and which are omitted).

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning(
            'log_close_to_bounds() called with a logger whose '
            'effective level is higher than logging.INFO: no output '
            'will be logged reguardless of bound status'
        )

    for obj, val, close in find_close_to_bounds(m, tol):
        if not close:
            if obj.ctype is Var:
                logger.debug(f"Skipping VAR {obj.name} with no assigned value.")
            elif obj.ctype is Constraint:
                logger.info(f"Skipping CONSTR {obj.name}: evaluation error.")
            else:
                logger.error(
                    f"Object {obj.name} was neither a Var nor Constraint")
            continue
        if close & 1:
            logger.info(f'{obj.name} near LB of {obj.lb}')
        if close & 2:
            logger.info(f'{obj.name} near UB of {obj.ub}')
    return


@deprecated("log_active_constraints is deprecated.  "
            "Please use pyomo.util.blockutil.log_model_constraints()",
            version="5.7.3")
def log_active_constraints(m, logger=logger):
    log_model_constraints(m, logger)
