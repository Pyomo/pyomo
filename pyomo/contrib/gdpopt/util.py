"""Utility functions and classes for the GDPopt solver."""
from __future__ import division

import logging
from math import fabs

from pyomo.core import value, Var, Constraint


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class GDPoptSolveData(object):
    """Data container to hold solve-instance data."""
    pass


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def copy_values(from_model, to_model, config):
    """Copy variable values from one model to another."""
    for v_from, v_to in zip(from_model.GDPopt_utils.initial_var_list,
                            to_model.GDPopt_utils.initial_var_list):
        if v_from.is_stale():
            # Skip stale variable values.
            continue
        try:
            v_to.set_value(v_from.value)
        except ValueError as err:
            if 'is not in domain Binary' in err.message:
                # Check to see if this is just a tolerance issue
                if (fabs(v_from.value - 1) <= config.integer_tolerance or
                        fabs(v_from.value) <= config.integer_tolerance):
                    v_to.set_value(round(v_from.value))
                else:
                    raise


def is_feasible(model, config):
    config.logger.info('Checking if model is feasible.')
    for constr in model.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        # Check constraint lower bound
        if (constr.lower is not None and (
            value(constr.lower) - value(constr.body)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s < LB %s' % (
                constr.name, value(constr.body), value(constr.lower)))
            return False
        # check constraint upper bound
        if (constr.upper is not None and (
            value(constr.body) - value(constr.upper)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s > UB %s' % (
                constr.name, value(constr.body), value(constr.upper)))
            return False
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        # Check variable lower bound
        if (var.has_lb() and
                value(var.lb) - value(var) >= config.variable_tolerance):
            config.logger.info('%s: %s < LB %s' % (
                var.name, value(var), value(var.lb)))
            return False
        # Check variable upper bound
        if (var.has_ub() and
                value(var) - value(var.ub) >= config.variable_tolerance):
            config.logger.info('%s: %s > UB %s' % (
                var.name, value(var), value(var.ub)))
            return False
    config.logger.info('Model is feasible.')
    return True
