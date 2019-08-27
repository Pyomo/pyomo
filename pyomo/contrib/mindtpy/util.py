"""Utility functions and classes for the MindtPy solver."""
from __future__ import division

import logging
from math import fabs, floor, log

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals,
                        Objective, Reals, Suffix, Var, minimize, value)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
    """
    pass


def model_is_valid(solve_data, config):
    """Validate that the model is solveable by MindtPy.

    Also preforms some preprocessing such as moving the objective to the
    constraints.

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if (prob.number_of_binary_variables == 0 and
        prob.number_of_integer_variables == 0 and
            prob.number_of_disjunctions == 0):
        config.logger.info('Problem has no discrete decisions.')
        if len(MindtPy.working_nonlinear_constraints) > 0:
            config.logger.info(
                "Your model is an NLP (nonlinear program). "
                "Using NLP solver %s to solve." % config.nlp)
            SolverFactory(config.nlp).solve(
                solve_data.original_model, **config.nlp_options)
            return False
        else:
            config.logger.info(
                "Your model is an LP (linear program). "
                "Using LP solver %s to solve." % config.mip)
            SolverFactory(config.mip).solve(
                solve_data.original_model, **config.mip_options)
            return False

    if not hasattr(m, 'dual'):  # Set up dual value reporting
        m.dual = Suffix(direction=Suffix.IMPORT)

    # TODO if any continuous variables are multipled with binary ones, need
    # to do some kind of transformation (Glover?) or throw an error message
    return True


def calc_jacobians(solve_data, config):
    """Generate a map of jacobians."""
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    solve_data.jacobians = ComponentMap()
    for c in solve_data.mip.MindtPy_utils.constraint_list:
        if c.body.polynomial_degree() in (1, 0):
            continue  # skip linear constraints
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(c.body, wrt_list=vars_in_constr)
        solve_data.jacobians[c] = ComponentMap(
            (var, jac_wrt_var)
            for var, jac_wrt_var in zip(vars_in_constr, jac_list))


def add_feas_slacks(m):
    MindtPy = m.MindtPy_utils
    # generate new constraints
    for i, constr in enumerate(MindtPy.constraint_list, 1):
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        c = MindtPy.MindtPy_feas.feas_constraints.add(
            constr.body - rhs
            <= MindtPy.MindtPy_feas.slack_var[i])


