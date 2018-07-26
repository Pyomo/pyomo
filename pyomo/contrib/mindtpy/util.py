"""Utility functions and classes for the MindtPy solver."""
from __future__ import division

import logging
from math import fabs, floor, log

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals,
                        Objective, Reals, Suffix, Var, minimize, value)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
        - linear_GDP: the linear-discrete master problem
    """
    pass


def model_is_valid(solve_data, config):
    """Validate that the model is solveable by MindtPy.

    Also preforms some preprocessing such as moving the objective to the
    constraints.

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    # Check for any integer variables
    if any(True for v in m.component_data_objects(
            ctype=Var, descend_into=True)
            if v.is_integer() and not v.fixed):
        raise ValueError('Model contains unfixed integer variables. '
                         'MindtPy does not currently support solution of '
                         'such problems.')
        # TODO add in the reformulation using base 2

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

    # Handle missing or multiple objectives
    objs = list(m.component_data_objects(
        ctype=Objective, active=True, descend_into=True))
    num_objs = len(objs)
    solve_data.results.problem.number_of_objectives = num_objs
    if num_objs == 0:
        config.logger.warning(
            'Model has no active objectives. Adding dummy objective.')
        MindtPy.dummy_objective = Objective(expr=1)
        main_obj = MindtPy.dummy_objective
    elif num_objs > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        main_obj = objs[0]
    solve_data.working_objective_expr = main_obj.expr

    # Move the objective to the constraints

    # TODO only move the objective if nonlinear?
    MindtPy.objective_value = Var(domain=Reals, initialize=0)
    solve_data.objective_sense = main_obj.sense
    if main_obj.sense == minimize:
        MindtPy.objective_expr = Constraint(
            expr=MindtPy.objective_value >= main_obj.expr)
        solve_data.results.problem.sense = ProblemSense.minimize
    else:
        MindtPy.objective_expr = Constraint(
            expr=MindtPy.objective_value <= main_obj.expr)
        solve_data.results.problem.sense = ProblemSense.maximize
    main_obj.deactivate()
    MindtPy.objective = Objective(
        expr=MindtPy.objective_value, sense=main_obj.sense)

    if not hasattr(m, 'dual'):  # Set up dual value reporting
        m.dual = Suffix(direction=Suffix.IMPORT)

    # TODO if any continuous variables are multipled with binary ones, need
    # to do some kind of transformation (Glover?) or throw an error message
    return True


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def build_ordered_component_lists(model):
    """Define lists used for future data transfer."""
    MindtPy = model.MindtPy_utils
    var_set = ComponentSet()
    MindtPy.constraints = list(model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True))

    # Identify the non-fixed variables in (potentially) active constraints
    for constr in MindtPy.constraints:
        for v in EXPR.identify_variables(constr.body, include_fixed=False):
            var_set.add(v)

    # We use component_data_objects rather than list(var_set) in order to
    # preserve a deterministic ordering.
    MindtPy.var_list = list(
        v for v in model.component_data_objects(ctype=Var, descend_into=True)
        if v in var_set)
    MindtPy.binary_vars = list(v for v in MindtPy.var_list if v.is_binary())
    MindtPy.nonlinear_constraints = list(
        c for c in MindtPy.constraints
        if c.body.polynomial_degree() not in (0, 1))


def copy_values(from_model, to_model, config, skip_stale=False):
    """Copy variable values from one model to another."""
    copy_var_list_values(from_model.MindtPy_utils.var_list,
                         to_model.MindtPy_utils.var_list,
                         config, skip_stale)


def copy_var_list_values(from_list, to_list, config, skip_stale=False):
    """Copy variable values from one list to another."""
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        try:
            v_to.set_value(value(v_from, exception=False))
            if skip_stale:
                v_to.stale = False
        except ValueError as err:
            if 'is not in domain Binary' in str(err):
                # Check to see if this is just a tolerance issue
                v_from_val = value(v_from, exception=False)
                if (fabs(v_from_val - 1) <= config.integer_tolerance or
                        fabs(v_from_val) <= config.integer_tolerance):
                    v_to.set_value(round(v_from_val))
                else:
                    # Simply do not copy if there is a binary domain violation.
                    continue
            if 'is not in domain NonNegativeReals' in str(err):
                v_from_val = value(v_from, exception=False)
                if fabs(v_from_val) <= config.zero_tolerance:
                    v_to.set_value(0)
                else:
                    raise
            else:
                raise


def detect_nonlinear_vars(solve_data, config):
    """Identify the variables that participate in nonlinear terms."""
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    nonlinear_var_set = ComponentSet()

    for constr in MindtPy.nonlinear_constraints:
        if isinstance(constr.body, EXPR.SumExpression):
            # go through each term and check to see if the term is
            # nonlinear
            for expr in constr.body.args():
                # Check to see if the expression is nonlinear
                if expr.polynomial_degree() not in (0, 1):
                    # collect variables
                    nonlinear_var_set.update(
                        EXPR.identify_variables(expr, include_fixed=False))
        # if the root expression object is not a summation, then something
        # else is the cause of the nonlinearity. Collect all participating
        # variables.
        else:
            # collect variables
            nonlinear_var_set.update(
                EXPR.identify_variables(constr.body, include_fixed=False))

    MindtPy.nonlinear_variables = list(nonlinear_var_set)


def calc_jacobians(solve_data, config):
    """Generate a map of jacobians."""
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    solve_data.jacobians = ComponentMap()
    for c in solve_data.mip.MindtPy_utils.nonlinear_constraints:
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(c.body, wrt_list=vars_in_constr)
        solve_data.jacobians[c] = ComponentMap(
            (var, jac_wrt_var)
            for var, jac_wrt_var in zip(vars_in_constr, jac_list))


def add_feas_slacks(solve_data, config):
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    # generate new constraints
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        c = MindtPy.MindtPy_feas.feas_constraints.add(
            constr.body - rhs
            <= MindtPy.MindtPy_feas.slack_var[solve_data.feas_map[constr]])
        MindtPy.feas_constr_map[constr, solve_data.nlp_iter] = c
