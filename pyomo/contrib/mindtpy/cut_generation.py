"""Cut generation."""
from __future__ import division

from math import copysign

from pyomo.core import Constraint, minimize, value
from pyomo.core.expr import current as EXPR
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables


def add_objective_linearization(solve_data, config):
    """Adds initial linearized objective in case it is nonlinear.

    This should be done for initializing the ECP method.

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    solve_data.mip_iter += 1
    gen = (obj for obj in MindtPy.jacs
           if obj is MindtPy.MindtPy_objective_expr)
    MindtPy.MindtPy_linear_cuts.mip_iters.add(solve_data.mip_iter)
    sign_adjust = 1 if MindtPy.obj.sense == minimize else -1
    # generate new constraints
    # TODO some kind of special handling if the dual is phenomenally small?
    for obj in gen:
        c = MindtPy.MindtPy_linear_cuts.ecp_cuts.add(
            expr=sign_adjust * sum(
                value(MindtPy.jacs[obj][id(var)]) * (var - value(var))
                for var in list(EXPR.identify_variables(obj.body))) +
                 value(obj.body) <= 0)
        MindtPy.ECP_constr_map[obj, solve_data.mip_iter] = c


def add_oa_cuts(target_model, dual_values, solve_data, config,
                linearize_active=True,
                linearize_violated=True,
                linearize_inactive=False,
                use_slack_var=False):
    """Linearizes nonlinear constraints.

    For nonconvex problems, turn on 'use_slack_var'. Slack variables will
    always be used for nonlinear equality constraints.
    """
    for (constr, dual_value) in zip(target_model.MindtPy_utils.constraint_list,
                                    dual_values):
        if constr.body.polynomial_degree() in (0, 1):
            continue

        constr_vars = list(identify_variables(constr.body))
        jacs = solve_data.jacobians

        # Equality constraint (makes the problem nonconvex)
        if constr.has_ub() and constr.has_lb() and constr.upper == constr.lower:
            sign_adjust = -1 if solve_data.objective_sense == minimize else 1
            rhs = ((0 if constr.upper is None else constr.upper)
                   + (0 if constr.lower is None else constr.lower))
            rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
            slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()
            target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * dual_value)
                     * (sum(value(jacs[constr][var]) * (var - value(var))
                            for var in list(EXPR.identify_variables(constr.body)))
                        + value(constr.body) - rhs)
                     - slack_var <= 0)

        else:  # Inequality constraint (possibly two-sided)
            if constr.has_ub() \
               and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) \
                    or (linearize_violated and constr.uslack() < 0) \
                    or (linearize_inactive and constr.uslack() > 0):
                if use_slack_var:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars)
                          - (slack_var if use_slack_var else 0)
                          <= constr.upper)
                )

            if constr.has_lb() \
               and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) \
                    or (linearize_violated and constr.lslack() < 0) \
                    or (linearize_inactive and constr.lslack() > 0):
                if use_slack_var:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars)
                          + (slack_var if use_slack_var else 0)
                          >= constr.lower)
                )


def add_oa_equality_relaxation(var_values, duals, solve_data, config, ignore_integrality=False):
    """More general case for outer approximation

    This method covers nonlinear inequalities g(x)<=b and g(x)>=b as well as 
    equalities g(x)=b all in the same linearization call. It combines the dual
    with the objective sense to figure out how to generate the cut.
    Note that the dual sign is defined as follows (according to IPOPT):
      sgn  | min | max
    -------|-----|-----
    g(x)<=b|  +1 | -1
    g(x)>=b|  -1 | +1

    Note additionally that the dual value is not strictly neccesary for inequality
    constraints, but definitely neccesary for equality constraints. For equality 
    constraints the cut will always be generated so that the side with the worse objective
    function is the 'interior'.

    ignore_integrality: Accepts float values for discrete variables.
                        Useful for cut in initial relaxation
    """

    m = solve_data.mip
    MindtPy = m.MindtPy_utils
    MindtPy.MindtPy_linear_cuts.nlp_iters.add(solve_data.nlp_iter)
    sign_adjust = -1 if solve_data.objective_sense == minimize else 1

    copy_var_list_values(from_list=var_values,
                         to_list=MindtPy.variable_list,
                         config=config,
                         ignore_integrality=ignore_integrality)

    # generate new constraints
    # TODO some kind of special handling if the dual is phenomenally small?
    # TODO-romeo conditional for 'global' option, i.e. slack or no slack
    jacs = solve_data.jacobians
    for constr, dual_value in zip(MindtPy.constraint_list, duals):
        if constr.body.polynomial_degree() in (1, 0):
            continue
        rhs = ((0 if constr.upper is None else constr.upper)
               + (0 if constr.lower is None else constr.lower))
        # Properly handle equality constraints and ranged inequalities
        # TODO special handling for ranged inequalities? a <= x <= b
        rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
        slack_var = MindtPy.MindtPy_linear_cuts.slack_vars.add()
        MindtPy.MindtPy_linear_cuts.oa_cuts.add(
            expr=copysign(1, sign_adjust * dual_value)
                 * (sum(value(jacs[constr][var]) * (var - value(var))
                        for var in list(EXPR.identify_variables(constr.body)))
                    + value(constr.body) - rhs)
                 - slack_var <= 0)


def add_int_cut(var_values, solve_data, config, feasible=False):
    if not config.integer_cuts:
        return

    config.logger.info("Adding integer cuts")

    m = solve_data.mip
    MindtPy = m.MindtPy_utils
    int_tol = config.integer_tolerance

    binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]

    # copy variable values over
    for var, val in zip(MindtPy.variable_list, var_values):
        if not var.is_binary():
            continue
        var.value = val

    # check to make sure that binary variables are all 0 or 1
    for v in binary_vars:
        if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
            raise ValueError('Binary {} = {} is not 0 or 1'.format(
                v.name, value(v)))

    if not binary_vars:  # if no binary variables, skip.
        return

    int_cut = (sum(1 - v for v in binary_vars
                   if value(abs(v - 1)) <= int_tol) +
               sum(v for v in binary_vars
                   if value(abs(v)) <= int_tol) >= 1)

    MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)

    # TODO need to handle theoretical implications of backtracking
    # if not feasible:
    #     # Add the integer cut
    #     MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
    # else:
    #     MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)
