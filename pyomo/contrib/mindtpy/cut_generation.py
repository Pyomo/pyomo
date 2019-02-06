"""Cut generation."""
from __future__ import division

from math import copysign, fabs

from pyomo.core import Constraint, Var, minimize, value
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.current import ExpressionReplacementVisitor
from pyomo.repn import generate_standard_repn
from pyomo.core.kernel.component_set import ComponentSet


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


def add_oa_cut(var_values, duals, solve_data, config):
    m = solve_data.mip
    MindtPy = m.MindtPy_utils
    MindtPy.MindtPy_linear_cuts.nlp_iters.add(solve_data.nlp_iter)
    sign_adjust = -1 if solve_data.objective_sense == minimize else 1

    # Copy values over
    for var, val in zip(MindtPy.variable_list, var_values):
        if val is not None and not var.fixed:
            var.value = val

    # generate new constraints
    # TODO some kind of special handling if the dual is phenomenally small?
    jacs = solve_data.jacobians
    for constr, dual_value in zip(MindtPy.constraint_list, duals):
        if constr.body.polynomial_degree() in (1, 0):
            continue
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        # Properly handle equality constraints and ranged inequalities
        # TODO special handling for ranged inequalities? a <= x <= b
        rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
        slack_var = MindtPy.MindtPy_linear_cuts.slack_vars.add()
        MindtPy.MindtPy_linear_cuts.oa_cuts.add(
            expr=copysign(1, sign_adjust * dual_value) * (sum(
                value(jacs[constr][var]) * (var - value(var))
                for var in list(EXPR.identify_variables(constr.body))) +
                                                          value(constr.body) - rhs) - slack_var <= 0)


def add_int_cut(var_values, solve_data, config, feasible=False):
    if not config.integer_cuts:
        return

    m = solve_data.working_model
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

    if not feasible:
        # Add the integer cut
        MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
    else:
        MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)
