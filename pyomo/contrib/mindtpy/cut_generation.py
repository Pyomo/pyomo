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
    sign_adjust = -1 if MindtPy.objective.sense == minimize else 1

    # Copy values over
    for var, val in zip(MindtPy.var_list, var_values):
        if val is not None and not var.fixed:
            var.value = val

    # Copy duals over
    for constr, dual_value in zip(MindtPy.constraints, duals):
        m.dual[constr] = dual_value

    # generate new constraints
    # TODO some kind of special handling if the dual is phenomenally small?
    jacs = solve_data.jacobians
    for constr in MindtPy.nonlinear_constraints:
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        MindtPy.MindtPy_linear_cuts.oa_cuts.add(
            expr=copysign(1, sign_adjust * m.dual[constr]) * (sum(
                value(jacs[constr][var]) * (var - value(var))
                for var in list(EXPR.identify_variables(constr.body))) +
                value(constr.body) - rhs) +
            MindtPy.MindtPy_linear_cuts.slack_vars[
                solve_data.nlp_iter,
                MindtPy.nl_map[constr]] <= 0)


def add_ecp_cut(solve_data, config):
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    MindtPy.MindtPy_linear_cuts.mip_iters.add(solve_data.mip_iter)
    # generate new constraints
    # TODO some kind of special handling if the dual is phenomenally small?
    gen = (constr for constr in MindtPy.jacs
           if (0 if constr.upper is None
               else abs(value(constr.body) - constr.upper)) +
           (0 if constr.lower is None
            else abs(constr.lower - value(constr.body)))
           > config.ECP_tolerance)
    for constr in gen:
        constr_dir = -1 if value(constr.upper) is None else 1
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        # this only happens if a constraint is >=
        c = MindtPy.MindtPy_linear_cuts.ecp_cuts.add(
            expr=copysign(1, constr_dir)
            * (sum(value(MindtPy.jacs[constr][id(var)]) * (var - value(var))
                   for var in list(EXPR.identify_variables(constr.body))) +
               value(constr.body) - rhs) <= 0)
        MindtPy.ECP_constr_map[constr, solve_data.mip_iter] = c


def add_psc_cut(solve_data, config, nlp_feasible=True):
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    sign_adjust = 1 if MindtPy.obj.sense == minimize else -1

    # generate the sum of all multipliers with the nonlinear constraints
    var_to_val = {id(var): value(var)
                  for var in MindtPy.nonlinear_variables}

    def get_substituted_expr(expr):
        # Returns the expression with the substitution applied
        return ExpressionReplacementVisitor(
            substitute=var_to_val).dfs_postorder_stack(expr)
    sum_nonlinear = (
        # Address constraints of form f(x) <= upper
        sum(value(m.dual[c]) * -1 *
            (get_substituted_expr(c.body) - c.upper)
            for c in MindtPy.nonlinear_constraints
            if value(fabs(m.dual[c])) > config.small_dual_tolerance
            and c.upper is not None) +
        # Address constraints of form f(x) >= lower
        sum(value(m.dual[c]) *
            (c.lower - get_substituted_expr(c.body))
            for c in MindtPy.nonlinear_constraints
            if value(fabs(m.dual[c])) > config.small_dual_tolerance
            and c.lower is not None))
    # Generate the sum of all multipliers with linear constraints
    # containing nonlinear variables
    #
    # For now, need to generate canonical representation in order to get
    # the coefficients on the linear terms.
    lin_cons = [c for c in m.component_data_objects(
        ctype=Constraint, active=True, descend_into=True)
        if c.body.polynomial_degree() in (0, 1)]
    # Create a coefficient dictionary mapping variables to their
    # coefficient in the expression. Constraint -> (id(Var) -> coefficient)
    coef_dict = {}
    constr_vars = {}
    for constr in lin_cons:
        repn = generate_standard_repn(constr.body)
        if repn.variables is None or repn.linear is None:
            repn.variables = []
            repn.linear = []
        coef_dict[constr] = {id(var): coef for var, coef in
                             zip(repn.variables, repn.linear)}
        constr_vars[constr] = repn.variables
    sum_linear = sum(
        m.dual[c] *
        sum(coef_dict[c][id(var)] * (var - value(var))
            for var in constr_vars[c]
            if id(var) in solve_data.nonlinear_variable_IDs)
        for c in lin_cons
        if value(fabs(m.dual[c])) > config.small_dual_tolerance)

    # Generate the sum of all bound multipliers with nonlinear variables
    sum_var_bounds = (
        sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
            for var in MindtPy.nonlinear_variables
            if value(fabs(m.ipopt_zL_out.get(var, 0))) >
            config.small_dual_tolerance) +
        sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
            for var in MindtPy.nonlinear_variables
            if value(fabs(m.ipopt_zU_out.get(var, 0))) >
            config.small_dual_tolerance))

    if nlp_feasible:
        # Optimality cut (for feasible NLP)
        MindtPy.MindtPy_linear_cuts.psc_cuts.add(
            expr=MindtPy.obj.expr * sign_adjust >= sign_adjust * (
                MindtPy.obj.expr + sum_nonlinear + sum_linear +
                sum_var_bounds))
    else:
        # Feasibility cut (for infeasible NLP)
        MindtPy.MindtPy_linear_cuts.psc_cuts.add(
            expr=(sum_nonlinear + sum_linear + sum_var_bounds) <= 0)


def add_gbd_cut(solve_data, config, nlp_feasible=True):
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    sign_adjust = 1 if MindtPy.obj.sense == minimize else -1

    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True):
        if value(c.upper) is None and value(c.lower) is None:
            raise ValueError(
                'Oh no, Pyomo did something MindtPy does not expect. '
                'The value of c.upper for {} is None: {} <= {} <= {}'
                .format(c.name, c.lower, c.body, c.upper))
    # TODO handle the case where constraint upper and lower is None

    # only substitute non-binary variables to their values
    binary_vars = ComponentSet(MindtPy.binary_vars)
    var_to_val = {id(var): value(var)
                  for var in m.component_data_objects(ctype=Var,
                                                      descend_into=True)
                  if var not in binary_vars}

    def get_substituted_expr(expr):
        # Returns the expression with the substitution applied
        return ExpressionReplacementVisitor(
            substitute=var_to_val).dfs_postorder_stack(expr)
    # generate the sum of all multipliers with the active (from a duality
    # sense) constraints
    sum_constraints = (
        sum(value(m.dual[c]) * -1 *
            (get_substituted_expr(c.body) - c.upper)
            for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if value(fabs(m.dual[c])) > config.small_dual_tolerance
            and c.upper is not None) +
        sum(value(m.dual[c]) *
            (c.lower - get_substituted_expr(c.body))
            for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if value(fabs(m.dual[c])) > config.small_dual_tolerance
            and c.lower is not None))
    # and not c.upper == c.lower

    # add in variable bound dual contributions
    #
    # Generate the sum of all bound multipliers with nonlinear variables
    sum_var_bounds = (
        sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
            for var in m.component_data_objects(ctype=Var,
                                                descend_into=True)
            if (var not in binary_vars and
                value(abs(m.ipopt_zL_out.get(var, 0))) >
                config.small_dual_tolerance)) +
        sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
            for var in m.component_data_objects(ctype=Var,
                                                descend_into=True)
            if (var not in binary_vars and
                value(abs(m.ipopt_zU_out.get(var, 0))) >
                config.small_dual_tolerance)))

    if nlp_feasible:
        MindtPy.MindtPy_linear_cuts.gbd_cuts.add(
            expr=MindtPy.obj.expr * sign_adjust >= sign_adjust * (
                value(MindtPy.obj.expr) + sum_constraints + sum_var_bounds))
    else:
        if sum_constraints + sum_var_bounds != 0:
            MindtPy.MindtPy_linear_cuts.gbd_cuts.add(
                expr=(sum_constraints + sum_var_bounds) <= 0)


def add_int_cut(var_values, solve_data, config, feasible=False):
    if not config.integer_cuts:
        return

    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    int_tol = config.integer_tolerance

    # copy variable values over
    for var, val in zip(MindtPy.var_list, var_values):
        if not var.is_binary():
            continue
        var.value = val

    # check to make sure that binary variables are all 0 or 1
    for v in MindtPy.binary_vars:
        if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
            raise ValueError('Binary {} = {} is not 0 or 1'.format(
                v.name, value(v)))

    if not MindtPy.binary_vars:  # if no binary variables, skip.
        return

    int_cut = (sum(1 - v for v in MindtPy.binary_vars
                   if value(abs(v - 1)) <= int_tol) +
               sum(v for v in MindtPy.binary_vars
                   if value(abs(v)) <= int_tol) >= 1)

    if not feasible:
        # Add the integer cut
        MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
    else:
        MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)
