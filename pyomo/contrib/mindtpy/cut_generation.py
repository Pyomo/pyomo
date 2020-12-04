#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Cut generation."""
from __future__ import division
import logging
from math import copysign

from pyomo.core import minimize, value, Block, ConstraintList
from pyomo.core.expr import current as EXPR
from pyomo.contrib.gdpopt.util import identify_variables
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error

logger = logging.getLogger('pyomo.contrib.mindtpy')


def add_objective_linearization(solve_data, config):
    """
    If objective is nonlinear, then this function adds a linearized objective. This function should be used to
    initialize the ECP method.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
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


'''
def add_oa_cuts(target_model, dual_values, solve_data, config,
                linearize_active=True,
                linearize_violated=True):
    """
    Linearizes nonlinear constraints; modifies the model to include the OA cuts

    For nonconvex problems, turn on 'config.add_slack'. Slack variables will
    always be used for nonlinear equality constraints.

    Parameters
    ----------
    target_model:
        this is the MIP/MILP model for the OA algorithm; we want to add the OA cuts to 'target_model'
    dual_values:
        contains the value of the duals for each constraint
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    linearize_active: bool, optional
        this parameter acts as a Boolean flag that signals whether the linearized constraint is active
    linearize_violated: bool, optional
        this parameter acts as a Boolean flag that signals whether the nonlinear constraint represented by the
        linearized constraint has been violated
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
            rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
            if config.add_slack:
                slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()
            target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * dual_value)
                * (sum(value(jacs[constr][var]) * (var - value(var))
                       for var in list(EXPR.identify_variables(constr.body)))
                    + value(constr.body) - rhs)
                - (slack_var if config.add_slack else 0) <= 0)

        else:  # Inequality constraint (possibly two-sided)
            if constr.has_ub() \
                and (linearize_active and abs(constr.uslack()) < config.bound_tolerance) \
                    or (linearize_violated and constr.uslack() < 0) \
                    or (config.linearize_inactive and constr.uslack() > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars) + value(constr.body)
                          - (slack_var if config.add_slack else 0)
                          <= constr.upper)
                )

            if constr.has_lb() \
                and (linearize_active and abs(constr.lslack()) < config.bound_tolerance) \
                    or (linearize_violated and constr.lslack() < 0) \
                    or (config.linearize_inactive and constr.lslack() > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars) + value(constr.body)
                          + (slack_var if config.add_slack else 0)
                          >= constr.lower)
                )
'''


def add_oa_cuts(target_model, dual_values, solve_data, config,
                linearize_active=True,
                linearize_violated=True):
    """
    Linearizes nonlinear constraints; modifies the model to include the OA cuts
    For nonconvex problems, turn on 'config.add_slack'. Slack variables will
    always be used for nonlinear equality constraints.
    Parameters
    ----------
    target_model:
        this is the MIP/MILP model for the OA algorithm; we want to add the OA cuts to 'target_model'
    dual_values:
        contains the value of the duals for each constraint
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    linearize_active: bool, optional
        this parameter acts as a Boolean flag that signals whether the linearized constraint is active
    linearize_violated: bool, optional
        this parameter acts as a Boolean flag that signals whether the nonlinear constraint represented by the
        linearized constraint has been violated
    """
    for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
        if constr.body.polynomial_degree() in (0, 1):
            continue

        constr_vars = list(identify_variables(constr.body))
        jacs = solve_data.jacobians

        # Equality constraint (makes the problem nonconvex)
        if constr.has_ub() and constr.has_lb() and constr.upper == constr.lower and config.use_dual:
            sign_adjust = -1 if solve_data.objective_sense == minimize else 1
            rhs = constr.lower
            if config.add_slack:
                slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()
            target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * dual_values[index])
                * (sum(value(jacs[constr][var]) * (var - value(var))
                       for var in list(EXPR.identify_variables(constr.body)))
                    + value(constr.body) - rhs)
                - (slack_var if config.add_slack else 0) <= 0)

        else:  # Inequality constraint (possibly two-sided)
            if constr.has_ub() \
                and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) \
                    or (linearize_violated and constr.uslack() < 0) \
                    or (config.linearize_inactive and constr.uslack() > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars) + value(constr.body)
                          - (slack_var if config.add_slack else 0)
                          <= constr.upper)
                )

            if constr.has_lb() \
                and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) \
                    or (linearize_violated and constr.lslack() < 0) \
                    or (config.linearize_inactive and constr.lslack() > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.oa_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars) + value(constr.body)
                          + (slack_var if config.add_slack else 0)
                          >= constr.lower)
                )


def add_ecp_cuts(target_model, solve_data, config,
                 linearize_active=True,
                 linearize_violated=True):
    """
    Linearizes nonlinear constraints. Adds the cuts for the ECP method.

    For nonconvex problems, turn on 'config.add_slack'. Slack variables will
    always be used for nonlinear equality constraints.

    Parameters
    ----------
    target_model:
        this is the MIP/MILP model for the OA algorithm; we want to add the OA cuts to 'target_model'
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    linearize_active: bool, optional
        this parameter acts as a Boolean flag that signals whether the linearized constraint is active
    linearize_violated: bool, optional
        this parameter acts as a Boolean flag that signals whether the nonlinear constraint represented by the
        linearized constraint has been violated
    """
    for constr in target_model.MindtPy_utils.constraint_list:

        if constr.body.polynomial_degree() in (0, 1):
            continue

        constr_vars = list(identify_variables(constr.body))
        jacs = solve_data.jacobians

        if constr.has_lb() and constr.has_ub():
            config.logger.warning(
                'constraint {} has both a lower '
                'and upper bound.'
                '\n'.format(
                    constr))
            continue
        if constr.has_ub():
            try:
                upper_slack = constr.uslack()
            except (ValueError, OverflowError):
                config.logger.warning(
                    'constraint {} has caused either a '
                    'ValueError or OverflowError.'
                    '\n'.format(
                        constr))
                continue
            if (linearize_active and abs(upper_slack) < config.ecp_tolerance) \
                    or (linearize_violated and upper_slack < 0) \
                    or (config.linearize_inactive and upper_slack > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.ecp_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars)
                          - (slack_var if config.add_slack else 0)
                          <= upper_slack)
                )

        if constr.has_lb():
            try:
                lower_slack = constr.lslack()
            except (ValueError, OverflowError):
                config.logger.warning(
                    'constraint {} has caused either a '
                    'ValueError or OverflowError.'
                    '\n'.format(
                        constr))
                continue
            if (linearize_active and abs(lower_slack) < config.ecp_tolerance) \
                    or (linearize_violated and lower_slack < 0) \
                    or (config.linearize_inactive and lower_slack > 0):
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                target_model.MindtPy_utils.MindtPy_linear_cuts.ecp_cuts.add(
                    expr=(sum(value(jacs[constr][var])*(var - var.value)
                              for var in constr_vars)
                          + (slack_var if config.add_slack else 0)
                          >= -lower_slack)
                )

# def add_oa_equality_relaxation(var_values, duals, solve_data, config, ignore_integrality=False):
#     """More general case for outer approximation

#     This method covers nonlinear inequalities g(x)<=b and g(x)>=b as well as
#     equalities g(x)=b all in the same linearization call. It combines the dual
#     with the objective sense to figure out how to generate the cut.
#     Note that the dual sign is defined as follows (according to IPOPT):
#       sgn  | min | max
#     -------|-----|-----
#     g(x)<=b|  +1 | -1
#     g(x)>=b|  -1 | +1

#     Note additionally that the dual value is not strictly neccesary for inequality
#     constraints, but definitely neccesary for equality constraints. For equality
#     constraints the cut will always be generated so that the side with the worse objective
#     function is the 'interior'.

#     ignore_integrality: Accepts float values for discrete variables.
#                         Useful for cut in initial relaxation
#     """

#     m = solve_data.mip
#     MindtPy = m.MindtPy_utils
#     MindtPy.MindtPy_linear_cuts.nlp_iters.add(solve_data.nlp_iter)
#     sign_adjust = -1 if solve_data.objective_sense == minimize else 1

#     copy_var_list_values(from_list=var_values,
#                          to_list=MindtPy.variable_list,
#                          config=config,
#                          ignore_integrality=ignore_integrality)

#     # generate new constraints
#     # TODO some kind of special handling if the dual is phenomenally small?
#     # TODO-romeo conditional for 'global' option, i.e. slack or no slack
#     jacs = solve_data.jacobians
#     for constr, dual_value in zip(MindtPy.constraint_list, duals):
#         if constr.body.polynomial_degree() in (1, 0):
#             continue
#         rhs = ((0 if constr.upper is None else constr.upper)
#                + (0 if constr.lower is None else constr.lower))
#         # Properly handle equality constraints and ranged inequalities
#         # TODO special handling for ranged inequalities? a <= x <= b
#         rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
#         slack_var = MindtPy.MindtPy_linear_cuts.slack_vars.add()
#         MindtPy.MindtPy_linear_cuts.oa_cuts.add(
#             expr=copysign(1, sign_adjust * dual_value)
#             * (sum(value(jacs[constr][var]) * (var - value(var))
#                    for var in list(EXPR.identify_variables(constr.body)))
#                + value(constr.body) - rhs)
#             - slack_var <= 0)


def add_nogood_cuts(var_values, solve_data, config, feasible=False):
    """
    Adds integer cuts; modifies the model to include integer cuts

    Parameters
    ----------
    var_values: list
        values of the current variables, used to generate the cut
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    feasible: bool, optional
        boolean indicating if integer combination yields a feasible or infeasible NLP
    """
    if not config.add_nogood_cuts:
        return

    config.logger.info("Adding nogood cuts")

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

    if not binary_vars:  # if no binary variables, skip
        return

    int_cut = (sum(1 - v for v in binary_vars
                   if value(abs(v - 1)) <= int_tol) +
               sum(v for v in binary_vars
                   if value(abs(v)) <= int_tol) >= 1)

    MindtPy.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)


def add_affine_cuts(solve_data, config):
    """
    Adds affine cuts using MCPP; modifies the model to include affine cuts

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """

    m = solve_data.mip
    config.logger.info("Adding affine cuts")
    counter = 0

    for constr in m.MindtPy_utils.constraint_list:
        if constr.body.polynomial_degree() in (1, 0):
            continue

        vars_in_constr = list(
            identify_variables(constr.body))
        if any(var.value is None for var in vars_in_constr):
            continue  # a variable has no values

        # mcpp stuff
        try:
            mc_eqn = mc(constr.body)
        except MCPP_Error as e:
            config.logger.debug(
                "Skipping constraint %s due to MCPP error %s" % (constr.name, str(e)))
            continue  # skip to the next constraint

        ccSlope = mc_eqn.subcc()
        cvSlope = mc_eqn.subcv()
        ccStart = mc_eqn.concave()
        cvStart = mc_eqn.convex()

        # check if the value of ccSlope and cvSlope is not Nan or inf. If so, we skip this.
        concave_cut_valid = True
        convex_cut_valid = True
        for var in vars_in_constr:
            if not var.fixed:
                if ccSlope[var] == float('nan') or ccSlope[var] == float('inf'):
                    concave_cut_valid = False
                if cvSlope[var] == float('nan') or cvSlope[var] == float('inf'):
                    convex_cut_valid = False
        # check if the value of ccSlope and cvSlope all equals zero. if so, we skip this.
        if not any(list(ccSlope.values())):
            concave_cut_valid = False
        if not any(list(cvSlope.values())):
            convex_cut_valid = False
        if ccStart == float('nan') or ccStart == float('inf'):
            concave_cut_valid = False
        if cvStart == float('nan') or cvStart == float('inf'):
            convex_cut_valid = False
        if (concave_cut_valid or convex_cut_valid) is False:
            continue

        ub_int = min(constr.upper, mc_eqn.upper()
                     ) if constr.has_ub() else mc_eqn.upper()
        lb_int = max(constr.lower, mc_eqn.lower()
                     ) if constr.has_lb() else mc_eqn.lower()

        parent_block = constr.parent_block()
        # Create a block on which to put outer approximation cuts.
        # TODO: create it at the beginning.
        aff_utils = parent_block.component('MindtPy_aff')
        if aff_utils is None:
            aff_utils = parent_block.MindtPy_aff = Block(
                doc="Block holding affine constraints")
            aff_utils.MindtPy_aff_cons = ConstraintList()
        aff_cuts = aff_utils.MindtPy_aff_cons
        if concave_cut_valid:
            concave_cut = sum(ccSlope[var] * (var - var.value)
                              for var in vars_in_constr
                              if not var.fixed) + ccStart >= lb_int
            aff_cuts.add(expr=concave_cut)
            counter += 1
        if convex_cut_valid:
            convex_cut = sum(cvSlope[var] * (var - var.value)
                             for var in vars_in_constr
                             if not var.fixed) + cvStart <= ub_int
            aff_cuts.add(expr=convex_cut)
            counter += 1

    config.logger.info("Added %s affine cuts" % counter)
