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
from math import copysign
from pyomo.core import minimize, value
from pyomo.core.expr import current as EXPR
from pyomo.contrib.gdpopt.util import identify_variables, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error


def add_oa_cuts(target_model, dual_values, solve_data, config,
                cb_opt=None,
                linearize_active=True,
                linearize_violated=True):
    """Adds OA cuts.

    Generates and adds OA cuts (linearizes nonlinear constraints).
    For nonconvex problems, turn on 'config.add_slack'. 
    Slack variables will always be used for nonlinear equality constraints.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    dual_values : list
        The value of the duals for each constraint.
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    cb_opt : SolverFactory, optional
        Gurobi_persistent solver, by default None.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(solve_data.timing, 'OA cut generation'):
        for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
            # TODO: here the index is correlated to the duals, try if this can be fixed when temp duals are removed.
            if constr.body.polynomial_degree() in {0, 1}:
                continue

            constr_vars = list(identify_variables(constr.body))
            jacs = solve_data.jacobians

            # Equality constraint (makes the problem nonconvex)
            if constr.has_ub() and constr.has_lb() and value(constr.lower) == value(constr.upper) and config.equality_relaxation:
                sign_adjust = -1 if solve_data.objective_sense == minimize else 1
                rhs = constr.lower
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                target_model.MindtPy_utils.cuts.oa_cuts.add(
                    expr=copysign(1, sign_adjust * dual_values[index])
                    * (sum(value(jacs[constr][var]) * (var - value(var))
                           for var in EXPR.identify_variables(constr.body))
                        + value(constr.body) - rhs)
                    - (slack_var if config.add_slack else 0) <= 0)
                if config.single_tree and config.mip_solver == 'gurobi_persistent' and solve_data.mip_iter > 0 and cb_opt is not None:
                    cb_opt.cbLazy(
                        target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])

            else:  # Inequality constraint (possibly two-sided)
                if (constr.has_ub()
                    and (linearize_active and abs(constr.uslack()) < config.zero_tolerance)
                        or (linearize_violated and constr.uslack() < 0)
                        or (config.linearize_inactive and constr.uslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_ub()):
                    # always add the linearization for the epigraph of the objective
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.oa_cuts.add(
                        expr=(sum(value(jacs[constr][var])*(var - var.value)
                                  for var in constr_vars) + value(constr.body)
                              - (slack_var if config.add_slack else 0)
                              <= value(constr.upper))
                    )
                    if config.single_tree and config.mip_solver == 'gurobi_persistent' and solve_data.mip_iter > 0 and cb_opt is not None:
                        cb_opt.cbLazy(
                            target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])

                if (constr.has_lb()
                    and (linearize_active and abs(constr.lslack()) < config.zero_tolerance)
                        or (linearize_violated and constr.lslack() < 0)
                        or (config.linearize_inactive and constr.lslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_lb()):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.oa_cuts.add(
                        expr=(sum(value(jacs[constr][var])*(var - var.value)
                                  for var in constr_vars) + value(constr.body)
                              + (slack_var if config.add_slack else 0)
                              >= value(constr.lower))
                    )
                    if config.single_tree and config.mip_solver == 'gurobi_persistent' and solve_data.mip_iter > 0 and cb_opt is not None:
                        cb_opt.cbLazy(
                            target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])


def add_ecp_cuts(target_model, solve_data, config,
                 linearize_active=True,
                 linearize_violated=True):
    """Linearizes nonlinear constraints. Adds the cuts for the ECP method.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(solve_data.timing, 'ECP cut generation'):
        for constr in target_model.MindtPy_utils.nonlinear_constraint_list:
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
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.ecp_cuts.add(
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
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.ecp_cuts.add(
                        expr=(sum(value(jacs[constr][var])*(var - var.value)
                                  for var in constr_vars)
                              + (slack_var if config.add_slack else 0)
                              >= -lower_slack)
                    )


def add_no_good_cuts(var_values, solve_data, config):
    """Adds no-good cuts.

    This adds an no-good cuts to the no_good_cuts ConstraintList, which is not activated by default.
    However, it may be activated as needed in certain situations or for certain values of option flags.


    Parameters
    ----------
    var_values : list
        Variable values of the current solution, used to generate the cut.
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Raises
    ------
    ValueError
        The value of binary variable is not 0 or 1.
    """
    if not config.add_no_good_cuts:
        return
    with time_code(solve_data.timing, 'no_good cut generation'):

        config.logger.debug('Adding no-good cuts')

        m = solve_data.mip
        MindtPy = m.MindtPy_utils
        int_tol = config.integer_tolerance

        binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]

        # copy variable values over
        for var, val in zip(MindtPy.variable_list, var_values):
            if not var.is_binary():
                continue
            var.set_value(val, skip_validation=True)

        # check to make sure that binary variables are all 0 or 1
        for v in binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError(
                    'Binary {} = {} is not 0 or 1'.format(v.name, value(v)))

        if not binary_vars:  # if no binary variables, skip
            return

        int_cut = (sum(1 - v for v in binary_vars
                       if value(abs(v - 1)) <= int_tol) +
                   sum(v for v in binary_vars
                       if value(abs(v)) <= int_tol) >= 1)

        MindtPy.cuts.no_good_cuts.add(expr=int_cut)


def add_affine_cuts(solve_data, config):
    """Adds affine cuts using MCPP.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    with time_code(solve_data.timing, 'Affine cut generation'):
        m = solve_data.mip
        config.logger.debug('Adding affine cuts')
        counter = 0

        for constr in m.MindtPy_utils.nonlinear_constraint_list:
            vars_in_constr = list(
                identify_variables(constr.body))
            if any(var.value is None for var in vars_in_constr):
                continue  # a variable has no values

            # mcpp stuff
            try:
                mc_eqn = mc(constr.body)
            except MCPP_Error as e:
                config.logger.debug(
                    'Skipping constraint %s due to MCPP error %s' % (constr.name, str(e)))
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
            if not (concave_cut_valid or convex_cut_valid):
                continue

            ub_int = min(value(constr.upper), mc_eqn.upper()
                         ) if constr.has_ub() else mc_eqn.upper()
            lb_int = max(value(constr.lower), mc_eqn.lower()
                         ) if constr.has_lb() else mc_eqn.lower()

            aff_cuts = m.MindtPy_utils.cuts.aff_cuts
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

        config.logger.debug('Added %s affine cuts' % counter)
