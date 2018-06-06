"""This module provides functions for cut generation."""
from __future__ import division

from math import copysign, fabs

from pyomo.core import (Block, ConstraintList, NonNegativeReals, VarList,
                        minimize, value)
from pyomo.core.base.symbolic import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentMap, ComponentSet


def add_outer_approximation_cuts(var_values, duals, solve_data, config):
    """Add outer approximation cuts to the linear GDP model."""
    m = solve_data.linear_GDP
    GDPopt = m.GDPopt_utils
    sign_adjust = -1 if GDPopt.objective.sense == minimize else 1

    # copy values over
    for var, val in zip(GDPopt.initial_var_list, var_values):
        if val is not None and not var.fixed:
            var.value = val

    # TODO some kind of special handling if the dual is phenomenally small?
    config.logger.debug('Adding OA cuts.')

    nonlinear_constraints = ComponentSet(GDPopt.initial_nonlinear_constraints)
    counter = 0
    for constr, dual_value in zip(GDPopt.initial_constraints_list, duals):
        if dual_value is None or constr not in nonlinear_constraints:
            continue

        # Determine if the user pre-specified that OA cuts should not be
        # generated for the given constraint.
        parent_block = constr.parent_block()
        ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
        config.logger.debug('Ignore_set %s' % ignore_set)
        if (ignore_set and (constr in ignore_set or
                            constr.parent_component() in ignore_set)):
            config.logger.debug(
                'OA cut addition for %s skipped because it is in '
                'the ignore set.' % constr.name)
            continue

        config.logger.debug(
            "Adding OA cut for %s with dual value %s"
            % (constr.name, dual_value))

        # TODO make this more efficient by not having to use differentiate()
        # at each iteration.
        constr_vars = list(EXPR.identify_variables(constr.body))
        jac_list = differentiate(constr.body, wrt_list=constr_vars)
        jacobians = ComponentMap(zip(constr_vars, jac_list))

        # Create a block on which to put outer approximation cuts.
        oa_utils = parent_block.component('GDPopt_OA')
        if oa_utils is None:
            oa_utils = parent_block.GDPopt_OA = Block(
                doc="Block holding outer approximation cuts "
                "and associated data.")
            oa_utils.GDPopt_OA_cuts = ConstraintList()
            oa_utils.GDPopt_OA_slacks = VarList(
                bounds=(0, config.max_slack),
                domain=NonNegativeReals, initialize=0)

        oa_cuts = oa_utils.GDPopt_OA_cuts
        slack_var = oa_utils.GDPopt_OA_slacks.add()
        oa_cuts.add(
            expr=copysign(1, sign_adjust * dual_value) * (
                value(constr.body) + sum(
                    value(jacobians[var]) * (var - value(var))
                    for var in constr_vars)) + slack_var <= 0)
        counter += 1

    config.logger.info('Added %s OA cuts' % counter)


def add_integer_cut(var_values, solve_data, config, feasible=False):
    """Add an integer cut to the linear GDP model."""
    m = solve_data.linear_GDP
    GDPopt = m.GDPopt_utils
    var_value_is_one = ComponentSet()
    var_value_is_zero = ComponentSet()
    for var, val in zip(GDPopt.initial_var_list, var_values):
        if not var.is_binary():
            continue
        if var.fixed:
            if val is not None and var.value != val:
                # val needs to be None or match var.value. Otherwise, we have a
                # contradiction.
                raise ValueError(
                    "Fixed variable %s has value %s != "
                    "provided value of %s." % (var.name, var.value, val))
            val = var.value
        # TODO we can also add a check to skip binary variables that are not an
        # indicator_var on disjuncts.
        if fabs(val - 1) <= config.integer_tolerance:
            var_value_is_one.add(var)
        elif fabs(val) <= config.integer_tolerance:
            var_value_is_zero.add(var)
        else:
            raise ValueError(
                'Binary %s = %s is not 0 or 1' % (var.name, val))

    if not (var_value_is_one or var_value_is_zero):
        # if no remaining binary variables, then terminate algorithm.
        config.logger.info(
            'Adding integer cut to a model without binary variables. '
            'Model is now infeasible.')
        if solve_data.objective_sense == minimize:
            solve_data.LB = float('inf')
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = float('-inf')
            solve_data.UB_progress.append(solve_data.UB)
        return False

    int_cut = (sum(1 - v for v in var_value_is_one) +
               sum(v for v in var_value_is_zero)) >= 1

    if not feasible:
        config.logger.info('Adding integer cut')
        GDPopt.integer_cuts.add(expr=int_cut)
    else:
        backtracking_enabled = (
            "disabled" if GDPopt.no_backtracking.active else "allowed")
        config.logger.info(
            'Registering explored configuration. '
            'Backtracking is currently %s.' % backtracking_enabled)
        GDPopt.no_backtracking.add(expr=int_cut)
