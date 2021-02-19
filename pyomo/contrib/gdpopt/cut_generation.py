#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""This module provides functions for cut generation."""
from __future__ import division

from collections import namedtuple
from math import copysign, fabs
from six import iteritems
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.gdp_bounds.info import disjunctive_bounds
from pyomo.contrib.gdpopt.util import time_code, constraints_in_True_disjuncts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.core import (Block, ConstraintList, NonNegativeReals, VarList,
                        minimize, value, TransformationFactory)
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables

MAX_SYMBOLIC_DERIV_SIZE = 1000
JacInfo = namedtuple('JacInfo', ['mode','vars','jac'])

def add_subproblem_cuts(subprob_result, solve_data, config):
    if config.strategy == "LOA":
        return add_outer_approximation_cuts(subprob_result, solve_data, config)
    elif config.strategy == "GLOA":
        return add_affine_cuts(subprob_result, solve_data, config)


def add_outer_approximation_cuts(nlp_result, solve_data, config):
    """Add outer approximation cuts to the linear GDP model."""
    with time_code(solve_data.timing, 'OA cut generation'):
        m = solve_data.linear_GDP
        GDPopt = m.GDPopt_utils
        sign_adjust = -1 if solve_data.objective_sense == minimize else 1

        # copy values over
        for var, val in zip(GDPopt.variable_list, nlp_result.var_values):
            if val is not None and not var.fixed:
                var.value = val

        # TODO some kind of special handling if the dual is phenomenally small?
        config.logger.debug('Adding OA cuts.')

        counter = 0
        if not hasattr(GDPopt, 'jacobians'):
            GDPopt.jacobians = ComponentMap()
        for constr, dual_value in zip(GDPopt.constraint_list,
                                      nlp_result.dual_values):
            if dual_value is None or constr.body.polynomial_degree() in (1, 0):
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

            # Cache jacobian
            jacobian = GDPopt.jacobians.get(constr, None)
            if jacobian is None:
                constr_vars = list(identify_variables(
                    constr.body, include_fixed=False))
                if len(constr_vars) >= MAX_SYMBOLIC_DERIV_SIZE:
                    mode = differentiate.Modes.reverse_numeric
                else:
                    mode = differentiate.Modes.sympy

                try:
                    jac_list = differentiate(
                        constr.body, wrt_list=constr_vars, mode=mode)
                    jac_map = ComponentMap(zip(constr_vars, jac_list))
                except:
                    if mode is differentiate.Modes.reverse_numeric:
                        raise
                    mode = differentiate.Modes.reverse_numeric
                    jac_map = ComponentMap()
                jacobian = JacInfo(mode=mode, vars=constr_vars, jac=jac_map)
                GDPopt.jacobians[constr] = jacobian
            # Recompute numeric derivatives
            if not jacobian.jac:
                jac_list = differentiate(
                    constr.body, wrt_list=jacobian.vars, mode=jacobian.mode)
                jacobian.jac.update(zip(jacobian.vars, jac_list))

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
            rhs = value(constr.lower) if constr.has_lb() else value(constr.upper)
            try:
                new_oa_cut = (
                    copysign(1, sign_adjust * dual_value) * (
                        value(constr.body) - rhs + sum(
                            value(jac) * (var - value(var))
                            for var, jac in iteritems(jacobian.jac))
                        ) - slack_var <= 0)
                if new_oa_cut.polynomial_degree() not in (1, 0):
                    for var, jac in iteritems(jacobian.jac):
                        print(var.name, value(jac))
                oa_cuts.add(expr=new_oa_cut)
                counter += 1
            except ZeroDivisionError:
                config.logger.warning(
                    "Zero division occured attempting to generate OA cut for constraint %s.\n"
                    "Skipping OA cut generation for this constraint."
                    % (constr.name,)
                )
                # Simply continue on to the next constraint.
            # Clear out the numeric Jacobian values
            if jacobian.mode is differentiate.Modes.reverse_numeric:
                jacobian.jac.clear()

        config.logger.info('Added %s OA cuts' % counter)


def add_affine_cuts(nlp_result, solve_data, config):
    with time_code(solve_data.timing, "affine cut generation"):
        m = solve_data.linear_GDP
        if config.calc_disjunctive_bounds:
            with time_code(solve_data.timing, "disjunctive variable bounding"):
                TransformationFactory('contrib.compute_disj_var_bounds').apply_to(
                    m,
                    solver=config.mip_solver if config.obbt_disjunctive_bounds else None
                )
        config.logger.info("Adding affine cuts.")
        GDPopt = m.GDPopt_utils
        counter = 0
        for var, val in zip(GDPopt.variable_list, nlp_result.var_values):
            if val is not None and not var.fixed:
                var.value = val

        for constr in constraints_in_True_disjuncts(m, config):
            # Note: this includes constraints that are deactivated in the current model (linear_GDP)

            disjunctive_var_bounds = disjunctive_bounds(constr.parent_block())

            if constr.body.polynomial_degree() in (1, 0):
                continue

            vars_in_constr = list(
                identify_variables(constr.body))
            if any(var.value is None for var in vars_in_constr):
                continue  # a variable has no values

            # mcpp stuff
            try:
                mc_eqn = mc(constr.body, disjunctive_var_bounds)
            except MCPP_Error as e:
                config.logger.debug("Skipping constraint %s due to MCPP error %s" % (constr.name, str(e)))
                continue  # skip to the next constraint
            ccSlope = mc_eqn.subcc()
            cvSlope = mc_eqn.subcv()
            ccStart = mc_eqn.concave()
            cvStart = mc_eqn.convex()
            ub_int = min(constr.upper, mc_eqn.upper()) if constr.has_ub() else mc_eqn.upper()
            lb_int = max(constr.lower, mc_eqn.lower()) if constr.has_lb() else mc_eqn.lower()

            parent_block = constr.parent_block()
            # Create a block on which to put outer approximation cuts.
            aff_utils = parent_block.component('GDPopt_aff')
            if aff_utils is None:
                aff_utils = parent_block.GDPopt_aff = Block(
                    doc="Block holding affine constraints")
                aff_utils.GDPopt_aff_cons = ConstraintList()
            aff_cuts = aff_utils.GDPopt_aff_cons
            concave_cut = sum(ccSlope[var] * (var - var.value)
                              for var in vars_in_constr
                              if not var.fixed) + ccStart >= lb_int
            convex_cut = sum(cvSlope[var] * (var - var.value)
                             for var in vars_in_constr
                             if not var.fixed) + cvStart <= ub_int
            aff_cuts.add(expr=concave_cut)
            aff_cuts.add(expr=convex_cut)
            counter += 2

        config.logger.info("Added %s affine cuts" % counter)


def add_integer_cut(var_values, target_model, solve_data, config, feasible=False):
    """Add an integer cut to the target GDP model."""
    with time_code(solve_data.timing, 'integer cut generation'):
        m = target_model
        GDPopt = m.GDPopt_utils
        var_value_is_one = ComponentSet()
        var_value_is_zero = ComponentSet()
        indicator_vars = ComponentSet(disj.indicator_var for disj in GDPopt.disjunct_list)
        for var, val in zip(GDPopt.variable_list, var_values):
            if not var.is_binary():
                continue
            if var.fixed:
                # if val is not None and var.value != val:
                #     # val needs to be None or match var.value. Otherwise, we have a
                #     # contradiction.
                #     raise ValueError(
                #         "Fixed variable %s has value %s != "
                #         "provided value of %s." % (var.name, var.value, val))

                # Note: FBBT may cause some disjuncts to be fathomed, which can cause
                # a fixed variable to be different than the subproblem value.
                # In this case, we simply construct the integer cut as usual with
                # the subproblem value rather than its fixed value.
                if val is None:
                    val = var.value

            if not config.force_subproblem_nlp:
                # By default (config.force_subproblem_nlp = False), we only want
                # the integer cuts to be over disjunct indicator vars.
                if var not in indicator_vars:
                    continue

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
                'Adding integer cut to a model without discrete variables. '
                'Model is now infeasible.')
            if solve_data.objective_sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
            return False

        int_cut = (sum(1 - v for v in var_value_is_one) +
                   sum(v for v in var_value_is_zero)) >= 1

        # Exclude the current binary combination
        config.logger.info('Adding integer cut')
        GDPopt.integer_cuts.add(expr=int_cut)

    if config.calc_disjunctive_bounds:
        with time_code(solve_data.timing, "disjunctive variable bounding"):
            TransformationFactory('contrib.compute_disj_var_bounds').apply_to(
                m,
                solver=config.mip_solver if config.obbt_disjunctive_bounds else None
            )
