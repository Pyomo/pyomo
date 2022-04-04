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
from collections import namedtuple
from math import copysign, fabs
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdp_bounds.info import disjunctive_bounds
from pyomo.contrib.gdpopt.util import (
    constraints_in_True_disjuncts, _add_bigm_constraint_to_transformed_model)
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.core import (Block, NonNegativeReals, VarList, minimize, value,
                        Constraint, NonNegativeIntegers)
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn import generate_standard_repn

MAX_SYMBOLIC_DERIV_SIZE = 1000
JacInfo = namedtuple('JacInfo', ['mode','vars','jac'])

def add_cuts_according_to_algorithm(subproblem_util_block, master_util_block,
                                    objective_sense, config, timing):
    if config.algorithm == "LOA":
        return add_outer_approximation_cuts(subproblem_util_block,
                                            master_util_block, objective_sense,
                                            config)
    elif config.algorithm == "GLOA":
        return add_affine_cuts(subproblem_util_block, master_util_block, config,
                               timing)
    elif config.algorithm == "RIC":
        pass
    else:
        raise ValueError('Unrecognized strategy: ' + config.strategy)

def add_outer_approximation_cuts(subproblem_util_block, master_util_block,
                                 objective_sense, config):
    """Add outer approximation cuts to the linear GDP model."""
    m = master_util_block.model()
    nlp = subproblem_util_block.model()
    sign_adjust = -1 if objective_sense == minimize else 1
    # Dictionary mapping blocks to their child blocks we use to store OA
    # cuts. We do it this way because we don't know for sure we can have any
    # given name since we are sticking these on a clone of a user-generated
    # model. But this keeps track that we find one if we've already created it,
    # so that we can add the cuts as indexed constraints.
    if hasattr(master_util_block, 'oa_cut_blocks'):
        oa_cut_blocks = master_util_block.oa_cut_blocks
    else:
        oa_cut_blocks = master_util_block.oa_cut_blocks = dict()

    for master_var, subprob_var in zip(
            subproblem_util_block.algebraic_variable_list,
            master_util_block.algebraic_variable_list):
        val = subprob_var.value
        if val is not None and not master_var.fixed:
            master_var.set_value(val, skip_validation=True)

    config.logger.debug('Adding OA cuts.')

    counter = 0
    if not hasattr(master_util_block, 'jacobians'):
        master_util_block.jacobians = ComponentMap()
    for constr, subprob_constr in zip(master_util_block.constraint_list,
                                      subproblem_util_block.constraint_list):
        dual_value = nlp.dual.get(subprob_constr, None)
        if (dual_value is None or
            generate_standard_repn(constr.body).is_linear()):
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

        config.logger.debug( "Adding OA cut for %s with dual value %s" %
                             (constr.name, dual_value))

        # Cache jacobian
        jacobian = master_util_block.jacobians.get(constr, None)
        if jacobian is None:
            constr_vars = list(identify_variables(constr.body,
                                                  include_fixed=False))
            if len(constr_vars) >= MAX_SYMBOLIC_DERIV_SIZE:
                mode = differentiate.Modes.reverse_numeric
            else:
                mode = differentiate.Modes.sympy

            try:
                jac_list = differentiate( constr.body, wrt_list=constr_vars,
                                          mode=mode)
                jac_map = ComponentMap(zip(constr_vars, jac_list))
            except:
                if mode is differentiate.Modes.reverse_numeric:
                    raise
                mode = differentiate.Modes.reverse_numeric
                jac_map = ComponentMap()
            jacobian = JacInfo(mode=mode, vars=constr_vars, jac=jac_map)
            master_util_block.jacobians[constr] = jacobian
        # Recompute numeric derivatives
        if not jacobian.jac:
            jac_list = differentiate( constr.body, wrt_list=jacobian.vars,
                                      mode=jacobian.mode)
            jacobian.jac.update(zip(jacobian.vars, jac_list))

        # Create a block on which to put outer approximation cuts, if we don't
        # have one already on this parent block.
        oa_utils = oa_cut_blocks.get(parent_block)
        if oa_utils is None:
            nm = unique_component_name(parent_block, 'GDPopt_OA_cuts')
            oa_utils = Block(doc="Block holding outer approximation cuts "
                             "and associated data.")
            parent_block.add_component(nm, oa_utils)
            oa_cut_blocks[parent_block] = oa_utils
            oa_utils.cuts = Constraint(NonNegativeIntegers)
        master_oa_utils = master_util_block.component('GDPopt_OA_slacks')
        if master_oa_utils is None:
            master_oa_utils = master_util_block.GDPopt_OA_slacks = Block(
                doc="Block holding outer approximation slacks for the "
                "whole model (so that the writers can find them).")
            master_oa_utils.slacks = VarList( bounds=(0, config.max_slack),
                                              domain=NonNegativeReals,
                                              initialize=0)

        oa_cuts = oa_utils.cuts
        slack_var = master_oa_utils.slacks.add()
        rhs = value(constr.lower) if constr.has_lb() else value(
            constr.upper)
        try:
            new_oa_cut = (
                copysign(1, sign_adjust * dual_value) * (
                    value(constr.body) - rhs + sum(
                        value(jac) * (var - value(var))
                        for var, jac in jacobian.jac.items())
                    ) - slack_var <= 0)
            assert new_oa_cut.polynomial_degree() in (1, 0)
            idx = len(oa_cuts)
            oa_cuts[idx] = new_oa_cut
            _add_bigm_constraint_to_transformed_model(m, oa_cuts[idx], oa_cuts)
            config.logger.debug("Cut expression: %s" % new_oa_cut)
            counter += 1
        except ZeroDivisionError:
            config.logger.warning(
                "Zero division occured attempting to generate OA cut for "
                "constraint %s.\n"
                "Skipping OA cut generation for this constraint."
                % (constr.name,)
            )
            # Simply continue on to the next constraint.
        # Clear out the numeric Jacobian values
        if jacobian.mode is differentiate.Modes.reverse_numeric:
            jacobian.jac.clear()

    config.logger.debug('Added %s OA cuts' % counter)

def add_affine_cuts(subproblem_util_block, master_util_block, config, timing):
    m = master_util_block.model()

    config.logger.debug("Adding affine cuts.")
    counter = 0
    for master_var, subprob_var in zip(
            master_util_block.algebraic_variable_list,
            subproblem_util_block.algebraic_variable_list):
        val = subprob_var.value
        if val is not None and not master_var.fixed:
            master_var.set_value(val, skip_validation=True)

    for constr in constraints_in_True_disjuncts(m, config):
        # Note: this includes constraints that are deactivated in the
        # current model (linear_GDP)

        disjunctive_var_bounds = disjunctive_bounds(constr.parent_block())

        if constr.body.polynomial_degree() in (1, 0):
            continue

        vars_in_constr = list(identify_variables(constr.body))
        if any(var.value is None for var in vars_in_constr):
            continue  # a variable has no values

        # mcpp stuff
        try:
            mc_eqn = mc(constr.body, disjunctive_var_bounds)
        except MCPP_Error as e:
            config.logger.debug("Skipping constraint %s due to MCPP "
                                "error %s" % (constr.name, str(e)))
            continue  # skip to the next constraint
        ccSlope = mc_eqn.subcc()
        cvSlope = mc_eqn.subcv()
        ccStart = mc_eqn.concave()
        cvStart = mc_eqn.convex()
        ub_int = min(constr.upper, mc_eqn.upper()) if constr.has_ub() \
                 else mc_eqn.upper()
        lb_int = max(constr.lower, mc_eqn.lower()) if constr.has_lb() \
                 else mc_eqn.lower()

        parent_block = constr.parent_block()
        # Create a block on which to put outer approximation cuts.
        aff_utils = parent_block.component('GDPopt_aff')
        if aff_utils is None:
            aff_utils = parent_block.GDPopt_aff = Block(
                doc="Block holding affine constraints")
            aff_utils.GDPopt_aff_cons = Constraint(NonNegativeIntegers)
        aff_cuts = aff_utils.GDPopt_aff_cons
        concave_cut = sum(ccSlope[var] * (var - var.value)
                          for var in vars_in_constr
                          if not var.fixed) + ccStart >= lb_int
        convex_cut = sum(cvSlope[var] * (var - var.value)
                         for var in vars_in_constr
                         if not var.fixed) + cvStart <= ub_int
        idx = len(aff_cuts)
        aff_cuts[idx] = concave_cut
        aff_cuts[idx+1] = convex_cut
        _add_bigm_constraint_to_transformed_model(m, aff_cuts[idx], aff_cuts)
        _add_bigm_constraint_to_transformed_model(m, aff_cuts[idx+1], aff_cuts)
        counter += 2

    config.logger.debug("Added %s affine cuts" % counter)


def add_no_good_cut(target_model_util_block, config):
    """Cut the current integer solution from the target model."""
    var_value_is_one = ComponentSet()
    var_value_is_zero = ComponentSet()
    for var in target_model_util_block.transformed_boolean_variable_list:
        val = value(var)
        if var.fixed:
            # ESJ TODO: I don't understand this
            # Note: FBBT may cause some disjuncts to be fathomed, which can
            # cause a fixed variable to be different than the subproblem value.
            # In this case, we simply construct the integer cut as usual with
            # the subproblem value rather than its fixed value.
            if val is None:
                val = var.value

        # TODO: If this is true, we actually need to iterate through more stuff.
        # if not config.force_subproblem_nlp:
        #     # By default (config.force_subproblem_nlp = False), we only want
        #     # the integer cuts to be over disjunct indicator vars.
        #     # ESJ: Is the above true? Or should it be over all BooleanVars??
        #     if var not in indicator_vars:
        #         continue

        if fabs(val - 1) <= config.integer_tolerance:
            var_value_is_one.add(var)
        elif fabs(val) <= config.integer_tolerance:
            var_value_is_zero.add(var)
        else:
            raise ValueError(
                'Binary %s = %s is not 0 or 1' % (var.name, val))

    if not (var_value_is_one or var_value_is_zero):
        # if no remaining binary variables, then terminate algorithm.
        config.logger.debug(
            'No remaining discrete solutions to explore.')
        return False

    int_cut = (sum(1 - v for v in var_value_is_one) +
               sum(v for v in var_value_is_zero)) >= 1

    # Exclude the current binary combination
    config.logger.debug('Adding no-good cut: %s' % int_cut)
    target_model_util_block.no_good_cuts.add(expr=int_cut)

    return True
