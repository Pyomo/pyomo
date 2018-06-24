"""Functions for initializing the master problem
in Logic-based outer approximation.
"""

from __future__ import division

from math import fabs

from pyomo.contrib.gdpopt.cut_generation import (add_integer_cut,
                                                 add_outer_approximation_cuts)
from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.nlp_solve import (solve_NLP,
                                            update_nlp_progress_indicators)
from pyomo.contrib.gdpopt.util import copy_and_fix_mip_values_to_nlp
from pyomo.core import (Block, Constraint, Objective, TransformationFactory,
                        Var, maximize, minimize)
from pyomo.gdp import Disjunct


def init_custom_disjuncts(solve_data, config):
    """Initialize by using user-specified custom disjuncts."""
    # TODO error checking to make sure that the user gave proper disjuncts
    for active_disjunct_set in config.custom_init_disjuncts:
        # custom_init_disjuncts contains a list of sets, giving the disjuncts
        # active at each initialization iteration

        # fix the disjuncts in the linear GDP and send for solution.
        solve_data.mip_iteration += 1
        linear_GDP = solve_data.linear_GDP.clone()
        config.logger.info(
            "Generating initial linear GDP approximation by "
            "solving subproblems with user-specified active disjuncts.")
        for orig_disj, clone_disj in zip(
            solve_data.original_model.GDPopt_utils.orig_disjuncts_list,
            linear_GDP.GDPopt_utils.orig_disjuncts_list
        ):
            if orig_disj in active_disjunct_set:
                clone_disj.indicator_var.fix(1)
        mip_result = solve_linear_GDP(linear_GDP, solve_data, config)
        if mip_result:
            _, mip_var_values = mip_result
            # use the mip_var_values to create the NLP subproblem
            nlp_model = solve_data.working_model.clone()
            # copy in the discrete variable values
            copy_and_fix_mip_values_to_nlp(
                nlp_model.GDPopt_utils.working_var_list,
                mip_var_values, config)
            TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
            solve_data.nlp_iteration += 1
            nlp_result = solve_NLP(nlp_model, solve_data, config)
            nlp_feasible, nlp_var_values, nlp_duals = nlp_result
            if nlp_feasible:
                update_nlp_progress_indicators(nlp_model, solve_data, config)
                add_outer_approximation_cuts(
                    nlp_var_values, nlp_duals, solve_data, config)
            add_integer_cut(
                mip_var_values, solve_data, config, feasible=nlp_feasible)
        else:
            config.logger.error(
                'Linear GDP infeasible for user-specified '
                'custom initialization disjunct set %s. '
                'Skipping that set and continuing on.'
                % list(disj.name for disj in active_disjunct_set))


def init_fixed_disjuncts(solve_data, config):
    """Initialize by solving the problem with the current disjunct values."""
    # TODO error checking to make sure that the user gave proper disjuncts

    # fix the disjuncts in the linear GDP and send for solution.
    solve_data.mip_iteration += 1
    linear_GDP = solve_data.linear_GDP.clone()
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving subproblem with original user-specified disjunct values.")
    TransformationFactory('gdp.fix_disjuncts').apply_to(linear_GDP)
    mip_result = solve_linear_GDP(linear_GDP, solve_data, config)
    if mip_result:
        _, mip_var_values = mip_result
        # use the mip_var_values to create the NLP subproblem
        nlp_model = solve_data.working_model.clone()
        # copy in the discrete variable values
        copy_and_fix_mip_values_to_nlp(
            nlp_model.GDPopt_utils.working_var_list,
            mip_var_values, config)
        TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
        solve_data.nlp_iteration += 1
        nlp_result = solve_NLP(nlp_model, solve_data, config)
        nlp_feasible, nlp_var_values, nlp_duals = nlp_result
        if nlp_feasible:
            update_nlp_progress_indicators(nlp_model, solve_data, config)
            add_outer_approximation_cuts(
                nlp_var_values, nlp_duals, solve_data, config)
        add_integer_cut(
            mip_var_values, solve_data, config, feasible=nlp_feasible)
    else:
        config.logger.error(
            'Linear GDP infeasible for initial user-specified '
            'disjunct values. '
            'Skipping initialization.')


def init_max_binaries(solve_data, config):
    """Initialize by maximizing binary variables and disjuncts.

    This function activates as many binary variables and disjucts as
    feasible.

    """
    solve_data.mip_iteration += 1
    linear_GDP = solve_data.linear_GDP.clone()
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving a subproblem that maximizes "
        "the sum of all binary and logical variables.")
    # Set up binary maximization objective
    linear_GDP.GDPopt_utils.objective.deactivate()
    binary_vars = (
        v for v in linear_GDP.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v.is_binary() and not v.fixed)
    linear_GDP.GDPopt_utils.max_binary_obj = Objective(
        expr=sum(binary_vars), sense=maximize)

    # Solve
    mip_results = solve_linear_GDP(linear_GDP, solve_data, config)
    if mip_results:
        _, mip_var_values = mip_results
        # use the mip_var_values to create the NLP subproblem
        nlp_model = solve_data.working_model.clone()
        # copy in the discrete variable values
        copy_and_fix_mip_values_to_nlp(nlp_model.GDPopt_utils.working_var_list,
                                       mip_var_values, config)
        TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
        solve_data.nlp_iteration += 1
        nlp_result = solve_NLP(nlp_model, solve_data, config)
        nlp_feasible, nlp_var_values, nlp_duals = nlp_result
        if nlp_feasible:
            update_nlp_progress_indicators(nlp_model, solve_data, config)
            add_outer_approximation_cuts(
                nlp_var_values, nlp_duals, solve_data, config)
        add_integer_cut(
            mip_var_values, solve_data, config, feasible=nlp_feasible)
    else:
        config.logger.info(
            "Linear relaxation for initialization was infeasible. "
            "Problem is infeasible.")
        return False


def init_set_covering(solve_data, config):
    """Initialize by solving problems to cover the set of all disjuncts.

    The purpose of this initialization is to generate linearizations
    corresponding to each of the disjuncts.

    This work is based upon prototyping work done by Eloy Fernandez at
    Carnegie Mellon University.

    """
    config.logger.info(
        "Generating initial linear GDP approximation by solving subproblems "
        "to cover all nonlinear disjuncts.")
    disjunct_needs_cover = list(
        any(constr.body.polynomial_degree() not in (0, 1)
            for constr in disj.component_data_objects(
                ctype=Constraint, active=True, descend_into=True))
        for disj in solve_data.working_model.GDPopt_utils.
        working_disjuncts_list)
    iter_count = 1
    while (any(disjunct_needs_cover) and
           iter_count <= config.set_cover_iterlim):
        solve_data.mip_iteration += 1
        linear_GDP = solve_data.linear_GDP.clone()
        linear_GDP.GDPopt_utils.no_backtracking.activate()
        # Solve set covering MIP
        mip_results = solve_set_cover_MIP(
            linear_GDP, disjunct_needs_cover, solve_data, config)
        if not mip_results:
            # problem is infeasible. break
            return False
        # solve local NLP
        _, mip_var_values, mip_disjunct_values = mip_results
        nlp_model = solve_data.working_model.clone()
        copy_and_fix_mip_values_to_nlp(nlp_model.GDPopt_utils.working_var_list,
                                       mip_var_values, config)
        TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
        solve_data.nlp_iteration += 1
        nlp_result = solve_NLP(nlp_model, solve_data, config)
        nlp_feasible, nlp_var_values, nlp_duals = nlp_result
        if nlp_feasible:
            # if successful, updated sets
            active_disjuncts = list(
                fabs(val - 1) <= config.integer_tolerance
                for val in mip_disjunct_values)
            disjunct_needs_cover = list(
                (needed_cover and not was_active)
                for (needed_cover, was_active) in zip(disjunct_needs_cover,
                                                      active_disjuncts))
            update_nlp_progress_indicators(nlp_model, solve_data, config)
            add_outer_approximation_cuts(
                nlp_var_values, nlp_duals, solve_data, config)
        add_integer_cut(
            mip_var_values, solve_data, config, feasible=nlp_feasible)

        iter_count += 1

    if any(disjunct_needs_cover):
        # Iteration limit was hit without a full covering of all nonlinear
        # disjuncts
        config.logger.warning(
            'Iteration limit reached for set covering initialization '
            'without covering all disjuncts.')
        return False
    return True


def solve_set_cover_MIP(linear_GDP_model, disj_needs_cover,
                        solve_data, config):
    """Solve the set covering MIP to determine next configuration."""
    m = linear_GDP_model
    GDPopt = linear_GDP_model.GDPopt_utils
    # number of disjuncts that still need to be covered
    num_needs_cover = sum(1 for disj_bool in disj_needs_cover if disj_bool)
    # number of disjuncts that have been covered
    num_covered = len(disj_needs_cover) - num_needs_cover
    # weights for the set covering problem
    weights = list((num_covered + 1 if disj_bool else 1)
                   for disj_bool in disj_needs_cover)
    # Set up set covering objective
    GDPopt.objective.deactivate()
    GDPopt.set_cover_obj = Objective(
        expr=sum(weight * disj.indicator_var
                 for (weight, disj) in zip(weights,
                                           GDPopt.working_disjuncts_list)),
        sense=maximize)

    # Deactivate potentially non-rigorous generated cuts
    for constr in m.component_objects(ctype=Constraint, active=True,
                                      descend_into=(Block, Disjunct)):
        if (constr.local_name == 'GDPopt_OA_cuts'):
            constr.deactivate()

    mip_results = solve_linear_GDP(m, solve_data, config)
    if mip_results:
        config.logger.info('Solved set covering MIP')
        return mip_results + (
            list(disj.indicator_var.value
                 for disj in GDPopt.working_disjuncts_list),)
    else:
        config.logger.info(
            'Set covering problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
        if GDPopt.mip_iter <= 1:
            config.logger.warning(
                'Set covering problem was infeasible. '
                'Check your linear and logical constraints '
                'for contradictions.')
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
        return False
