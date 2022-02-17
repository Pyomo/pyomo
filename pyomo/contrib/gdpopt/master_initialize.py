"""Functions for initializing the master problem
in Logic-based outer approximation.
"""
from math import fabs

from pyomo.contrib.gdpopt.cut_generation import (
    add_integer_cut, add_subproblem_cuts)
from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.nlp_solve import solve_subproblem
from pyomo.contrib.gdpopt.util import (
    _DoNothing, fix_master_solution_in_subproblem)
from pyomo.core import (
    Block, Constraint, Objective, Suffix, TransformationFactory, Var, maximize,
    minimize
)
from pyomo.gdp import Disjunct

def init_custom_disjuncts(util_block, master_util_block, subprob_util_block,
                          config, solver):
    """Initialize by using user-specified custom disjuncts."""
    used_disjuncts = {}
    for count, active_disjunct_set in enumerate(config.custom_init_disjuncts):
        used_disjuncts = set()
        # custom_init_disjuncts contains a list of sets, giving the disjuncts
        # active at each initialization iteration

        subproblem = subprob_util_block.model()
        # fix the disjuncts in the linear GDP and solve
        solver.mip_iteration += 1
        config.logger.info(
            "Generating initial linear GDP approximation by "
            "solving subproblems with user-specified active disjuncts.")
        for orig_disj, clone_disj in zip(util_block.disjunct_list,
                                         master_util_block.disjunct_list):
            if orig_disj in active_disjunct_set:
                used_disjuncts.add(orig_disj)
                clone_disj.indicator_var.fix(True)
        unused = set(config.custom_init_disjuncts) - used_disjuncts
        if len(unused) > 0:
            disj_str = ""
            for disj in unused:
                disj_str += "%s, " % disj.name # TODO: make this efficient
            config.logger.warning('The following disjuncts the custom disjunct'
                                  'initialization set number %s were unused: '
                                  '%s\nThey may not be Disjunct objects or '
                                  'they may not be on the active subtree being '
                                  'solved.' % (count, disj_str))
        mip_result = solve_linear_GDP(master, util_block, config,
                                      solver.timing)
        if mip_result.feasible:
            with fix_master_solution_in_subproblem(master_util_block,
                                                   subprob_util_block,
                                                   config.force_subproblem_nlp):
                nlp_result = solve_subproblem(subproblem, subprob_util_block,
                                              config, timing)
                if nlp_result.feasible:
                    add_subproblem_cuts(nlp_result, solve_data, config)
                add_integer_cut(mip_result.var_values, solve_data.linear_GDP,
                                solve_data, config,
                                feasible=nlp_result.feasible)
        else:
            config.logger.error(
                'Linear GDP infeasible for user-specified '
                'custom initialization disjunct set %s. '
                'Skipping that set and continuing on.'
                % list(disj.name for disj in active_disjunct_set))


def init_fixed_disjuncts(util_block, master_util_block, subprob_util_block,
                         config, solver):
    """Initialize by solving the problem with the current disjunct values."""
    # TODO error checking to make sure that the user gave proper disjuncts

    # fix the disjuncts in the linear GDP and send for solution.
    solve_data.mip_iteration += 1
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving subproblem with original user-specified disjunct values.")
    linear_GDP = solve_data.linear_GDP.clone()
    TransformationFactory('gdp.fix_disjuncts').apply_to(linear_GDP)
    mip_result = solve_linear_GDP(linear_GDP, solve_data, config)
    if mip_result.feasible:
        nlp_result = solve_disjunctive_subproblem(mip_result, solve_data,
                                                  config)
        if nlp_result.feasible:
            add_subproblem_cuts(nlp_result, solve_data, config)
        add_integer_cut(
            mip_result.var_values, solve_data.linear_GDP, solve_data, config,
            feasible=nlp_result.feasible)
    else:
        config.logger.error(
            'Linear GDP infeasible for initial user-specified '
            'disjunct values. '
            'Skipping initialization.')


def init_max_binaries(util_block, master_util_block, subprob_util_block, config,
                      solver):
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
    next(linear_GDP.component_data_objects(Objective, active=True)).deactivate()
    binary_vars = (
        v for v in linear_GDP.component_data_objects(
        ctype=Var, descend_into=(Block, Disjunct))
        if v.is_binary() and not v.fixed)
    linear_GDP.GDPopt_utils.max_binary_obj = Objective(
        expr=sum(binary_vars), sense=maximize)

    # Solve
    mip_results = solve_linear_GDP(linear_GDP, solve_data, config)
    if mip_results.feasible:
        nlp_result = solve_disjunctive_subproblem(mip_results, solve_data,
                                                  config)
        if nlp_result.feasible:
            add_subproblem_cuts(nlp_result, solve_data, config)
        add_integer_cut(mip_results.var_values, solve_data.linear_GDP,
                        solve_data, config, feasible=nlp_result.feasible)
    else:
        config.logger.info(
            "Linear relaxation for initialization was infeasible. "
            "Problem is infeasible.")
        return False


def init_set_covering(util_block, master_util_block, subprob_util_block, config,
                      solver):
    """Initialize by solving problems to cover the set of all disjuncts.

    The purpose of this initialization is to generate linearizations
    corresponding to each of the disjuncts.

    This work is based upon prototyping work done by Eloy Fernandez at
    Carnegie Mellon University.

    """
    config.logger.info("Starting set covering initialization.")
    # List of True/False if the corresponding disjunct in
    # disjunct_list still needs to be covered by the initialization
    disjunct_needs_cover = list(
        any(constr.body.polynomial_degree() not in (0, 1)
            for constr in disj.component_data_objects(
            ctype=Constraint, active=True, descend_into=True))
        for disj in util_block.disjunct_list)

    # Set up set covering mip
    #set_cover_mip = solve_data.linear_GDP.clone()
    # Deactivate nonlinear constraints
    # for obj in set_cover_mip.component_data_objects(Objective, active=True):
    #     obj.deactivate()
    iter_count = 1
    while (any(disjunct_needs_cover) and
           iter_count <= config.set_cover_iterlim):
        config.logger.info(
            "%s disjuncts need to be covered." %
            disjunct_needs_cover.count(True)
        )
        # Solve set covering MIP
        mip_result = solve_set_cover_mip(
            set_cover_mip, disjunct_needs_cover, solve_data, config)
        if not mip_result.feasible:
            # problem is infeasible. break
            return False
        # solve local NLP
        subprob_result = solve_disjunctive_subproblem(mip_result, solve_data,
                                                      config)
        if subprob_result.feasible:
            # if successful, updated sets
            active_disjuncts = list(
                fabs(val - 1) <= config.integer_tolerance
                for val in mip_result.disjunct_values)
            # Update the disjunct needs cover list
            disjunct_needs_cover = list(
                (needed_cover and not was_active)
                for (needed_cover, was_active) in zip(disjunct_needs_cover,
                                                      active_disjuncts))
            add_subproblem_cuts(subprob_result, solve_data, config)
        add_integer_cut(
            mip_result.var_values, solve_data.linear_GDP, solve_data, config,
            feasible=subprob_result.feasible)
        add_integer_cut(
            mip_result.var_values, set_cover_mip, solve_data, config,
            feasible=subprob_result.feasible)

        iter_count += 1

    if any(disjunct_needs_cover):
        # Iteration limit was hit without a full covering of all nonlinear
        # disjuncts
        config.logger.warning(
            'Iteration limit reached for set covering initialization '
            'without covering all disjuncts.')
        return False

    config.logger.info("Initialization complete.")
    return True


def solve_set_cover_mip(master_util_block, disj_needs_cover, config, solver):
    """Solve the set covering MIP to determine next configuration."""
    m = master_util_block.model()

    # number of disjuncts that still need to be covered
    num_needs_cover = sum(1 for disj_bool in disj_needs_cover if disj_bool)
    # number of disjuncts that have been covered
    num_covered = len(disj_needs_cover) - num_needs_cover
    # weights for the set covering problem
    weights = list((num_covered + 1 if disj_bool else 1)
                   for disj_bool in disj_needs_cover)
    # Set up set covering objective
    if hasattr(master_util_block, "set_cover_obj"):
        del master_util_block.set_cover_obj
    master_util_block.set_cover_obj = Objective(
        expr=sum(weight * disj.binary_indicator_var
                 for (weight, disj) in zip(
            weights, master_util_block.disjunct_list)), sense=maximize)

    mip_results = solve_linear_GDP(master_util_block, config, solver.timing)
    if mip_results.feasible:
        config.logger.info('Solved set covering MIP')
    else:
        config.logger.info(
            'Set covering problem is infeasible. '
            'Problem may have no more feasible '
            'disjunctive realizations.')
        if solve_data.mip_iteration <= 1:
            config.logger.warning(
                'Set covering problem was infeasible. '
                'Check your linear and logical constraints '
                'for contradictions.')
        if solver.objective_sense == minimize:
            solver._update_bounds(primal=float('inf'))
        else:
            solver._update_bounds(primal=float('-inf'))

    return mip_results


# Valid initialization strategies
valid_init_strategies = {
    'no_init': _DoNothing,
    'set_covering': init_set_covering,
    'max_binary': init_max_binaries,
    'fix_disjuncts': init_fixed_disjuncts,
    'custom_disjuncts': init_custom_disjuncts
}
