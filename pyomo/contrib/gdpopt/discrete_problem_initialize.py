#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Functions for initializing the main problem
in Logic-based outer approximation.
"""
from contextlib import contextmanager
from math import fabs

from pyomo.common.collections import ComponentMap

from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc


def _collect_original_bounds(discrete_prob_util_block):
    original_bounds = ComponentMap()
    for v in discrete_prob_util_block.all_mip_variables:
        original_bounds[v] = (v.lb, v.ub)
    return original_bounds


def _restore_bounds(original_bounds):
    for v, (l, u) in original_bounds.items():
        v.setlb(l)
        v.setub(u)


# This contextmanager is for use when we solve the discrete problem with some
# variables fixed. In that case, the bounds tightening that might be done during
# preprocessing is not valid later, and we need to restore the variable bounds.
@contextmanager
def preserve_discrete_problem_feasible_region(
    discrete_problem_util_block, config, original_bounds=None
):
    if config.mip_presolve and original_bounds is None:
        original_bounds = _collect_original_bounds(discrete_problem_util_block)

    yield

    if config.mip_presolve:
        _restore_bounds(original_bounds)


def init_custom_disjuncts(
    util_block, discrete_problem_util_block, subprob_util_block, config, solver
):
    """Initialize by using user-specified custom disjuncts."""
    solver._log_header(config.logger)

    used_disjuncts = {}

    # We are going to fix indicator_vars in the discrete problem before we
    # solve it, so the bounds tightening will not necessarily be valid
    # afterward. So we save these bounds and restore them in each iteration. We
    # collect them here since there's no point in doing that every iteration.
    if config.mip_presolve:
        original_bounds = _collect_original_bounds(discrete_problem_util_block)

    for count, active_disjunct_set in enumerate(config.custom_init_disjuncts):
        # custom_init_disjuncts contains a list of sets, giving the disjuncts
        # active at each initialization iteration
        used_disjuncts = set()

        subproblem = subprob_util_block.parent_block()
        # fix the disjuncts in the linear GDP and solve
        config.logger.info(
            "Generating initial linear GDP approximation by "
            "solving subproblems with user-specified active disjuncts."
        )
        for orig_disj, discrete_problem_disj in zip(
            util_block.disjunct_list, discrete_problem_util_block.disjunct_list
        ):
            if orig_disj in active_disjunct_set:
                used_disjuncts.add(orig_disj)
                discrete_problem_disj.indicator_var.fix(True)
            else:
                discrete_problem_disj.indicator_var.fix(False)
        unused = set(active_disjunct_set) - used_disjuncts
        if len(unused) > 0:
            config.logger.warning(
                'The following disjuncts from the custom disjunct '
                'initialization set number %s were unused: '
                '%s\nThey may not be Disjunct objects or '
                'they may not be on the active subtree being '
                'solved.' % (count, ", ".join([disj.name for disj in unused]))
            )
        with preserve_discrete_problem_feasible_region(
            discrete_problem_util_block, config, original_bounds
        ):
            mip_termination = solve_MILP_discrete_problem(
                discrete_problem_util_block, solver, config
            )
        if mip_termination is not tc.infeasible:
            solver._fix_discrete_soln_solve_subproblem_and_add_cuts(
                discrete_problem_util_block, subprob_util_block, config
            )
            # remove the integer solution
            add_no_good_cut(discrete_problem_util_block, config)
        else:
            config.logger.error(
                'MILP relaxation infeasible for user-specified '
                'custom initialization disjunct set %s. '
                'Skipping that set and continuing on.'
                % list(disj.name for disj in active_disjunct_set)
            )
        solver.initialization_iteration += 1


def init_fixed_disjuncts(
    util_block, discrete_problem_util_block, subprob_util_block, config, solver
):
    """Initialize by solving the problem with the current disjunct values."""

    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving subproblem with original user-specified disjunct values."
    )
    solver._log_header(config.logger)

    # Again, if we presolve, we are going to tighten the bounds after fixing the
    # indicator_vars, so it won't be valid afterwards and we need to restore it.
    with preserve_discrete_problem_feasible_region(discrete_problem_util_block, config):
        # fix the disjuncts in the discrete problem and send for solution.
        already_fixed = set()
        for disj in discrete_problem_util_block.disjunct_list:
            indicator = disj.indicator_var
            if indicator.fixed:
                already_fixed.add(disj)
            else:
                indicator.fix()

        # We copied the variables over when we cloned, and because the Booleans
        # are auto-linked to the binaries, we shouldn't have to change
        # anything. So first we solve the discrete problem in case we need
        # values for other discrete variables, and to make sure it's feasible.
        mip_termination = solve_MILP_discrete_problem(
            discrete_problem_util_block, solver, config
        )

        # restore the fixed status of the indicator_variables
        for disj in discrete_problem_util_block.disjunct_list:
            if disj not in already_fixed:
                disj.indicator_var.unfix()

    if mip_termination is not tc.infeasible:
        solver._fix_discrete_soln_solve_subproblem_and_add_cuts(
            discrete_problem_util_block, subprob_util_block, config
        )
        add_no_good_cut(discrete_problem_util_block, config)
    else:
        config.logger.error(
            'MILP relaxation infeasible for initial user-specified '
            'disjunct values. '
            'Skipping initialization.'
        )
    solver.initialization_iteration += 1


@contextmanager
def use_discrete_problem_for_max_binary_initialization(discrete_problem_util_block):
    m = discrete_problem_util_block.parent_block()

    # Set up binary maximization objective
    original_objective = next(
        m.component_data_objects(Objective, active=True, descend_into=True)
    )
    original_objective.deactivate()

    binary_vars = (
        v
        for v in m.component_data_objects(ctype=Var, descend_into=(Block, Disjunct))
        if v.is_binary() and not v.fixed
    )
    discrete_problem_util_block.max_binary_obj = Objective(
        expr=sum(binary_vars), sense=maximize
    )

    yield

    # clean up the objective. We don't clean up the no-good cuts because we
    # still want them. We've already considered those solutions.
    del discrete_problem_util_block.max_binary_obj
    original_objective.activate()


def init_max_binaries(
    util_block, discrete_problem_util_block, subprob_util_block, config, solver
):
    """Initialize by maximizing binary variables and disjuncts.

    This function activates as many binary variables and disjuncts as
    feasible.

    """
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving a subproblem that maximizes "
        "the sum of all binary and logical variables."
    )
    solver._log_header(config.logger)

    # As with set covering, this is only a change of objective. The formulation
    # may be tightened, but that is valid for the duration.
    with use_discrete_problem_for_max_binary_initialization(
        discrete_problem_util_block
    ):
        mip_termination = solve_MILP_discrete_problem(
            discrete_problem_util_block, solver, config
        )
        if mip_termination is not tc.infeasible:
            solver._fix_discrete_soln_solve_subproblem_and_add_cuts(
                discrete_problem_util_block, subprob_util_block, config
            )
        else:
            config.logger.debug(
                "MILP relaxation for initialization was infeasible. "
                "Problem is infeasible."
            )
            solver._update_dual_bound_to_infeasible()
            return False
        add_no_good_cut(discrete_problem_util_block, config)

    solver.initialization_iteration += 1


@contextmanager
def use_discrete_problem_for_set_covering(discrete_problem_util_block):
    m = discrete_problem_util_block.parent_block()

    original_objective = next(
        m.component_data_objects(Objective, active=True, descend_into=True)
    )
    original_objective.deactivate()
    # placeholder for the objective
    discrete_problem_util_block.set_cover_obj = Objective(expr=0, sense=maximize)

    yield

    # clean up the objective. We don't clean up the no-good cuts because we
    # still want them. We've already considered those solutions.
    del discrete_problem_util_block.set_cover_obj
    original_objective.activate()


def update_set_covering_objective(discrete_problem_util_block, disj_needs_cover):
    # number of disjuncts that still need to be covered
    num_needs_cover = sum(1 for disj_bool in disj_needs_cover if disj_bool)
    # number of disjuncts that have been covered
    num_covered = len(disj_needs_cover) - num_needs_cover
    # weights for the set covering problem
    weights = list(
        (num_covered + 1 if disj_bool else 1) for disj_bool in disj_needs_cover
    )
    # Update set covering objective
    discrete_problem_util_block.set_cover_obj.expr = sum(
        weight * disj.binary_indicator_var
        for (weight, disj) in zip(weights, discrete_problem_util_block.disjunct_list)
    )


def init_set_covering(
    util_block, discrete_problem_util_block, subprob_util_block, config, solver
):
    """Initialize by solving problems to cover the set of all disjuncts.

    The purpose of this initialization is to generate linearizations
    corresponding to each of the disjuncts.

    This work is based upon prototyping work done by Eloy Fernandez at
    Carnegie Mellon University.

    """
    config.logger.info("Starting set covering initialization.")
    solver._log_header(config.logger)

    # List of True/False if the corresponding disjunct in
    # disjunct_list still needs to be covered by the initialization
    disjunct_needs_cover = list(
        any(
            constr.body.polynomial_degree() not in (0, 1)
            for constr in disj.component_data_objects(
                ctype=Constraint, active=True, descend_into=True
            )
        )
        for disj in util_block.disjunct_list
    )
    subprob = subprob_util_block.parent_block()

    # We borrow the discrete problem to be the set covering MIP. This is only a
    # change of objective. The formulation may have its bounds tightened as a
    # result of preprocessing in the MIP solves, but that is okay because the
    # feasible region is the same as the original discrete problem and any
    # feasibility-based tightening will remain valid for the duration.
    with use_discrete_problem_for_set_covering(discrete_problem_util_block):
        iter_count = 1
        while any(disjunct_needs_cover) and iter_count <= config.set_cover_iterlim:
            config.logger.debug(
                "%s disjuncts need to be covered." % disjunct_needs_cover.count(True)
            )
            ## Solve set covering MIP
            update_set_covering_objective(
                discrete_problem_util_block, disjunct_needs_cover
            )

            mip_termination = solve_MILP_discrete_problem(
                discrete_problem_util_block, solver, config
            )

            if mip_termination is tc.infeasible:
                config.logger.debug(
                    'Set covering problem is infeasible. '
                    'Problem may have no more feasible '
                    'disjunctive realizations.'
                )
                if iter_count <= 1:
                    config.logger.warning(
                        'Set covering problem is infeasible. '
                        'Check your linear and logical constraints '
                        'for contradictions.'
                    )
                solver._update_dual_bound_to_infeasible()
                # problem is infeasible. break
                return False
            else:
                config.logger.debug('Solved set covering MIP')

            ## solve local NLP
            nlp_feasible = solver._fix_discrete_soln_solve_subproblem_and_add_cuts(
                discrete_problem_util_block, subprob_util_block, config
            )
            if nlp_feasible:
                # if successful, update sets
                active_disjuncts = list(
                    fabs(value(disj.binary_indicator_var) - 1)
                    <= config.integer_tolerance
                    for disj in discrete_problem_util_block.disjunct_list
                )
                # Update the disjunct needs cover list
                disjunct_needs_cover = list(
                    (needed_cover and not was_active)
                    for (needed_cover, was_active) in zip(
                        disjunct_needs_cover, active_disjuncts
                    )
                )
            add_no_good_cut(discrete_problem_util_block, config)
            iter_count += 1
            solver.initialization_iteration += 1

        if any(disjunct_needs_cover):
            # Iteration limit was hit without a full covering of all nonlinear
            # disjuncts
            config.logger.warning(
                'Iteration limit reached for set covering initialization '
                'without covering all disjuncts.'
            )
            return False

    config.logger.info("Initialization complete.")
    return True


# Valid initialization strategies
valid_init_strategies = {
    'no_init': _DoNothing,
    'set_covering': init_set_covering,
    'max_binary': init_max_binaries,
    'fix_disjuncts': init_fixed_disjuncts,
    'custom_disjuncts': init_custom_disjuncts,
}
