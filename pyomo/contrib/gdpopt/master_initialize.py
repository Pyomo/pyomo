"""Functions for initializing the master problem
in Logic-based outer approximation.
"""
from math import fabs
from contextlib import contextmanager

from pyomo.contrib.gdpopt.cut_generation import (
    add_no_good_cut, add_cuts_according_to_algorithm)
from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.nlp_solve import solve_subproblem
from pyomo.contrib.gdpopt.util import (
    _DoNothing, fix_master_solution_in_subproblem)
from pyomo.core import (
    Block, Constraint, Objective, Suffix, TransformationFactory, Var, maximize,
    minimize, value
)
from pyomo.common.collections import ComponentMap
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct

from pytest import set_trace

# ESJ TODO: These have a lot of redundant code in terms of not messing up the
# m,aster problem when we are coopting it, and also just the usual "solve the
# master, then solve the subproblem and do all the right things" routine.

def init_custom_disjuncts(util_block, master_util_block, subprob_util_block,
                          config, solver):
    """Initialize by using user-specified custom disjuncts."""
    used_disjuncts = {}
    if config.mip_presolve:
        original_bounds = ComponentMap()
        for v in master_util_block.variable_list:
            original_bounds[v] = (v.lb, v.ub)
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
        for orig_disj, master_disj in zip(util_block.disjunct_list,
                                         master_util_block.disjunct_list):
            if orig_disj in active_disjunct_set:
                used_disjuncts.add(orig_disj)
                master_disj.indicator_var.fix(True)
            else:
                master_disj.indicator_var.fix(False)
        unused = set(active_disjunct_set) - used_disjuncts
        if len(unused) > 0:
            disj_str = ""
            for disj in unused:
                disj_str += "%s, " % disj.name # TODO: make this efficient
            config.logger.warning('The following disjuncts the custom disjunct'
                                  'initialization set number %s were unused: '
                                  '%s\nThey may not be Disjunct objects or '
                                  'they may not be on the active subtree being '
                                  'solved.' % (count, disj_str))
        mip_feasible = solve_linear_GDP(master_util_block, config,
                                        solver.timing)
        if config.mip_presolve:
            # restore bounds
            for v, (l, u) in original_bounds.items():
                v.setlb(l)
                v.setub(u)

        if mip_feasible:
            with fix_master_solution_in_subproblem(master_util_block,
                                                   subprob_util_block,
                                                   config,
                                                   config.force_subproblem_nlp):
                nlp_feasible = solve_subproblem(subprob_util_block, config,
                                                solver.timing)
                if nlp_feasible:
                    primal_improved = solver._update_bounds(
                        primal=value(subprob_util_block.obj.expr),
                        logger=config.logger)
                    if primal_improved:
                        solver.update_incumbent(subprob_util_block)
                    add_cuts_according_to_algorithm(subprob_util_block,
                                                    master_util_block,
                                                    solver.objective_sense,
                                                    config, solver.timing)
        else:
            config.logger.error(
                'Linear GDP infeasible for user-specified '
                'custom initialization disjunct set %s. '
                'Skipping that set and continuing on.'
                % list(disj.name for disj in active_disjunct_set))
        add_no_good_cut(master_util_block, config)

def init_fixed_disjuncts(util_block, master_util_block, subprob_util_block,
                         config, solver):
    """Initialize by solving the problem with the current disjunct values."""
    # TODO error checking to make sure that the user gave proper disjuncts

    # fix the disjuncts in the master problem and send for solution.
    solver.mip_iteration += 1
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving subproblem with original user-specified disjunct values.")

    if config.mip_presolve:
        original_bounds = ComponentMap()
        for v in master_util_block.variable_list:
            original_bounds[v] = (v.lb, v.ub)
    
    # We copied the variables over when we cloned, and because the Booleans are
    # auto-linked to the binaries, we shouldn't have to change anything. So
    # first we solve the master problem in case we need values for other
    # discrete variables.
    mip_feasible = solve_linear_GDP(master_util_block, config, solver.timing)
    if config.mip_presolve:
        # restore bounds
        for v, (l, u) in original_bounds.items():
            v.setlb(l)
            v.setub(u)
    
    if mip_feasible:
        with fix_master_solution_in_subproblem(master_util_block,
                                           subprob_util_block, config,
                                           config.force_subproblem_nlp):
            nlp_feasible = solve_subproblem(subprob_util_block, config,
                                            solver.timing)
            if nlp_feasible:
                primal_improved = solver._update_bounds(
                    primal=value(subprob_util_block.obj.expr),
                    logger=config.logger)
                if primal_improved:
                    solver.update_incumbent(subprob_util_block)
            add_cuts_according_to_algorithm(subprob_util_block,
                                            master_util_block,
                                            solver.objective_sense, config,
                                            solver.timing)
    else:
        config.logger.error(
            'Linear GDP infeasible for initial user-specified '
            'disjunct values. '
            'Skipping initialization.')
    add_no_good_cut(master_util_block, config)

@contextmanager
def use_master_for_max_binary_initialization(master_util_block):
    m = master_util_block.model()
    original_bounds = ComponentMap()
    for v in get_vars_from_components(m, ctype=(Constraint, Objective),
                                      active=True, descend_into=Block):
        original_bounds[v] = (v.lb, v.ub)
    active_constraints = []
    for c in m.component_data_objects(Constraint, active=True,
                                      descend_into=Block):
        active_constraints.append(c)

    # Set up binary maximization objective
    original_objective = next( m.component_data_objects(Objective, active=True,
                                                        descend_into=True))
    original_objective.deactivate()

    binary_vars = (v for v in m.component_data_objects(
        ctype=Var, descend_into=(Block, Disjunct)) if v.is_binary() 
                   and not v.fixed)
    master_util_block.max_binary_obj = Objective(expr=sum(binary_vars),
                                                 sense=maximize)
    
    yield

    # clean up the objective. We don't clean up the no-good cuts because we
    # still want them. We've already considered those solutions.
    del master_util_block.max_binary_obj
    original_objective.activate()

    # undo what fbbt might have done in preprocessing
    for v, (l, u) in original_bounds.items():
        v.setlb(l)
        v.setub(u)
    for c in active_constraints:
        c.activate()

def init_max_binaries(util_block, master_util_block, subprob_util_block, config,
                      solver):
    """Initialize by maximizing binary variables and disjuncts.

    This function activates as many binary variables and disjucts as
    feasible.

    """
    solver.mip_iteration += 1
    config.logger.info(
        "Generating initial linear GDP approximation by "
        "solving a subproblem that maximizes "
        "the sum of all binary and logical variables.")

    with use_master_for_max_binary_initialization(master_util_block):
        mip_feasible = solve_linear_GDP(master_util_block, config,
                                        solver.timing)
        if mip_feasible:
            with fix_master_solution_in_subproblem(master_util_block,
                                           subprob_util_block, config,
                                           config.force_subproblem_nlp):
                nlp_feasible = solve_subproblem(subprob_util_block, config,
                                                solver.timing)
                if nlp_feasible:
                    primal_improved = solver._update_bounds(
                        primal=value(subprob_util_block.obj.expr),
                        logger=config.logger)
                    if primal_improved:
                        solver.update_incumbent(subprob_util_block)
                add_cuts_according_to_algorithm(subprob_util_block,
                                                master_util_block,
                                                solver.objective_sense, config,
                                                solver.timing)
        else:
            config.logger.info(
                "Linear relaxation for initialization was infeasible. "
                "Problem is infeasible.")
            solver._update_dual_bound_to_infeasible(config.logger)
            return False
        add_no_good_cut(master_util_block, config)

@contextmanager
def use_master_for_set_covering(master_util_block):
    m = master_util_block.model()
    original_bounds = ComponentMap()
    for v in get_vars_from_components(m, ctype=(Constraint, Objective),
                                      active=True, descend_into=Block):
        original_bounds[v] = (v.lb, v.ub)
    active_constraints = []
    for c in m.component_data_objects(Constraint, active=True,
                                      descend_into=Block):
        active_constraints.append(c)

    original_objective = next( m.component_data_objects(Objective, active=True,
                                                        descend_into=True))
    original_objective.deactivate()
    # placeholder for the objective
    master_util_block.set_cover_obj = Objective(expr=0)

    yield

    # clean up the objective. We don't clean up the no-good cuts because we
    # still want them. We've already considered those solutions.
    del master_util_block.set_cover_obj
    original_objective.activate()

    # undo what fbbt might have done in preprocessing
    for v, (l, u) in original_bounds.items():
        v.setlb(l)
        v.setub(u)
    for c in active_constraints:
        c.activate()

def update_set_covering_objective(master_util_block, disj_needs_cover):
    # number of disjuncts that still need to be covered
    num_needs_cover = sum(1 for disj_bool in disj_needs_cover if disj_bool)
    # number of disjuncts that have been covered
    num_covered = len(disj_needs_cover) - num_needs_cover
    # weights for the set covering problem
    weights = list((num_covered + 1 if disj_bool else 1)
                   for disj_bool in disj_needs_cover)
    # Update set covering objective
    if hasattr(master_util_block, "set_cover_obj"):
        del master_util_block.set_cover_obj
    master_util_block.set_cover_obj = Objective(
        expr=sum(weight * disj.binary_indicator_var
                 for (weight, disj) in zip(
            weights, master_util_block.disjunct_list)), sense=maximize)

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
    subprob = subprob_util_block.model()

    if config.mip_presolve:
        original_bounds = ComponentMap()
        for v in master_util_block.variable_list:
            original_bounds[v] = (v.lb, v.ub)

    # borrow the master problem to be the set covering MIP. This is only a
    # change of objective
    with use_master_for_set_covering(master_util_block):
        iter_count = 1
        while (any(disjunct_needs_cover) and
               iter_count <= config.set_cover_iterlim):
            config.logger.info(
                "%s disjuncts need to be covered." %
                disjunct_needs_cover.count(True)
            )
            ## Solve set covering MIP
            update_set_covering_objective(master_util_block,
                                          disjunct_needs_cover)
            
            mip_feasible = solve_linear_GDP(master_util_block, config,
                                            solver.timing)
            if config.mip_presolve:
                # restore bounds
                for v, (l, u) in original_bounds.items():
                    v.setlb(l)
                    v.setub(u)
                
            if not mip_feasible:
                config.logger.info('Set covering problem is infeasible. '
                                   'Problem may have no more feasible '
                                   'disjunctive realizations.')
                if solver.mip_iteration <= 1:
                    config.logger.warning(
                        'Set covering problem was infeasible. '
                        'Check your linear and logical constraints '
                        'for contradictions.')
                solver._update_dual_bound_to_infeasible(config.logger)
                # problem is infeasible. break
                return False
            else:
                config.logger.info('Solved set covering MIP')

            ## solve local NLP
            with fix_master_solution_in_subproblem(
                    master_util_block,
                    subprob_util_block,
                    config,
                    make_subproblem_continuous=config.force_subproblem_nlp):
                subprob_feasible = solve_subproblem(subprob_util_block, config,
                                                    solver.timing)
                if subprob_feasible:
                    primal_improved = solver._update_bounds(
                        primal=value(subprob_util_block.obj.expr),
                        logger=config.logger)
                    if primal_improved:
                        solver.update_incumbent(subprob_util_block)
                    # if successful, updated sets
                    active_disjuncts = list(
                        fabs(value(disj.binary_indicator_var) - 1) <=
                        config.integer_tolerance for disj in
                        master_util_block.disjunct_list)
                    # Update the disjunct needs cover list
                    disjunct_needs_cover = list( 
                        (needed_cover and not was_active) for 
                        (needed_cover, was_active) in
                        zip(disjunct_needs_cover, active_disjuncts))
                    add_cuts_according_to_algorithm(subprob_util_block,
                                                    master_util_block,
                                                    solver.objective_sense,
                                                    config, solver.timing)
            add_no_good_cut(master_util_block, config)
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

# Valid initialization strategies
valid_init_strategies = {
    'no_init': _DoNothing,
    'set_covering': init_set_covering,
    'max_binary': init_max_binaries,
    'fix_disjuncts': init_fixed_disjuncts,
    'custom_disjuncts': init_custom_disjuncts
}
