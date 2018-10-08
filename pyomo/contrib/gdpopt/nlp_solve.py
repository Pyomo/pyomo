"""Functions for solving the nonlinear subproblem."""
from __future__ import division

from pyomo.contrib.gdpopt.data_class import SubproblemResult
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning,
                                       copy_and_fix_mip_values_to_nlp,
                                       is_feasible)
from pyomo.core import Constraint, TransformationFactory, minimize, value
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory


def solve_NLP(nlp_model, solve_data, config):
    """Solve the NLP subproblem."""
    config.logger.info(
        'Solving nonlinear subproblem for '
        'fixed binaries and logical realizations.')
    unfixed_discrete_vars = detect_unfixed_discrete_vars(nlp_model)
    if unfixed_discrete_vars:
        discrete_var_names = list(v.name for v in unfixed_discrete_vars)
        config.logger.warning(
            "Unfixed discrete variables exist on the NLP subproblem: %s"
            % (discrete_var_names,))

    GDPopt = nlp_model.GDPopt_utils

    preprocessing_transformations = [
        # Propagate variable bounds
        'contrib.propagate_eq_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Propagate fixed variables
        'contrib.propagate_fixed_vars',
        # Remove zero terms in linear expressions
        'contrib.remove_zero_terms',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Transform bound constraints
        'contrib.constraints_to_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Remove trivial constraints
        'contrib.deactivate_trivial_constraints']
    for xfrm in preprocessing_transformations:
        TransformationFactory(xfrm).apply_to(nlp_model)

    initialize_NLP(nlp_model, solve_data)

    # Callback immediately before solving NLP subproblem
    config.call_before_subproblem_solve(nlp_model, solve_data)

    nlp_solver = SolverFactory(config.nlp_solver)
    if not nlp_solver.available():
        raise RuntimeError("NLP solver %s is not available." %
                           config.nlp_solver)
    with SuppressInfeasibleWarning():
        results = nlp_solver.solve(nlp_model, **config.nlp_solver_args)

    nlp_result = SubproblemResult()
    nlp_result.feasible = True
    nlp_result.var_values = list(v.value for v in GDPopt.working_var_list)
    nlp_result.pyomo_results = results
    nlp_result.dual_values = list(
        nlp_model.dual.get(c, None)
        for c in GDPopt.working_constraints_list)

    subprob_terminate_cond = results.solver.termination_condition
    if (subprob_terminate_cond is tc.optimal or
            subprob_terminate_cond is tc.locallyOptimal):
        pass
    elif subprob_terminate_cond is tc.infeasible:
        config.logger.info('NLP subproblem was locally infeasible.')
        nlp_result.feasible = False
    elif subprob_terminate_cond is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        if is_feasible(nlp_model, config):
            config.logger.info(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
        else:
            nlp_result.feasible = False
    elif subprob_terminate_cond is tc.internalSolverError:
        # Possible that IPOPT had a restoration failture
        config.logger.info(
            "NLP solver had an internal failure: %s" % results.solver.message)
        nlp_result.feasible = False
    else:
        raise ValueError(
            'GDPopt unable to handle NLP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(nlp_model, solve_data)

    # if feasible, call the NLP post-feasible callback
    if nlp_result.feasible:
        config.call_after_subproblem_feasible(nlp_model, solve_data)

    return nlp_result


def detect_unfixed_discrete_vars(model):
    """Detect unfixed discrete variables in use on the model."""
    var_set = ComponentSet()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=True):
        var_set.update(
            v for v in EXPR.identify_variables(
                constr.body, include_fixed=False)
            if v.is_binary())
    return var_set


def initialize_NLP(model, solve_data):
    """Perform initialization of the NLP.

    Presently, this just restores the variable to the original model values.

    """
    # restore original variable values
    for var, old_value in zip(model.GDPopt_utils.working_var_list,
                              solve_data.initial_var_values):
        if not var.fixed and not var.is_binary():
            if old_value is not None:
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                # Set the value
                var.set_value(old_value)


def update_nlp_progress_indicators(solved_model, solve_data, config):
    """Update the progress indicators for the NLP subproblem."""
    GDPopt = solved_model.GDPopt_utils
    if GDPopt.objective.sense == minimize:
        old_UB = solve_data.UB
        solve_data.UB = min(
            value(GDPopt.objective.expr), solve_data.UB)
        solve_data.feasible_solution_improved = (solve_data.UB < old_UB)
    else:
        old_LB = solve_data.LB
        solve_data.LB = max(
            value(GDPopt.objective.expr), solve_data.LB)
        solve_data.feasible_solution_improved = (solve_data.LB > old_LB)
    solve_data.iteration_log[
        (solve_data.master_iteration,
         solve_data.mip_iteration,
         solve_data.nlp_iteration)
    ] = (
        value(GDPopt.objective.expr),
        value(GDPopt.objective.expr),
        [v.value for v in GDPopt.working_var_list]
    )

    if solve_data.feasible_solution_improved:
        solve_data.best_solution_found = [
            v.value for v in GDPopt.working_var_list]

    improvement_tag = (
        "(IMPROVED) " if solve_data.feasible_solution_improved else "")
    lb_improved, ub_improved = (
        ("", improvement_tag)
        if solve_data.objective_sense == minimize
        else (improvement_tag, ""))
    config.logger.info(
        'ITER %s.%s.%s-NLP: OBJ: %s  LB: %s %s UB: %s %s'
        % (solve_data.master_iteration,
           solve_data.mip_iteration,
           solve_data.nlp_iteration,
           value(GDPopt.objective.expr),
           solve_data.LB, lb_improved,
           solve_data.UB, ub_improved))


def solve_LOA_subproblem(mip_var_values, solve_data, config):
    """Set up and solve the local LOA subproblem."""
    nlp_model = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1
    # copy in the discrete variable values
    copy_and_fix_mip_values_to_nlp(nlp_model.GDPopt_utils.working_var_list,
                                   mip_var_values, config)
    TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)

    nlp_result = solve_NLP(nlp_model, solve_data, config)
    if nlp_result.feasible:  # NLP is feasible
        update_nlp_progress_indicators(nlp_model, solve_data, config)
    return nlp_result


def solve_global_NLP(mip_var_values, solve_data, config):
    """Set up and solve the global LOA subproblem."""
    nlp_model = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1
    # copy in the discrete variable values
    copy_and_fix_mip_values_to_nlp(nlp_model.GDPopt_utils.working_var_list,
                                   mip_var_values, config)
    TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
    nlp_model.dual.deactivate()  # global solvers may not give dual info

    nlp_result = solve_NLP(nlp_model, solve_data, config)
    if nlp_result.feasible:  # NLP is feasible
        update_nlp_progress_indicators(nlp_model, solve_data, config)
    return nlp_result
