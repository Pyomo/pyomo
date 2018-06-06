"""Functions for solving the nonlinear subproblem
in Logic-based outer approximation.
"""
from __future__ import division

from pyomo.contrib.gdpopt.util import is_feasible
from pyomo.core import Block, TransformationFactory, Var, minimize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus


def solve_NLP(nlp_model, solve_data, config):
    """Solve the NLP subproblem."""
    config.logger.info(
        'Solving nonlinear subproblem for '
        'fixed binaries and logical realizations.')
    if any(v for v in nlp_model.component_data_objects(
        ctype=Var, descend_into=(Block, Disjunct), active=True)
        if (v.is_binary() or v.is_integer()) and not v.fixed
    ):
        discrete_var_names = list(
            v.name for v in nlp_model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct), active=True)
            if (v.is_binary() or v.is_integer()) and not v.fixed)
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

    # restore original variable values
    for var, old_value in zip(GDPopt.initial_var_list,
                              GDPopt.initial_var_values):
        if not var.fixed and not var.is_binary():
            if old_value is not None:
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                # Set the value
                var.set_value(old_value)

    # Callback immediately before solving NLP subproblem
    config.subprob_presolve(nlp_model, solve_data)

    nlp_solver = SolverFactory(config.nlp)
    if not nlp_solver.available():
        raise RuntimeError("NLP solver %s is not available." % config.nlp)
    results = nlp_solver.solve(nlp_model, load_solutions=False,
                               **config.nlp_options)

    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        nlp_feasible = True
    elif subprob_terminate_cond is tc.infeasible:
        config.logger.info('NLP subproblem was locally infeasible.')
        # Suppress the warning message by setting solver status to ok.
        results.solver.status = SolverStatus.ok
        nlp_feasible = False
    elif subprob_terminate_cond is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        results.solver.status = SolverStatus.ok
        if is_feasible(nlp_model, config):
            config.logger.info(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
            nlp_feasible = True
        else:
            nlp_feasible = False
    else:
        raise ValueError(
            'GDPopt unable to handle NLP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

    nlp_model.solutions.load_from(results)

    # Call the NLP post-solve callback
    config.subprob_postsolve(nlp_model, solve_data)

    return (
        nlp_feasible,  # If solution is feasible.
        # Variable values
        list((v.value if not v.stale else None)
             for v in GDPopt.initial_var_list),
        # Dual values
        list(nlp_model.dual.get(c, None)
             for c in GDPopt.initial_constraints_list))


def update_nlp_progress_indicators(model, solve_data, config):
    """Update the progress indicators for the NLP subproblem."""
    GDPopt = model.GDPopt_utils
    if GDPopt.objective.sense == minimize:
        solve_data.UB = min(
            value(GDPopt.objective.expr), solve_data.UB)
        solve_data.solution_improved = (
            solve_data.UB < solve_data.UB_progress[-1])
        solve_data.UB_progress.append(solve_data.UB)
    else:
        solve_data.LB = max(
            value(GDPopt.objective.expr), solve_data.LB)
        solve_data.solution_improved = (
            solve_data.LB > solve_data.LB_progress[-1])
        solve_data.LB_progress.append(solve_data.LB)

    if solve_data.solution_improved:
        solve_data.best_solution_found = [
            v.value for v in GDPopt.initial_var_list]

    improvement_tag = (
        "(IMPROVED) " if solve_data.solution_improved else "")
    lb_improved, ub_improved = (
        ("", improvement_tag)
        if solve_data.objective_sense == minimize
        else (improvement_tag, ""))
    config.logger.info(
        'ITER %s.%s-NLP: OBJ: %s  LB: %s %s UB: %s %s'
        % (solve_data.master_iteration,
           solve_data.subproblem_iteration,
           value(GDPopt.objective.expr),
           solve_data.LB, lb_improved,
           solve_data.UB, ub_improved))
