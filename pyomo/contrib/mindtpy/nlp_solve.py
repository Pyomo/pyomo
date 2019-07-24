"""Solution of NLP subproblems."""
from __future__ import division

from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
        add_int_cut)
from pyomo.contrib.mindtpy.util import add_feas_slacks
from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import (Constraint, Objective, TransformationFactory, Var,
        minimize, value)
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning


def solve_NLP_subproblem(solve_data, config):
    """ Solves fixed NLP with fixed working model binaries

    Sets up local working model `fix_nlp`
    Fixes binaries
    Sets continuous variables to initial var values
    Precomputes dual values
    Deactivates trivial constraints
    Solves NLP model

    Returns the fixed-NLP model and the solver results
    """

    fix_nlp = solve_data.working_model.clone()
    MindtPy = fix_nlp.MindtPy_utils
    main_objective = next(fix_nlp.component_data_objects(Objective, active=True))
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                       % (solve_data.nlp_iter,))

    # Set up NLP
    TransformationFactory('core.fix_discrete').apply_to(fix_nlp)

    # restore original variable values
    for nlp_var, orig_val in zip(
            MindtPy.variable_list,
            solve_data.initial_var_values):
        if not nlp_var.fixed and not nlp_var.is_binary():
            nlp_var.value = orig_val

    MindtPy.MindtPy_linear_cuts.deactivate()
    fix_nlp.tmp_duals = ComponentMap()
    for c in fix_nlp.component_data_objects(ctype=Constraint, active=True,
                                            descend_into=True):
        rhs = ((0 if c.upper is None else c.upper)
               + (0 if c.lower is None else c.lower))
        sign_adjust = 1 if value(c.upper) is None else -1
        fix_nlp.tmp_duals[c] = sign_adjust * max(0,
                sign_adjust * (rhs - value(c.body)))
        # TODO check sign_adjust
    TransformationFactory('contrib.deactivate_trivial_constraints')\
        .apply_to(fix_nlp, tmp=True, ignore_infeasible=True)
    # Solve the NLP
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            fix_nlp, **config.nlp_solver_args)
    return fix_nlp, results


def handle_NLP_subproblem_optimal(fix_nlp, solve_data, config):
    """Copies result to working model, updates bound, adds OA and integer cut,
    stores best solution if new one is best"""
    copy_var_list_values(
        fix_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
    for c in fix_nlp.tmp_duals:
        if fix_nlp.dual.get(c, None) is None:
            fix_nlp.dual[c] = fix_nlp.tmp_duals[c]
    dual_values = list(fix_nlp.dual[c] for c in fix_nlp.MindtPy_utils.constraint_list)

    main_objective = next(fix_nlp.component_data_objects(Objective, active=True))
    if main_objective.sense == minimize:
        solve_data.UB = min(value(main_objective.expr), solve_data.UB)
        solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
        solve_data.UB_progress.append(solve_data.UB)
    else:
        solve_data.LB = max(value(main_objective.expr), solve_data.LB)
        solve_data.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
        solve_data.LB_progress.append(solve_data.LB)

    config.logger.info(
        'NLP {}: OBJ: {}  LB: {}  UB: {}'
        .format(solve_data.nlp_iter,
                value(main_objective.expr),
                solve_data.LB, solve_data.UB))

    if solve_data.solution_improved:
        solve_data.best_solution_found = fix_nlp.clone()

    # Add the linear cut
    if config.strategy == 'OA':
        copy_var_list_values(fix_nlp.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
    elif config.strategy == 'PSC':
        add_psc_cut(solve_data, config)
    elif config.strategy == 'GBD':
        add_gbd_cut(solve_data, config)

    # This adds an integer cut to the feasible_integer_cuts
    # ConstraintList, which is not activated by default. However, it
    # may be activated as needed in certain situations or for certain
    # values of option flags.
    var_values = list(v.value for v in fix_nlp.MindtPy_utils.variable_list)
    if config.add_integer_cuts:
        add_int_cut(var_values, solve_data, config, feasible=True)

    config.call_after_subproblem_feasible(fix_nlp, solve_data)


def handle_NLP_subproblem_infeasible(fix_nlp, solve_data, config):
    """Solve feasibility problem, add cut according to strategy.

    The solution of the feasibility problem is copied to the working model.
    """
    # TODO try something else? Reinitialize with different initial
    # value?
    config.logger.info('NLP subproblem was locally infeasible.')
    for c in fix_nlp.component_data_objects(ctype=Constraint):
        rhs = ((0 if c.upper is None else c.upper)
               + (0 if c.lower is None else c.lower))
        sign_adjust = 1 if value(c.upper) is None else -1
        fix_nlp.dual[c] = (sign_adjust
                * max(0, sign_adjust * (rhs - value(c.body))))
    dual_values = list(fix_nlp.dual[c] for c in fix_nlp.MindtPy_utils.constraint_list)

    if config.strategy == 'PSC' or config.strategy == 'GBD':
        for var in fix_nlp.component_data_objects(ctype=Var, descend_into=True):
            fix_nlp.ipopt_zL_out[var] = 0
            fix_nlp.ipopt_zU_out[var] = 0
            if var.ub is not None and abs(var.ub - value(var)) < config.bound_tolerance:
                fix_nlp.ipopt_zL_out[var] = 1
            elif var.lb is not None and abs(value(var) - var.lb) < config.bound_tolerance:
                fix_nlp.ipopt_zU_out[var] = -1

    elif config.strategy == 'OA':
        config.logger.info('Solving feasibility problem')
        if config.initial_feas:
            # add_feas_slacks(fix_nlp, solve_data)
            # config.initial_feas = False
            feas_NLP, feas_NLP_results = solve_NLP_feas(solve_data, config)
            copy_var_list_values(feas_NLP.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config)
            add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
    # Add an integer cut to exclude this discrete option
    var_values = list(v.value for v in fix_nlp.MindtPy_utils.variable_list)
    if config.add_integer_cuts:
        add_int_cut(var_values, solve_data, config)  # excludes current discrete option


def handle_NLP_subproblem_other_termination(fix_nlp, termination_condition,
                                            solve_data, config):
    """Case that fix-NLP is neither optimal nor infeasible (i.e. max_iterations)"""
    if termination_condition is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        var_values = list(v.value for v in fix_nlp.MindtPy_utils.variable_list)
        if config.add_integer_cuts:
            add_int_cut(var_values, solve_data, config)  # excludes current discrete option
    else:
        raise ValueError(
            'MindtPy unable to handle NLP subproblem termination '
            'condition of {}'.format(termination_condition))


def solve_NLP_feas(solve_data, config):
    """Solves feasibility NLP and copies result to working model

    Returns: Result values and dual values
    """
    fix_nlp = solve_data.working_model.clone()
    add_feas_slacks(fix_nlp)
    MindtPy = fix_nlp.MindtPy_utils
    next(fix_nlp.component_data_objects(Objective, active=True)).deactivate()
    for constr in fix_nlp.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        if constr.body.polynomial_degree() not in [0,1]:
            constr.deactivate()

    MindtPy.MindtPy_feas.activate()
    MindtPy.MindtPy_feas_obj = Objective(
        expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]),
        sense=minimize)
    TransformationFactory('core.fix_discrete').apply_to(fix_nlp)

    with SuppressInfeasibleWarning():
        feas_soln = SolverFactory(config.nlp_solver).solve(
            fix_nlp, **config.nlp_solver_args)
    subprob_terminate_cond = feas_soln.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
    elif subprob_terminate_cond is tc.infeasible:
        raise ValueError('Feasibility NLP infeasible. '
                         'This should never happen.')
    else:
        raise ValueError(
            'MindtPy unable to handle feasibility NLP termination condition '
            'of {}'.format(subprob_terminate_cond))

    var_values = [v.value for v in MindtPy.variable_list]
    duals = [0 for _ in MindtPy.constraint_list]

    for i, constr in enumerate(MindtPy.constraint_list):
        # TODO rhs only works if constr.upper and constr.lower do not both have values.
        # Sometimes you might have 1 <= expr <= 1. This would give an incorrect rhs of 2.
        rhs = ((0 if constr.upper is None else constr.upper)
               + (0 if constr.lower is None else constr.lower))
        sign_adjust = 1 if value(constr.upper) is None else -1
        duals[i] = sign_adjust * max(
            0, sign_adjust * (rhs - value(constr.body)))

    if value(MindtPy.MindtPy_feas_obj.expr) == 0:
        raise ValueError(
            'Problem is not feasible, check NLP solver')

    return fix_nlp, feas_soln
