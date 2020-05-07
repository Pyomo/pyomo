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

    Sets up local working model `fixed_nlp`
    Fixes binaries
    Sets continuous variables to initial var values
    Precomputes dual values
    Deactivates trivial constraints
    Solves NLP model

    Returns the fixed-NLP model and the solver results
    """

    fixed_nlp = solve_data.working_model.clone()
    MindtPy = fixed_nlp.MindtPy_utils
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                       % (solve_data.nlp_iter,))

    # Set up NLP
    TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)

    # restore original variable values
    for nlp_var, orig_val in zip(
            MindtPy.variable_list,
            solve_data.initial_var_values):
        if not nlp_var.fixed and not nlp_var.is_binary():
            nlp_var.value = orig_val

    MindtPy.MindtPy_linear_cuts.deactivate()
    fixed_nlp.tmp_duals = ComponentMap()
    # tmp_duals are the value of the dual variables stored before using deactivate trivial contraints
    # The values of the duals are computed as follows: (Complementary Slackness)
    #
    # | constraint | c_geq | status at x1 | tmp_dual (violation) |
    # |------------|-------|--------------|----------------------|
    # | g(x) <= b  | -1    | g(x1) <= b   | 0                    |
    # | g(x) <= b  | -1    | g(x1) > b    | g(x1) - b            |
    # | g(x) >= b  | +1    | g(x1) >= b   | 0                    |
    # | g(x) >= b  | +1    | g(x1) < b    | b - g(x1)            |

    for c in fixed_nlp.component_data_objects(ctype=Constraint, active=True,
                                              descend_into=True):
        # We prefer to include the upper bound as the right hand side since we are
        # considering c by default a (hopefully) convex function, which would make
        # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
        # if we don't have to
        rhs = c.upper if c. has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        # c_leq = 1 if c.has_ub else -1
        fixed_nlp.tmp_duals[c] = c_geq * max(
            0, c_geq*(rhs - value(c.body)))
        # fixed_nlp.tmp_duals[c] = c_leq * max(
        #     0, c_leq*(value(c.body) - rhs))
        # TODO: change logic to c_leq based on benchmarking

    TransformationFactory('contrib.deactivate_trivial_constraints')\
        .apply_to(fixed_nlp, tmp=True, ignore_infeasible=True)
    # Solve the NLP
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            fixed_nlp, **config.nlp_solver_args)
    return fixed_nlp, results


def handle_NLP_subproblem_optimal(fixed_nlp, solve_data, config):
    """Copies result to working model, updates bound, adds OA and integer cut,
    stores best solution if new one is best"""
    copy_var_list_values(
        fixed_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
    for c in fixed_nlp.tmp_duals:
        if fixed_nlp.dual.get(c, None) is None:
            fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
    dual_values = list(fixed_nlp.dual[c]
                       for c in fixed_nlp.MindtPy_utils.constraint_list)

    main_objective = next(
        fixed_nlp.component_data_objects(Objective, active=True))
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
        solve_data.best_solution_found = fixed_nlp.clone()

    # Add the linear cut
    if config.strategy == 'OA':
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
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
    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_integer_cuts:
        add_int_cut(var_values, solve_data, config, feasible=True)

    config.call_after_subproblem_feasible(fixed_nlp, solve_data)


def handle_NLP_subproblem_infeasible(fixed_nlp, solve_data, config):
    """Solve feasibility problem, add cut according to strategy.

    The solution of the feasibility problem is copied to the working model.
    """
    # TODO try something else? Reinitialize with different initial
    # value?
    config.logger.info('NLP subproblem was locally infeasible.')
    for c in fixed_nlp.component_data_objects(ctype=Constraint):
        rhs = c.upper if c. has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        fixed_nlp.dual[c] = (c_geq
                             * max(0, c_geq * (rhs - value(c.body))))
    dual_values = list(fixed_nlp.dual[c]
                       for c in fixed_nlp.MindtPy_utils.constraint_list)

    # if config.strategy == 'PSC' or config.strategy == 'GBD':
    #     for var in fixed_nlp.component_data_objects(ctype=Var, descend_into=True):
    #         fixed_nlp.ipopt_zL_out[var] = 0
    #         fixed_nlp.ipopt_zU_out[var] = 0
    #         if var.has_ub() and abs(var.ub - value(var)) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zL_out[var] = 1
    #         elif var.has_lb() and abs(value(var) - var.lb) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zU_out[var] = -1

    if config.strategy == 'OA':
        config.logger.info('Solving feasibility problem')
        if config.initial_feas:
            # add_feas_slacks(fixed_nlp, solve_data)
            # config.initial_feas = False
            feas_NLP, feas_NLP_results = solve_NLP_feas(solve_data, config)
            copy_var_list_values(feas_NLP.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config)
            add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
    # Add an integer cut to exclude this discrete option
    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_integer_cuts:
        # excludes current discrete option
        add_int_cut(var_values, solve_data, config)


def handle_NLP_subproblem_other_termination(fixed_nlp, termination_condition,
                                            solve_data, config):
    """Case that fix-NLP is neither optimal nor infeasible (i.e. max_iterations)"""
    if termination_condition is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        var_values = list(
            v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_integer_cuts:
            # excludes current discrete option
            add_int_cut(var_values, solve_data, config)
    else:
        raise ValueError(
            'MindtPy unable to handle NLP subproblem termination '
            'condition of {}'.format(termination_condition))


def solve_NLP_feas(solve_data, config):
    """Solves feasibility NLP and copies result to working model

    Returns: Result values and dual values
    """
    fixed_nlp = solve_data.working_model.clone()
    add_feas_slacks(fixed_nlp)
    MindtPy = fixed_nlp.MindtPy_utils
    next(fixed_nlp.component_data_objects(Objective, active=True)).deactivate()
    for constr in fixed_nlp.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        if constr.body.polynomial_degree() not in [0, 1]:
            constr.deactivate()

    MindtPy.MindtPy_feas.activate()
    MindtPy.MindtPy_feas_obj = Objective(
        expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]),
        sense=minimize)
    TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)

    with SuppressInfeasibleWarning():
        feas_soln = SolverFactory(config.nlp_solver).solve(
            fixed_nlp, **config.nlp_solver_args)
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

    for i, c in enumerate(MindtPy.constraint_list):
        rhs = c.upper if c. has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        duals[i] = c_geq * max(
            0, c_geq * (rhs - value(c.body)))

    if value(MindtPy.MindtPy_feas_obj.expr) == 0:
        raise ValueError(
            'Problem is not feasible, check NLP solver')

    return fixed_nlp, feas_soln
