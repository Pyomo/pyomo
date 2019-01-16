"""Solution of NLP subproblems."""
from __future__ import division

from pyomo.contrib.mindtpy.cut_generation import (add_gbd_cut, add_oa_cut,
                                                  add_psc_cut, add_int_cut)
from pyomo.contrib.mindtpy.util import add_feas_slacks
from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import (Constraint, Objective, TransformationFactory, Var,
                        minimize, value)
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning


def solve_NLP_subproblem(solve_data, config):
    m = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils
    main_objective = next(m.component_data_objects(Objective, active=True))
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                       % (solve_data.nlp_iter,))
    # Set up NLP
    for v in MindtPy.variable_list:
        if v.is_binary():
            v.fix(int(round(value(v))))

    # restore original variable values
    for nlp_var, orig_val in zip(
            MindtPy.variable_list,
            solve_data.initial_var_values):
        if not nlp_var.fixed and not nlp_var.is_binary():
            nlp_var.value = orig_val

    MindtPy.MindtPy_linear_cuts.deactivate()
    m.tmp_duals = ComponentMap()
    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True):
        rhs = ((0 if c.upper is None else c.upper) +
               (0 if c.lower is None else c.lower))
        sign_adjust = 1 if value(c.upper) is None else -1
        m.tmp_duals[c] = sign_adjust * max(0,
                                           sign_adjust * (rhs - value(c.body)))
        # TODO check sign_adjust
    t = TransformationFactory('contrib.deactivate_trivial_constraints')
    t.apply_to(m, tmp=True, ignore_infeasible=True)
    # Solve the NLP
    # m.pprint() # print nlp problem for debugging
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            m, **config.nlp_solver_args)
    var_values = list(v.value for v in MindtPy.variable_list)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        copy_var_list_values(
            m.MindtPy_utils.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        for c in m.tmp_duals:
            if m.dual.get(c, None) is None:
                m.dual[c] = m.tmp_duals[c]
        duals = list(m.dual[c] for c in MindtPy.constraint_list)
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
            .format(solve_data.nlp_iter, value(main_objective.expr), solve_data.LB, solve_data.UB))
        if solve_data.solution_improved:
            solve_data.best_solution_found = m.clone()
        # Add the linear cut
        if config.strategy == 'OA':
            add_oa_cut(var_values, duals, solve_data, config)
        elif config.strategy == 'PSC':
            add_psc_cut(solve_data, config)
        elif config.strategy == 'GBD':
            add_gbd_cut(solve_data, config)

        # This adds an integer cut to the feasible_integer_cuts
        # ConstraintList, which is not activated by default. However, it
        # may be activated as needed in certain situations or for certain
        # values of option flags.
        add_int_cut(var_values, solve_data, config, feasible=True)

        config.call_after_subproblem_feasible(m, solve_data)
    elif subprob_terminate_cond is tc.infeasible:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            rhs = ((0 if c.upper is None else c.upper) +
                   (0 if c.lower is None else c.lower))
            sign_adjust = 1 if value(c.upper) is None else -1
            m.dual[c] = sign_adjust * max(0,
                                          sign_adjust * (rhs - value(c.body)))
        for var in m.component_data_objects(ctype=Var,
                                            descend_into=True):

            if config.strategy == 'PSC' or config.strategy == 'GBD':
                m.ipopt_zL_out[var] = 0
                m.ipopt_zU_out[var] = 0
                if var.ub is not None and abs(var.ub - value(var)) < config.bound_tolerance:
                    m.ipopt_zL_out[var] = 1
                elif var.lb is not None and abs(value(var) - var.lb) < config.bound_tolerance:
                    m.ipopt_zU_out[var] = -1
        # m.pprint() #print infeasible nlp problem for debugging
        if config.strategy == 'PSC':
            config.logger.info('Adding PSC feasibility cut.')
            add_psc_cut(m, solve_data, config, nlp_feasible=False)
        elif config.strategy == 'GBD':
            config.logger.info('Adding GBD feasibility cut.')
            add_gbd_cut(m, solve_data, config, nlp_feasible=False)
        elif config.strategy == 'OA':
            config.logger.info('Solving feasibility problem')
            if config.initial_feas:
                # add_feas_slacks(m, solve_data)
                # config.initial_feas = False
                var_values, duals = solve_NLP_feas(solve_data, config)
                add_oa_cut(var_values, duals, solve_data, config)
        # Add an integer cut to exclude this discrete option
        add_int_cut(var_values, solve_data, config)
    elif subprob_terminate_cond is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem failed to converge within iteration limit.')
        # Add an integer cut to exclude this discrete option
        add_int_cut(solve_data, config)
    else:
        raise ValueError(
            'MindtPy unable to handle NLP subproblem termination '
            'condition of {}'.format(subprob_terminate_cond))

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(m, solve_data)


def solve_NLP_feas(solve_data, config):
    m = solve_data.working_model.clone()
    add_feas_slacks(m)
    MindtPy = m.MindtPy_utils
    next(m.component_data_objects(Objective, active=True)).deactivate()
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        constr.deactivate()
    MindtPy.MindtPy_feas.activate()
    MindtPy.MindtPy_feas_obj = Objective(
        expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]),
        sense=minimize)
    for v in MindtPy.variable_list:
        if v.is_binary():
            v.fix(int(round(v.value)))
    # m.pprint()  #print nlp feasibility problem for debugging
    with SuppressInfeasibleWarning():
        feas_soln = SolverFactory(config.nlp_solver).solve(
            m, **config.nlp_solver_args)
    subprob_terminate_cond = feas_soln.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        pass
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
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        sign_adjust = 1 if value(constr.upper) is None else -1
        duals[i] = sign_adjust * max(
            0, sign_adjust * (rhs - value(constr.body)))

    if value(MindtPy.MindtPy_feas_obj.expr) == 0:
        raise ValueError(
            'Problem is not infeasible, check NLP solver')

    return var_values, duals
