"""Solution of NLP subproblems."""
from __future__ import division

from pyomo.contrib.mindtpy.cut_generation import (add_gbd_cut, add_oa_cut,
                                                  add_psc_cut, add_int_cut)
from pyomo.contrib.mindtpy.util import copy_values, add_feas_slacks
from pyomo.core import (Constraint, Objective, TransformationFactory, Var,
                        minimize, value)
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning


def solve_NLP_subproblem(solve_data, config):
    m = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                       % (solve_data.nlp_iter,))
    # Set up NLP
    for v in MindtPy.binary_vars:
        v.fix(int(value(v) + 0.5))

    # restore original variable values
    for nlp_var, orig_var in zip(
            MindtPy.var_list,
            solve_data.original_model.MindtPy_utils.var_list):
        if not nlp_var.fixed and not nlp_var.is_binary():
            nlp_var.value = orig_var.value

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
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        copy_values(m, solve_data.working_model, config)
        var_values = list(v.value for v in MindtPy.var_list)
        for c in m.tmp_duals:
            if m.dual.get(c, None) is None:
                m.dual[c] = m.tmp_duals[c]
        duals = list(m.dual[c] for c in MindtPy.constraints)
        if MindtPy.objective.sense == minimize:
            solve_data.UB = min(value(MindtPy.objective.expr), solve_data.UB)
            solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)
        else:
            solve_data.LB = max(value(MindtPy.objective.expr), solve_data.LB)
            solve_data.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)
        print('NLP {}: OBJ: {}  LB: {}  UB: {}'
              .format(solve_data.nlp_iter, value(MindtPy.objective.expr), solve_data.LB,
                      solve_data.UB))
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
        print('NLP subproblem was locally infeasible.')
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
            print('Adding PSC feasibility cut.')
            add_psc_cut(m, solve_data, config, nlp_feasible=False)
        elif config.strategy == 'GBD':
            print('Adding GBD feasibility cut.')
            add_gbd_cut(m, solve_data, config, nlp_feasible=False)
        elif config.strategy == 'OA':
            print('Solving feasibility problem')
            if config.initial_feas:
                add_feas_slacks(m, solve_data, config)
                config.initial_feas = False
            solve_NLP_feas(solve_data, config)
            add_oa_cut(solve_data, config)
        # Add an integer cut to exclude this discrete option
        add_int_cut(solve_data, config)
    elif subprob_terminate_cond is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        print('NLP subproblem failed to converge within iteration limit.')
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
    MindtPy = m.MindtPy_utils
    MindtPy.MindtPy_objective.deactivate()
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        constr.deactivate()
    MindtPy.MindtPy_feas.activate()
    MindtPy.MindtPy_feas_obj = Objective(
        expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]),
        sense=minimize)
    for v in MindtPy.binary_vars:
        if value(v) > 0.5:
            v.fix(1)
        else:
            v.fix(0)
    # m.pprint()  #print nlp feasibility problem for debugging
    with SuppressInfeasibleWarning():
        feas_soln = config.nlp_solver.solve(
            m, **config.nlp_solver_args)
    subprob_terminate_cond = feas_soln.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        copy_values(m, solve_data.working_model, config)
    elif subprob_terminate_cond is tc.infeasible:
        raise ValueError('Feasibility NLP infeasible. '
                         'This should never happen.')
    else:
        raise ValueError(
            'MindtPy unable to handle feasibility NLP termination condition '
            'of {}'.format(subprob_terminate_cond))

    for v in MindtPy.binary_vars:
        v.unfix()

    MindtPy.MindtPy_feas.deactivate()
    MindtPy.MindtPy_feas_obj.deactivate()
    # MindtPy.MindtPy_objective_expr.activate()
    MindtPy.MindtPy_objective.activate()

    for constr in m.component_data_objects(
            ctype=Constraint, descend_into=True):
        constr.activate()
        rhs = ((0 if constr.upper is None else constr.upper) +
               (0 if constr.lower is None else constr.lower))
        sign_adjust = 1 if value(constr.upper) is None else -1
        m.dual[constr] = sign_adjust * max(0,
                                           sign_adjust * (rhs - value(constr.body)))

    if value(MindtPy.MindtPy_feas_obj.expr) == 0:
        raise ValueError(
            'Problem is not infeasible, check NLP solver')
