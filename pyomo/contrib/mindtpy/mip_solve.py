"""Master problem functions."""
from __future__ import division

from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import Constraint, Expression, Objective, minimize, value, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing
from pyomo.contrib.gdpopt.mip_solve import distinguish_mip_infeasible_or_unbounded
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import cplex
from cplex.callbacks import LazyConstraintCallback
from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal, handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination)
from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
                                                  add_int_cut)
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables
from math import copysign
from pyomo.environ import *
from pyomo.core import Constraint, minimize, value
from pyomo.core.expr import current as EXPR


class LazyOACallback(LazyConstraintCallback):

    def _get_gap(self):
        print('self.get_MIP_relative_gap: ', self.get_MIP_relative_gap())

    def add_lazy_oa_cuts(self, target_model, dual_values, solve_data, config, opt,
                         linearize_active=True,
                         linearize_violated=True,
                         linearize_inactive=False,
                         use_slack_var=False):
        """Linearizes nonlinear constraints.

        For nonconvex problems, turn on 'use_slack_var'. Slack variables will
        always be used for nonlinear equality constraints.
        """
        self._get_gap()
        for (constr, dual_value) in zip(target_model.MindtPy_utils.constraint_list,
                                        dual_values):
            if constr.body.polynomial_degree() in (0, 1):
                continue

            constr_vars = list(identify_variables(constr.body))
            jacs = solve_data.jacobians
            self._get_gap()

            # Equality constraint (makes the problem nonconvex)
            if constr.has_ub() and constr.has_lb() and constr.upper == constr.lower:
                self._get_gap()
                print('1.91------------')
                sign_adjust = -1 if solve_data.objective_sense == minimize else 1
                rhs = ((0 if constr.upper is None else constr.upper)
                       + (0 if constr.lower is None else constr.lower))
                rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs
                # slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()
                self._get_gap()
                print(self.get_MIP_relative_gap())
                # opt._add_var(slack_var)
                self._get_gap()
                print('1.911------------')
                # target_model.pprint()
                print('1.9111-----------')
                for var in list(EXPR.identify_variables(constr.body)):
                    var.set_value(self.get_values(opt._pyomo_var_to_solver_var_map[var]))
                
                print('1.9112')
                temp1 = copysign(1, sign_adjust * dual_value) 
                print('temp1: ',temp1)
                temp2 = value(constr.body)
                print('temp2: ',temp2)
                for key in  jacs.keys():
                    print(key.local_name)
                    print(key.parent_component)
                    print(key.root_block)
                    print(key==constr)
                    print(key,type(key))
                    key.pprint()
                    print(jacs[key])
                    if key.local_name == constr.local_name:
                        constr_temp = key
                    # for key2 in jacs[key].keys():
                    #     print(key2,type(key2))
                    #     print(jacs[key][key2])
                for var in list(EXPR.identify_variables(constr_temp.body)):
                    print(1)
                    print(constr_temp,type(constr_temp))
                    constr_temp.pprint()
                    print(jacs[constr_temp])
                    print(type(jacs[constr_temp][var]))
                temp3 =  sum(value(jacs[constr_temp][var]) for var in list(EXPR.identify_variables(constr_temp.body)))
                print('temp3: ',temp3)
                temp4 =  sum(( self.get_values(opt._pyomo_var_to_solver_var_map[var])) for var in list(EXPR.identify_variables(constr_temp.body)))
                print('temp4: ',temp4)
                temp5 =  sum(value(jacs[constr][var]) * (var - self.get_values(opt._pyomo_var_to_solver_var_map(var))) for var in list(EXPR.identify_variables(constr.body)))
                print('1.9113')
                cplex_expr, _ = opt._get_expr_from_pyomo_expr(copysign(1, sign_adjust * dual_value)
                                                              * (sum(self.get_values(opt._pyomo_var_to_solver_var_map[jacs[constr][var]]) * (var - self.get_values(opt._pyomo_var_to_solver_var_map(var))) for var in list(EXPR.identify_variables(constr.body))) + value(constr.body) - rhs))
                print('1.912-----------')
                self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                         sense="L",
                         rhs=0)
                print('2------------')

            else:  # Inequality constraint (possibly two-sided)
                if constr.has_ub() \
                    and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) \
                        or (linearize_violated and constr.uslack() < 0) \
                        or (linearize_inactive and constr.uslack() > 0):
                    print('1.92------------')
                    if use_slack_var:
                        slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(sum(value(jacs[constr][var]) * (var - var.value)
                                                                      for var in constr_vars))
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                             sense="L",
                             rhs=constr.upper)
                    print('2------------')

                if constr.has_lb() \
                    and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) \
                        or (linearize_violated and constr.lslack() < 0) \
                        or (linearize_inactive and constr.lslack() > 0):
                    print('1.93------------')
                    if use_slack_var:
                        slack_var = target_model.MindtPy_utils.MindtPy_linear_cuts.slack_vars.add()

                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(sum(value(jacs[constr][var]) * (var - var.value)
                                                                      for var in constr_vars))
                    print('2------------')
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                             sense="G",
                             rhs=constr.lower)

    def __call__(self):
        print('1**************')
        solve_data = self.solve_data
        config = self.config
        opt = self.opt
        master_mip = self.master_mip
        cpx = opt._solver_model
        print('2**************')

        handle_master_mip_optimal(master_mip, solve_data, config,copy=False)
        print('3**************')

        # Call the MILP post-solve callback
        # config.call_after_master_solve(master_mip, solve_data)

        # solve subproblem
        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        fix_nlp, fix_nlp_result = solve_NLP_subproblem(solve_data, config)
        print('1------------')
        print('self.get_MIP_relative_gap: ', self.get_MIP_relative_gap())
        self._get_gap()

        # add oa cuts
        # if fix_nlp_result.solver.termination_condition is tc.optimal:
        #     handle_NLP_subproblem_optimal(fix_nlp, solve_data, config)
        # elif fix_nlp_result.solver.termination_condition is tc.infeasible:
        #     handle_NLP_subproblem_infeasible(fix_nlp, solve_data, config)
        # else:
        #     handle_NLP_subproblem_other_termination(fix_nlp, fix_nlp_result.solver.termination_condition,
        #                                             solve_data, config)

        # need to be changed here, since oa cuts are need to be added through self.add
        # don't need we don't need to solve the master_mip again, we just need to continue branch and bound.
        # copy_var_list_values(
        #     fix_nlp.MindtPy_utils.variable_list,
        #     master_mip.MindtPy_utils.variable_list,
        #     # solve_data.working_model.MindtPy_utils.variable_list,
        #     config)
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
        print('1.5------------')

        if config.strategy == 'OA':
            # don't need we don't need to solve the master_mip again, we just need to continue branch and bound.
            # copy_var_list_values(fix_nlp.MindtPy_utils.variable_list,
            #                     master_mip.MindtPy_utils.variable_list,
            #                     # solve_data.mip.MindtPy_utils.variable_list,
            #                     config)
            print('1.7------------')
            self.add_lazy_oa_cuts(master_mip, dual_values, solve_data, config, opt)
            # self.add_lazy_oa_cuts(solve_data.mip, dual_values, solve_data, config, self.opt)

        # elif config.strategy == 'PSC':
        #     add_psc_cut(solve_data, config)
        # elif config.strategy == 'GBD':
        #     add_gbd_cut(solve_data, config)


def solve_OA_master(solve_data, config):
    solve_data.mip_iter += 1
    master_mip = solve_data.mip.clone()
    MindtPy = master_mip.MindtPy_utils
    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    # Set up MILP
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()

    MindtPy.MindtPy_linear_cuts.activate()
    main_objective = next(master_mip.component_data_objects(Objective, active=True))
    main_objective.deactivate()

    sign_adjust = 1 if main_objective.sense == minimize else -1
    if config.add_slack == True:
        MindtPy.MindtPy_penalty_expr = Expression(
            expr=sign_adjust * config.OA_penalty_factor * sum(
                v for v in MindtPy.MindtPy_linear_cuts.slack_vars[...]))

        MindtPy.MindtPy_oa_obj = Objective(
            expr=main_objective.expr + MindtPy.MindtPy_penalty_expr,
            sense=main_objective.sense)
    elif config.add_slack == False:
        MindtPy.MindtPy_oa_obj = Objective(
            expr=main_objective.expr,
            sense=main_objective.sense)


    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(master_mip, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(master_mip, 'ipopt_zU_out', _DoNothing()).deactivate()

    # master_mip.pprint() #print oa master problem for debugging
    with SuppressInfeasibleWarning():
        masteropt = SolverFactory(config.mip_solver)
        if isinstance(masteropt, PersistentSolver):
            masteropt.set_instance(master_mip)  # , symbolic_solver_labels=True)
            print('instance set!')
        if config.lazy_callback == True:
            # for i in solve_data.mip.component_data_objects(Var, active=True):
            #     print(i, '-------------')
            lazyoa = masteropt._solver_model.register_callback(LazyOACallback)
            lazyoa.solve_data = solve_data
            lazyoa.config = config
            lazyoa.opt = masteropt
            lazyoa.master_mip = master_mip
        print('1**************1')
        master_mip_results = masteropt.solve(master_mip, **config.mip_solver_args, tee=True)
    if master_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        master_mip_results, _ = distinguish_mip_infeasible_or_unbounded(master_mip, config)

    return master_mip, master_mip_results


def handle_master_mip_optimal(master_mip, solve_data, config,copy=True):
    """Copy the result to working model and update upper or lower bound"""
    # proceed. Just need integer values
    MindtPy = master_mip.MindtPy_utils
    main_objective = next(master_mip.component_data_objects(Objective, active=True))
    # initialize(warmstart) subproblem
    if copy == True:
        copy_var_list_values(
            master_mip.MindtPy_utils.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)

    if main_objective.sense == minimize:
        solve_data.LB = max(
            value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
        solve_data.LB_progress.append(solve_data.LB)
    else:
        solve_data.UB = min(
            value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
        solve_data.UB_progress.append(solve_data.UB)
    config.logger.info(
        'MIP %s: OBJ: %s  LB: %s  UB: %s'
        % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
           solve_data.LB, solve_data.UB))


def handle_master_mip_other_conditions(master_mip, master_mip_results, solve_data, config):
    if master_mip_results.solver.termination_condition is tc.infeasible:
        handle_master_mip_infeasible(master_mip, solve_data, config)
    elif master_mip_results.solver.termination_condition is tc.unbounded:
        handle_master_mip_unbounded(master_mip, solve_data, config)
    elif master_mip_results.solver.termination_condition is tc.maxTimeLimit:
        handle_master_mip_max_timelimit(master_mip, solve_data, config)
    elif (master_mip_results.solver.termination_condition is tc.other and
            master_mip_results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        MindtPy = master_mip.MindtPy_utils
        config.logger.info(
            'MILP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        copy_var_list_values(
            master_mip.MindtPy_utils.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
               solve_data.LB, solve_data.UB))
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of %s. Solver message: %s' %
            (master_mip_results.solver.termination_condition, master_mip_results.solver.message))


def handle_master_mip_infeasible(master_mip, solve_data, config):
    config.logger.info(
        'MILP master problem is infeasible. '
        'Problem may have no more feasible '
        'binary configurations.')
    if solve_data.mip_iter == 1:
        config.logger.warn(
            'MindtPy initialization may have generated poor '
            'quality cuts.')
    # set optimistic bound to infinity
    main_objective = next(master_mip.component_data_objects(Objective, active=True))
    if main_objective.sense == minimize:
        solve_data.LB = float('inf')
        solve_data.LB_progress.append(solve_data.UB)
    else:
        solve_data.UB = float('-inf')
        solve_data.UB_progress.append(solve_data.UB)


def handle_master_mip_max_timelimit(master_mip, solve_data, config):
    # TODO check that status is actually ok and everything is feasible
    MindtPy = master_mip.MindtPy_utils
    config.logger.info(
        'Unable to optimize MILP master problem '
        'within time limit. '
        'Using current solver feasible solution.')
    copy_var_list_values(
        master_mip.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
    if MindtPy.obj.sense == minimize:
        solve_data.LB = max(
            value(MindtPy.obj.expr), solve_data.LB)
        solve_data.LB_progress.append(solve_data.LB)
    else:
        solve_data.UB = min(
            value(MindtPy.obj.expr), solve_data.UB)
        solve_data.UB_progress.append(solve_data.UB)
    config.logger.info(
        'MIP %s: OBJ: %s  LB: %s  UB: %s'
        % (solve_data.mip_iter, value(MindtPy.obj.expr),
           solve_data.LB, solve_data.UB))


def handle_master_mip_unbounded(master_mip, solve_data, config):
    # Solution is unbounded. Add an arbitrary bound to the objective and resolve.
    # This occurs when the objective is nonlinear. The nonlinear objective is moved
    # to the constraints, and deactivated for the linear master problem.
    MindtPy = master_mip.MindtPy_utils
    config.logger.warning(
        'Master MILP was unbounded. '
        'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. '
        'You can change this bound with the option obj_bound.'.format(config.obj_bound))
    main_objective = next(master_mip.component_data_objects(Objective, active=True))
    MindtPy.objective_bound = Constraint(expr=(-config.obj_bound, main_objective.expr, config.obj_bound))
    with SuppressInfeasibleWarning():
        opt = SolverFactory(config.mip_solver)
        if isinstance(opt, PersistentSolver):
            opt.set_instance(master_mip)
        master_mip_results = opt.solve(
            master_mip, **config.mip_solver_args)
