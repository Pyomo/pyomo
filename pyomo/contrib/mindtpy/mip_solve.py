"""Master problem functions."""
from __future__ import division

from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import Constraint, Expression, Objective, minimize, value, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing
from pyomo.contrib.gdpopt.mip_solve import distinguish_mip_infeasible_or_unbounded
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal, handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination, solve_NLP_feas)
from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
                                                  add_int_cut)
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables
from math import copysign
from pyomo.environ import *
from pyomo.core import Constraint, minimize, value
from pyomo.core.expr import current as EXPR
from math import fabs

from pyomo.repn import generate_standard_repn

try:
    import cplex
    from cplex.callbacks import LazyConstraintCallback
except ImportError:
    print("Cplex python API is not found. Therefore, lp-nlp is not supported")
    '''Other solvers (e.g. Gurobi) are not supported yet'''


class LazyOACallback(LazyConstraintCallback):
    """Inherent class in Cplex to call Lazy callback."""

    def copy_lazy_var_list_values(self, opt, from_list, to_list, config,
                                  skip_stale=False, skip_fixed=True,
                                  ignore_integrality=False):
        """Copy variable values from one list to another.

        Rounds to Binary/Integer if neccessary
        Sets to zero for NonNegativeReals if neccessary
        """
        for v_from, v_to in zip(from_list, to_list):
            if skip_stale and v_from.stale:
                continue  # Skip stale variable values.
            if skip_fixed and v_to.is_fixed():
                continue  # Skip fixed variables.
            try:
                v_to.set_value(self.get_values(
                    opt._pyomo_var_to_solver_var_map[v_from]))
                if skip_stale:
                    v_to.stale = False
            except ValueError as err:
                err_msg = getattr(err, 'message', str(err))
                # get the value of current feasible solution
                # self.get_value() is an inherent function from Cplex
                var_val = self.get_values(
                    opt._pyomo_var_to_solver_var_map[v_from])
                rounded_val = int(round(var_val))
                # Check to see if this is just a tolerance issue
                if ignore_integrality \
                    and ('is not in domain Binary' in err_msg
                         or 'is not in domain Integers' in err_msg):
                    v_to.value = self.get_values(
                        opt._pyomo_var_to_solver_var_map[v_from])
                elif 'is not in domain Binary' in err_msg and (
                        fabs(var_val - 1) <= config.integer_tolerance or
                        fabs(var_val) <= config.integer_tolerance):
                    v_to.set_value(rounded_val)
                # TODO What about PositiveIntegers etc?
                elif 'is not in domain Integers' in err_msg and (
                        fabs(var_val - rounded_val) <= config.integer_tolerance):
                    v_to.set_value(rounded_val)
                # Value is zero, but shows up as slightly less than zero.
                elif 'is not in domain NonNegativeReals' in err_msg and (
                        fabs(var_val) <= config.zero_tolerance):
                    v_to.set_value(0)
                else:
                    raise

    def add_lazy_oa_cuts(self, target_model, dual_values, solve_data, config, opt,
                         linearize_active=True,
                         linearize_violated=True,
                         linearize_inactive=False,
                         use_slack_var=False):
        """Add oa_cuts through Cplex inherent function self.add()"""

        for (constr, dual_value) in zip(target_model.MindtPy_utils.constraint_list,
                                        dual_values):
            if constr.body.polynomial_degree() in (0, 1):
                continue

            constr_vars = list(identify_variables(constr.body))
            jacs = solve_data.jacobians

            # Equality constraint (makes the problem nonconvex)
            if constr.has_ub() and constr.has_lb() and constr.upper == constr.lower:
                sign_adjust = -1 if solve_data.objective_sense == minimize else 1
                rhs = ((0 if constr.upper is None else constr.upper)
                       + (0 if constr.lower is None else constr.lower))
                rhs = constr.lower if constr.has_lb() and constr.has_ub() else rhs

                # since the cplex requires the lazy cuts in cplex type, we need to transform the pyomo expression into cplex expression
                pyomo_expr = copysign(1, sign_adjust * dual_value) * (sum(value(jacs[constr][var]) * (
                    var - value(var)) for var in list(EXPR.identify_variables(constr.body))) + value(constr.body) - rhs)
                cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                         sense="L",
                         rhs=cplex_rhs)
            else:  # Inequality constraint (possibly two-sided)
                if constr.has_ub() \
                    and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) \
                        or (linearize_violated and constr.uslack() < 0) \
                        or (linearize_inactive and constr.uslack() > 0):

                    pyomo_expr = sum(
                        value(jacs[constr][var])*(var - var.value) for var in constr_vars) + value(constr.body)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                             sense="L",
                             rhs=constr.upper.value+cplex_rhs)
                if constr.has_lb() \
                    and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) \
                        or (linearize_violated and constr.lslack() < 0) \
                        or (linearize_inactive and constr.lslack() > 0):
                    pyomo_expr = sum(value(jacs[constr][var]) * (var - self.get_values(
                        opt._pyomo_var_to_solver_var_map[var])) for var in constr_vars) + value(constr.body)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                             sense="G",
                             rhs=constr.lower.value + cplex_rhs)

    def handle_lazy_master_mip_feasible_sol(self, master_mip, solve_data, config, opt):
        """ This function is called during the branch and bound of master mip, more exactly when a feasible solution is found and LazyCallback is activated.
        Copy the result to working model and update upper or lower bound
        In LP-NLP, upper or lower bound are updated during solving the master problem
        """
        # proceed. Just need integer values
        MindtPy = master_mip.MindtPy_utils
        main_objective = next(
            master_mip.component_data_objects(Objective, active=True))

        # this value copy is useful since we need to fix subproblem based on the solution of the master problem
        self.copy_lazy_var_list_values(opt,
                                       master_mip.MindtPy_utils.variable_list,
                                       solve_data.working_model.MindtPy_utils.variable_list,
                                       config)
        # update the bound
        if main_objective.sense == minimize:
            solve_data.LB = max(
                self.get_objective_value(),
                # self.get_best_objective_value(),
                solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                self.get_objective_value(),
                # self.get_best_objective_value(),
                solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
               solve_data.LB, solve_data.UB))

    def handle_lazy_NLP_subproblem_optimal(self, fix_nlp, solve_data, config, opt):
        """Copies result to mip(explaination see below), updates bound, adds OA and integer cut,
        stores best solution if new one is best"""
        for c in fix_nlp.tmp_duals:
            if fix_nlp.dual.get(c, None) is None:
                fix_nlp.dual[c] = fix_nlp.tmp_duals[c]
        dual_values = list(fix_nlp.dual[c]
                           for c in fix_nlp.MindtPy_utils.constraint_list)

        main_objective = next(
            fix_nlp.component_data_objects(Objective, active=True))
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

        if config.strategy == 'OA':
            # In OA algorithm, OA cuts are generated based on the solution of the subproblem
            # We need to first copy the value of variables from the subproblem and then add cuts
            # since value(constr.body), value(jacs[constr][var]), value(var) are used in self.add_lazy_oa_cuts()
            copy_var_list_values(fix_nlp.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config)
            self.add_lazy_oa_cuts(
                solve_data.mip, dual_values, solve_data, config, opt)

    def handle_lazy_NLP_subproblem_infeasible(self, fix_nlp, solve_data, config, opt):
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
        dual_values = list(fix_nlp.dual[c]
                           for c in fix_nlp.MindtPy_utils.constraint_list)

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
                # config.initial_feas = False
                feas_NLP, feas_NLP_results = solve_NLP_feas(solve_data, config)
                # In OA algorithm, OA cuts are generated based on the solution of the subproblem
                # We need to first copy the value of variables from the subproblem and then add cuts
                copy_var_list_values(feas_NLP.MindtPy_utils.variable_list,
                                     solve_data.mip.MindtPy_utils.variable_list,
                                     config)
                self.add_lazy_oa_cuts(
                    solve_data.mip, dual_values, solve_data, config, opt)

    def __call__(self):
        solve_data = self.solve_data
        config = self.config
        opt = self.opt
        master_mip = self.master_mip
        cpx = opt._solver_model  # Cplex model

        self.handle_lazy_master_mip_feasible_sol(
            master_mip, solve_data, config, opt)

        # solve subproblem
        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        fix_nlp, fix_nlp_result = solve_NLP_subproblem(solve_data, config)

        # add oa cuts
        if fix_nlp_result.solver.termination_condition is tc.optimal:
            self.handle_lazy_NLP_subproblem_optimal(
                fix_nlp, solve_data, config, opt)
        elif fix_nlp_result.solver.termination_condition is tc.infeasible:
            self.handle_lazy_NLP_subproblem_infeasible(
                fix_nlp, solve_data, config, opt)
        else:
            # TODO
            pass


def solve_OA_master(solve_data, config):
    solve_data.mip_iter += 1
    MindtPy = solve_data.mip.MindtPy_utils
    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    # Set up MILP
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()

    MindtPy.MindtPy_linear_cuts.activate()
    main_objective = next(
        solve_data.mip.component_data_objects(Objective, active=True))
    main_objective.deactivate()

    sign_adjust = 1 if main_objective.sense == minimize else -1
    if MindtPy.find_component('MindtPy_oa_obj') is not None:
        del MindtPy.MindtPy_oa_obj

    if config.add_slack == True:
        if MindtPy.find_component('MindtPy_penalty_expr') is not None:
            del MindtPy.MindtPy_penalty_expr

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
    getattr(solve_data.mip, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(solve_data.mip, 'ipopt_zU_out', _DoNothing()).deactivate()

    # with SuppressInfeasibleWarning():
    masteropt = SolverFactory(config.mip_solver)
    # determine if persistent solver is called.
    if isinstance(masteropt, PersistentSolver):
        masteropt.set_instance(solve_data.mip, symbolic_solver_labels=True)
    if config.single_tree == True:
        # Configuration of lazy callback
        lazyoa = masteropt._solver_model.register_callback(LazyOACallback)
        # pass necessary data and parameters to lazyoa
        lazyoa.master_mip = solve_data.mip
        lazyoa.solve_data = solve_data
        lazyoa.config = config
        lazyoa.opt = masteropt
        masteropt._solver_model.set_warning_stream(None)
        masteropt._solver_model.set_log_stream(None)
        masteropt._solver_model.set_error_stream(None)
        masteropt.options['timelimit'] = config.time_limit
    master_mip_results = masteropt.solve(
        solve_data.mip, **config.mip_solver_args)  # , tee=True)

    if master_mip_results.solver.termination_condition is tc.optimal:
        if config.single_tree == True:
            if main_objective.sense == minimize:
                solve_data.LB = max(
                    master_mip_results.problem.lower_bound, solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)

                solve_data.UB = min(
                    master_mip_results.problem.upper_bound, solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)

    elif master_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        master_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
            solve_data.mip, config)

    return solve_data.mip, master_mip_results


def handle_master_mip_optimal(master_mip, solve_data, config, copy=True):
    """Copy the result to working model and update upper or lower bound"""
    # proceed. Just need integer values
    MindtPy = master_mip.MindtPy_utils
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
    # check if the value of binary variable is valid
    for var in MindtPy.variable_list:
        if var.value == None:
            config.logger.warning(
                "Variables {} not initialized are set to it's lower bound when using the initial_binary initialization method".format(var.name))
            var.value = 0  # nlp_var.bounds[0]
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
        config.logger.warning(
            'MindtPy initialization may have generated poor '
            'quality cuts.')
    # set optimistic bound to infinity
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
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
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
    MindtPy.objective_bound = Constraint(
        expr=(-config.obj_bound, main_objective.expr, config.obj_bound))
    with SuppressInfeasibleWarning():
        opt = SolverFactory(config.mip_solver)
        if isinstance(opt, PersistentSolver):
            opt.set_instance(master_mip)
        master_mip_results = opt.solve(
            master_mip, **config.mip_solver_args)
