from __future__ import division


from pyomo.core import Constraint, Expression, Objective, minimize, value, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal, handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination, solve_NLP_feas)
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables
from math import copysign
from pyomo.environ import *
from pyomo.core.expr import current as EXPR
from math import fabs
from pyomo.repn import generate_standard_repn
import logging
from pyomo.common.dependencies import attempt_import
import cplex
from cplex.callbacks import LazyConstraintCallback


class LazyOACallback_cplex(LazyConstraintCallback):
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
                v_val = self.get_values(
                    opt._pyomo_var_to_solver_var_map[v_from])
                v_to.set_value(v_val)
                if skip_stale:
                    v_to.stale = False
            except ValueError:
                # Snap the value to the bounds
                if v_to.has_lb() and v_val < v_to.lb and v_to.lb - v_val <= config.zero_tolerance:
                    v_to.set_value(v_to.lb)
                elif v_to.has_ub() and v_val > v_to.ub and v_val - v_to.ub <= config.zero_tolerance:
                    v_to.set_value(v_to.ub)
                # ... or the nearest integer
                elif v_to.is_integer():
                    rounded_val = int(round(v_val))
                    if (ignore_integrality or fabs(v_val - rounded_val) <= config.integer_tolerance) \
                            and rounded_val in v_to.domain:
                        v_to.set_value(rounded_val)
                else:
                    raise

    def add_lazy_oa_cuts(self, target_model, dual_values, solve_data, config, opt,
                         linearize_active=True,
                         linearize_violated=True,
                         linearize_inactive=False):
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

    def handle_lazy_NLP_subproblem_optimal(self, fixed_nlp, solve_data, config, opt):
        """Copies result to mip(explaination see below), updates bound, adds OA and integer cut,
        stores best solution if new one is best"""
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

        if config.strategy == 'OA':
            # In OA algorithm, OA cuts are generated based on the solution of the subproblem
            # We need to first copy the value of variables from the subproblem and then add cuts
            # since value(constr.body), value(jacs[constr][var]), value(var) are used in self.add_lazy_oa_cuts()
            copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config)
            self.add_lazy_oa_cuts(
                solve_data.mip, dual_values, solve_data, config, opt)

    def handle_lazy_NLP_subproblem_infeasible(self, fixed_nlp, solve_data, config, opt):
        """Solve feasibility problem, add cut according to strategy.

        The solution of the feasibility problem is copied to the working model.
        """
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        for c in fixed_nlp.component_data_objects(ctype=Constraint):
            rhs = ((0 if c.upper is None else c.upper)
                   + (0 if c.lower is None else c.lower))
            sign_adjust = 1 if value(c.upper) is None else -1
            fixed_nlp.dual[c] = (sign_adjust
                                 * max(0, sign_adjust * (rhs - value(c.body))))
        dual_values = list(fixed_nlp.dual[c]
                           for c in fixed_nlp.MindtPy_utils.constraint_list)

        if config.strategy == 'OA':
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

    def handle_lazy_NLP_subproblem_other_termination(self, fixed_nlp, termination_condition,
                                                     solve_data, config):
        """Case that fix-NLP is neither optimal nor infeasible (i.e. max_iterations)"""
        if termination_condition is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial value?
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.')
            var_values = list(
                v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(termination_condition))

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
        fixed_nlp, fixed_nlp_result = solve_NLP_subproblem(solve_data, config)

        # add oa cuts
        if fixed_nlp_result.solver.termination_condition is tc.optimal:
            self.handle_lazy_NLP_subproblem_optimal(
                fixed_nlp, solve_data, config, opt)
        elif fixed_nlp_result.solver.termination_condition is tc.infeasible:
            self.handle_lazy_NLP_subproblem_infeasible(
                fixed_nlp, solve_data, config, opt)
        else:
            self.handle_lazy_NLP_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                              solve_data, config)
