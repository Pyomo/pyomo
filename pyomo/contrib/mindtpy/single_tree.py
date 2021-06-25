#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division
from pyomo.core import Constraint, minimize, value, maximize, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem, solve_feasibility_subproblem
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables, get_main_elapsed_time, time_code
from pyomo.contrib.mindtpy.util import get_integer_solution
from math import copysign
import pyomo.environ as pyo
from pyomo.core.expr import current as EXPR
from math import fabs
from pyomo.repn import generate_standard_repn
import logging
import cplex
from cplex.callbacks import LazyConstraintCallback
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.opt.results import ProblemSense
from pyomo.contrib.mindtpy.mip_solve import handle_main_optimal, solve_main
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts


logger = logging.getLogger('pyomo.contrib.mindtpy')


class LazyOACallback_cplex(LazyConstraintCallback):
    """Inherent class in Cplex to call Lazy callback."""

    def copy_lazy_var_list_values(self, opt, from_list, to_list, config,
                                  skip_stale=False, skip_fixed=True,
                                  ignore_integrality=False):
        """This function copies variable values from one list to another.
        Rounds to Binary/Integer if neccessary
        Sets to zero for NonNegativeReals if neccessary

        Parameters
        ----------
        opt: SolverFactory
            the mip solver
        from_list: variable list
            contains variables and their values
        to_list: variable list
            contains the variables that need to set value
        config: ConfigBlock
            contains the specific configurations for the algorithm
        """
        for v_from, v_to in zip(from_list, to_list):
            if skip_stale and v_from.stale:
                continue  # Skip stale variable values.
            if skip_fixed and v_to.is_fixed():
                continue  # Skip fixed variables.
            v_val = self.get_values(
                opt._pyomo_var_to_solver_var_map[v_from])
            try:
                v_to.set_value(v_val)
                if skip_stale:
                    v_to.stale = False
            except ValueError:
                # Snap the value to the bounds
                if v_to.has_lb() and v_val < v_to.lb and v_to.lb - v_val <= config.variable_tolerance:
                    v_to.set_value(v_to.lb)
                elif v_to.has_ub() and v_val > v_to.ub and v_val - v_to.ub <= config.variable_tolerance:
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
                         linearize_violated=True):
        """
        Linearizes nonlinear constraints; add the OA cuts through Cplex inherent function self.add()
        For nonconvex problems, turn on 'config.add_slack'. Slack variables will
        always be used for nonlinear equality constraints.
        Parameters
        ----------
        target_model:
            this is the MIP/MILP model for the OA algorithm; we want to add the OA cuts to 'target_model'
        dual_values:
            contains the value of the duals for each constraint
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        opt: SolverFactory
            the mip solver
        linearize_active: bool, optional
            this parameter acts as a Boolean flag that signals whether the linearized constraint is active
        linearize_violated: bool, optional
            this parameter acts as a Boolean flag that signals whether the nonlinear constraint represented by the
            linearized constraint has been violated
        """

        config.logger.info('Adding OA cuts')
        with time_code(solve_data.timing, 'OA cut generation'):
            for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
                if constr.body.polynomial_degree() in {0, 1}:
                    continue

                constr_vars = list(identify_variables(constr.body))
                jacs = solve_data.jacobians

                # Equality constraint (makes the problem nonconvex)
                if constr.has_ub() and constr.has_lb() and value(constr.lower) == value(constr.upper):
                    sign_adjust = -1 if solve_data.objective_sense == minimize else 1
                    rhs = constr.lower

                    # since the cplex requires the lazy cuts in cplex type, we need to transform the pyomo expression into cplex expression
                    pyomo_expr = copysign(1, sign_adjust * dual_values[index]) * (sum(value(jacs[constr][var]) * (
                        var - value(var)) for var in EXPR.identify_variables(constr.body)) + value(constr.body) - rhs)
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                             sense='L',
                             rhs=cplex_rhs)
                else:  # Inequality constraint (possibly two-sided)
                    if (constr.has_ub()
                        and (linearize_active and abs(constr.uslack()) < config.zero_tolerance)
                            or (linearize_violated and constr.uslack() < 0)
                            or (config.linearize_inactive and constr.uslack() > 0)) or (constr.name == 'MindtPy_utils.objective_constr' and constr.has_ub()):

                        pyomo_expr = sum(
                            value(jacs[constr][var])*(var - var.value) for var in constr_vars) + value(constr.body)
                        cplex_rhs = - \
                            generate_standard_repn(pyomo_expr).constant
                        cplex_expr, _ = opt._get_expr_from_pyomo_expr(
                            pyomo_expr)
                        self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                                 sense='L',
                                 rhs=value(constr.upper) + cplex_rhs)
                    if (constr.has_lb()
                        and (linearize_active and abs(constr.lslack()) < config.zero_tolerance)
                            or (linearize_violated and constr.lslack() < 0)
                            or (config.linearize_inactive and constr.lslack() > 0)) or (constr.name == 'MindtPy_utils.objective_constr' and constr.has_lb()):
                        pyomo_expr = sum(value(jacs[constr][var]) * (var - self.get_values(
                            opt._pyomo_var_to_solver_var_map[var])) for var in constr_vars) + value(constr.body)
                        cplex_rhs = - \
                            generate_standard_repn(pyomo_expr).constant
                        cplex_expr, _ = opt._get_expr_from_pyomo_expr(
                            pyomo_expr)
                        self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients),
                                 sense='G',
                                 rhs=value(constr.lower) + cplex_rhs)

    def add_lazy_affine_cuts(self, solve_data, config, opt):
        """
        Adds affine cuts using MCPP; add affine cuts through Cplex inherent function self.add()

        Parameters
        ----------
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        opt: SolverFactory
            the mip solver
        """
        with time_code(solve_data.timing, 'Affine cut generation'):
            m = solve_data.mip
            config.logger.info('Adding affine cuts')
            counter = 0

            for constr in m.MindtPy_utils.nonlinear_constraint_list:

                vars_in_constr = list(
                    identify_variables(constr.body))
                if any(var.value is None for var in vars_in_constr):
                    continue  # a variable has no values

                # mcpp stuff
                try:
                    mc_eqn = mc(constr.body)
                except MCPP_Error as e:
                    config.logger.debug(
                        'Skipping constraint %s due to MCPP error %s' % (constr.name, str(e)))
                    continue  # skip to the next constraint
                # TODO: check if the value of ccSlope and cvSlope is not Nan or inf. If so, we skip this.
                ccSlope = mc_eqn.subcc()
                cvSlope = mc_eqn.subcv()
                ccStart = mc_eqn.concave()
                cvStart = mc_eqn.convex()

                concave_cut_valid = True
                convex_cut_valid = True
                for var in vars_in_constr:
                    if not var.fixed:
                        if ccSlope[var] == float('nan') or ccSlope[var] == float('inf'):
                            concave_cut_valid = False
                        if cvSlope[var] == float('nan') or cvSlope[var] == float('inf'):
                            convex_cut_valid = False
                if ccStart == float('nan') or ccStart == float('inf'):
                    concave_cut_valid = False
                if cvStart == float('nan') or cvStart == float('inf'):
                    convex_cut_valid = False
                # check if the value of ccSlope and cvSlope all equals zero. if so, we skip this.
                if not any(ccSlope.values()):
                    concave_cut_valid = False
                if not any(cvSlope.values()):
                    convex_cut_valid = False
                if not (concave_cut_valid or convex_cut_valid):
                    continue

                ub_int = min(value(constr.upper), mc_eqn.upper()
                             ) if constr.has_ub() else mc_eqn.upper()
                lb_int = max(value(constr.lower), mc_eqn.lower()
                             ) if constr.has_lb() else mc_eqn.lower()

                if concave_cut_valid:
                    pyomo_concave_cut = sum(ccSlope[var] * (var - var.value)
                                            for var in vars_in_constr
                                            if not var.fixed) + ccStart
                    cplex_concave_rhs = generate_standard_repn(
                        pyomo_concave_cut).constant
                    cplex_concave_cut, _ = opt._get_expr_from_pyomo_expr(
                        pyomo_concave_cut)
                    self.add(constraint=cplex.SparsePair(ind=cplex_concave_cut.variables, val=cplex_concave_cut.coefficients),
                             sense='G',
                             rhs=lb_int - cplex_concave_rhs)
                    counter += 1
                if convex_cut_valid:
                    pyomo_convex_cut = sum(cvSlope[var] * (var - var.value)
                                           for var in vars_in_constr
                                           if not var.fixed) + cvStart
                    cplex_convex_rhs = generate_standard_repn(
                        pyomo_convex_cut).constant
                    cplex_convex_cut, _ = opt._get_expr_from_pyomo_expr(
                        pyomo_convex_cut)
                    self.add(constraint=cplex.SparsePair(ind=cplex_convex_cut.variables, val=cplex_convex_cut.coefficients),
                             sense='L',
                             rhs=ub_int - cplex_convex_rhs)
                    counter += 1

            config.logger.info('Added %s affine cuts' % counter)

    def add_lazy_no_good_cuts(self, var_values, solve_data, config, opt, feasible=False):
        """
        Adds no-good cuts; add the no-good cuts through Cplex inherent function self.add()

        Parameters
        ----------
        var_values: list
            values of the current variables, used to generate the cut
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        feasible: bool, optional
            boolean indicating if integer combination yields a feasible or infeasible NLP
        opt: SolverFactory
            the mip solver
        """
        if not config.add_no_good_cuts:
            return

        config.logger.info('Adding no-good cuts')
        with time_code(solve_data.timing, 'No-good cut generation'):
            m = solve_data.mip
            MindtPy = m.MindtPy_utils
            int_tol = config.integer_tolerance

            binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]

            # copy variable values over
            for var, val in zip(MindtPy.variable_list, var_values):
                if not var.is_binary():
                    continue
                var.value = val

            # check to make sure that binary variables are all 0 or 1
            for v in binary_vars:
                if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                    raise ValueError('Binary {} = {} is not 0 or 1'.format(
                        v.name, value(v)))

            if not binary_vars:  # if no binary variables, skip
                return

            pyomo_no_good_cut = sum(1 - v for v in binary_vars if value(abs(v - 1))
                                    <= int_tol) + sum(v for v in binary_vars if value(abs(v)) <= int_tol)
            cplex_no_good_rhs = generate_standard_repn(
                pyomo_no_good_cut).constant
            cplex_no_good_cut, _ = opt._get_expr_from_pyomo_expr(
                pyomo_no_good_cut)

            self.add(constraint=cplex.SparsePair(ind=cplex_no_good_cut.variables, val=cplex_no_good_cut.coefficients),
                     sense='G',
                     rhs=1 - cplex_no_good_rhs)

    def handle_lazy_main_feasible_solution(self, main_mip, solve_data, config, opt):
        """ This function is called during the branch and bound of main mip, more exactly when a feasible solution is found and LazyCallback is activated.
        Copy the result to working model and update upper or lower bound.
        In LP-NLP, upper or lower bound are updated during solving the main problem

        Parameters
        ----------
        main_mip: Pyomo model
            the MIP main problem
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        opt: SolverFactory
            the mip solver
        """
        # proceed. Just need integer values

        # this value copy is useful since we need to fix subproblem based on the solution of the main problem
        self.copy_lazy_var_list_values(opt,
                                       main_mip.MindtPy_utils.variable_list,
                                       solve_data.working_model.MindtPy_utils.variable_list,
                                       config)
        if solve_data.objective_sense == minimize:
            solve_data.LB = max(
                self.get_best_objective_value(), solve_data.LB)
            solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                self.get_best_objective_value(), solve_data.UB)
            solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ (at current node): %s  Bound: %s  LB: %s  UB: %s  TIME: %s'
            % (solve_data.mip_iter, self.get_objective_value(), self.get_best_objective_value(),
                solve_data.LB, solve_data.UB, round(get_main_elapsed_time(
                    solve_data.timing), 2)))

    def handle_lazy_subproblem_optimal(self, fixed_nlp, solve_data, config, opt):
        """
        This function copies  result to mip(explaination see below), updates bound, adds OA and no-good cuts,
        stores best solution if new one is best

        Parameters
        ----------
        fixed_nlp: Pyomo model
            Fixed-NLP from the model
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        opt: SolverFactory
            the mip solver
        """
        if config.calculate_dual:
            for c in fixed_nlp.tmp_duals:
                if fixed_nlp.dual.get(c, None) is None:
                    fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
            dual_values = list(fixed_nlp.dual[c]
                               for c in fixed_nlp.MindtPy_utils.constraint_list)
        else:
            dual_values = None
        main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
        if solve_data.objective_sense == minimize:
            solve_data.UB = min(value(main_objective.expr), solve_data.UB)
            solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)
        else:
            solve_data.LB = max(value(main_objective.expr), solve_data.LB)
            solve_data.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)

        config.logger.info(
            'Fixed-NLP {}: OBJ: {}  LB: {}  UB: {}  TIME: {}'
            .format(solve_data.nlp_iter, value(main_objective.expr), solve_data.LB, solve_data.UB, round(get_main_elapsed_time(solve_data.timing), 2)))

        if solve_data.solution_improved:
            solve_data.best_solution_found = fixed_nlp.clone()
            solve_data.best_solution_found_time = get_main_elapsed_time(
                solve_data.timing)
            if config.add_no_good_cuts or config.use_tabu_list:
                if solve_data.results.problem.sense == ProblemSense.minimize:
                    solve_data.stored_bound.update(
                        {solve_data.UB: solve_data.LB})
                else:
                    solve_data.stored_bound.update(
                        {solve_data.LB: solve_data.UB})

        # In OA algorithm, OA cuts are generated based on the solution of the subproblem
        # We need to first copy the value of variables from the subproblem and then add cuts
        # since value(constr.body), value(jacs[constr][var]), value(var) are used in self.add_lazy_oa_cuts()
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        if config.strategy == 'OA':
            self.add_lazy_oa_cuts(
                solve_data.mip, dual_values, solve_data, config, opt)
            if config.add_regularization is not None:
                add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
        elif config.strategy == 'GOA':
            self.add_lazy_affine_cuts(solve_data, config, opt)
        if config.add_no_good_cuts:
            var_values = list(
                v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            self.add_lazy_no_good_cuts(var_values, solve_data, config, opt)

    def handle_lazy_subproblem_infeasible(self, fixed_nlp, solve_data, config, opt):
        """
        Solves feasibility problem and adds cut according to the specified strategy

        Parameters
        ----------
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        opt: SolverFactory
            the mip solver
        """
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        solve_data.nlp_infeasible_counter += 1
        if config.calculate_dual:
            for c in fixed_nlp.component_data_objects(ctype=Constraint):
                rhs = ((0 if c.upper is None else c.upper)
                       + (0 if c.lower is None else c.lower))
                sign_adjust = 1 if c.upper is None else -1
                fixed_nlp.dual[c] = (sign_adjust
                                     * max(0, sign_adjust * (rhs - value(c.body))))
            dual_values = list(fixed_nlp.dual[c]
                               for c in fixed_nlp.MindtPy_utils.constraint_list)
        else:
            dual_values = None

        config.logger.info('Solving feasibility problem')
        feas_subproblem, feas_subproblem_results = solve_feasibility_subproblem(
            solve_data, config)
        # In OA algorithm, OA cuts are generated based on the solution of the subproblem
        # We need to first copy the value of variables from the subproblem and then add cuts
        copy_var_list_values(feas_subproblem.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        if config.strategy == 'OA':
            self.add_lazy_oa_cuts(
                solve_data.mip, dual_values, solve_data, config, opt)
            if config.add_regularization is not None:
                add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
        elif config.strategy == 'GOA':
            self.add_lazy_affine_cuts(solve_data, config, opt)
        if config.add_no_good_cuts:
            var_values = list(
                v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            self.add_lazy_no_good_cuts(var_values, solve_data, config, opt)

    def handle_lazy_subproblem_other_termination(self, fixed_nlp, termination_condition,
                                                 solve_data, config):
        """
        Handles the result of the latest iteration of solving the NLP subproblem given a solution that is neither optimal
        nor infeasible.

        Parameters
        ----------
        termination_condition: Pyomo TerminationCondition
            the termination condition of the NLP subproblem
        solve_data: MindtPy Data Container
            data container that holds solve-instance data
        config: ConfigBlock
            contains the specific configurations for the algorithm
        """
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

    def handle_lazy_regularization_problem(self, main_mip, main_mip_results, solve_data, config):
        if main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
            handle_main_optimal(
                main_mip, solve_data, config, update_bound=False)
        elif main_mip_results.solver.termination_condition in {tc.infeasible, tc.infeasibleOrUnbounded}:
            config.logger.info('regularization problem infeasible.')
            if config.reduce_level_coef:
                config.level_coef = config.level_coef / 2
                main_mip, main_mip_results = solve_main(
                    solve_data, config, regularization_problem=True)
                if main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
                    handle_main_optimal(
                        main_mip, solve_data, config, update_bound=False)
                elif main_mip_results.solver.termination_condition is tc.infeasible:
                    config.logger.info('regularization problem still infeasible with reduced level_coef. '
                                       'NLP subproblem is generated based on the incumbent solution of the main problem.')
                elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
                    config.logger.info(
                        'Regularization problem failed to converge within the time limit.')
                    solve_data.results.solver.termination_condition = tc.maxTimeLimit
                elif main_mip_results.solver.termination_condition is tc.unbounded:
                    config.logger.info(
                        'Regularization problem ubounded.'
                        'Sometimes solving MIQP using cplex, unbounded means infeasible.')
                elif main_mip_results.solver.termination_condition is tc.unknown:
                    config.logger.info(
                        'Termination condition of the regularization problem is unknown.')
                    if main_mip_results.problem.lower_bound != float('-inf'):
                        config.logger.info('Solution limit has been reached.')
                        handle_main_optimal(
                            main_mip, solve_data, config, update_bound=False)
                    else:
                        config.logger.info('No solution obtained from the regularization subproblem.'
                                           'Please set mip_solver_tee to True for more informations.'
                                           'The solution of the OA main problem will be adopted.')
                else:
                    raise ValueError(
                        'MindtPy unable to handle regularization problem termination condition '
                        'of %s. Solver message: %s' %
                        (main_mip_results.solver.termination_condition, main_mip_results.solver.message))
            elif config.use_bb_tree_incumbent:
                config.logger.info(
                    'Fixed subproblem will be generated based on the incumbent solution of the main problem.')
        elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.info(
                'Regularization problem failed to converge within the time limit.')
            solve_data.results.solver.termination_condition = tc.maxTimeLimit
        elif main_mip_results.solver.termination_condition is tc.unbounded:
            config.logger.info(
                'Regularization problem ubounded.'
                'Sometimes solving MIQP using cplex, unbounded means infeasible.')
        elif main_mip_results.solver.termination_condition is tc.unknown:
            config.logger.info(
                'Termination condition of the regularization problem is unknown.')
            if main_mip_results.problem.lower_bound != float('-inf'):
                config.logger.info('Solution limit has been reached.')
                handle_main_optimal(main_mip, solve_data,
                                    config, update_bound=False)
        else:
            raise ValueError(
                'MindtPy unable to handle regularization problem termination condition '
                'of %s. Solver message: %s' %
                (main_mip_results.solver.termination_condition, main_mip_results.solver.message))

    def __call__(self):
        """
        This is an inherent function in LazyConstraintCallback in cplex. 
        This function is called whenever an integer solution is found during the branch and bound process
        """
        solve_data = self.solve_data
        config = self.config
        opt = self.opt
        main_mip = self.main_mip

        if solve_data.should_terminate:
            self.abort()
            return

        self.handle_lazy_main_feasible_solution(
            main_mip, solve_data, config, opt)

        if config.add_cuts_at_incumbent:
            self.copy_lazy_var_list_values(opt,
                                           main_mip.MindtPy_utils.variable_list,
                                           solve_data.mip.MindtPy_utils.variable_list,
                                           config)
            if config.strategy == 'OA':
                self.add_lazy_oa_cuts(
                    solve_data.mip, None, solve_data, config, opt)

        # regularization is activated after the first feasible solution is found.
        if config.add_regularization is not None and solve_data.best_solution_found is not None:
            # the main problem might be unbounded, regularization is activated only when a valid bound is provided.
            if not solve_data.bound_improved and not solve_data.solution_improved:
                config.logger.info('the bound and the best found solution have neither been improved.'
                                   'We will skip solving the regularization problem and the Fixed-NLP subproblem')
                solve_data.solution_improved = False
                return
            if ((solve_data.objective_sense == minimize and solve_data.LB != float('-inf'))
                    or (solve_data.objective_sense == maximize and solve_data.UB != float('inf'))):
                main_mip, main_mip_results = solve_main(
                    solve_data, config, regularization_problem=True)
                self.handle_lazy_regularization_problem(
                    main_mip, main_mip_results, solve_data, config)

        if solve_data.LB + config.bound_tolerance >= solve_data.UB:
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                'LB: {} + (tol {}) >= UB: {}\n'.format(
                    solve_data.LB, config.bound_tolerance, solve_data.UB))
            solve_data.results.solver.termination_condition = tc.optimal
            self.abort()
            return

        # check if the same integer combination is obtained.
        solve_data.curr_int_sol = get_integer_solution(
            solve_data.working_model, string_zero=True)

        if solve_data.curr_int_sol in set(solve_data.integer_list):
            config.logger.info('This integer combination has been explored. '
                               'We will skip solving the Fixed-NLP subproblem.')
            solve_data.solution_improved = False
            if config.strategy == 'GOA':
                if config.add_no_good_cuts:
                    var_values = list(
                        v.value for v in solve_data.working_model.MindtPy_utils.variable_list)
                    self.add_lazy_no_good_cuts(
                        var_values, solve_data, config, opt)
                return
            elif config.strategy == 'OA':
                return
        else:
            solve_data.integer_list.append(solve_data.curr_int_sol)

        # solve subproblem
        # The constraint linearization happens in the handlers
        fixed_nlp, fixed_nlp_result = solve_subproblem(solve_data, config)

        # add oa cuts
        if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            self.handle_lazy_subproblem_optimal(
                fixed_nlp, solve_data, config, opt)
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                config.logger.info(
                    'MindtPy exiting on bound convergence. '
                    'LB: {} + (tol {}) >= UB: {}\n'.format(
                        solve_data.LB, config.bound_tolerance, solve_data.UB))
                solve_data.results.solver.termination_condition = tc.optimal
                return
        elif fixed_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
            self.handle_lazy_subproblem_infeasible(
                fixed_nlp, solve_data, config, opt)
        else:
            self.handle_lazy_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                          solve_data, config)
