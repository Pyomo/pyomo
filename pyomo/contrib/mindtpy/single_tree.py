#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_no_good_cuts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from math import copysign
from pyomo.contrib.mindtpy.util import get_integer_solution
from pyomo.contrib.gdpopt.util import (
    copy_var_list_values,
    get_main_elapsed_time,
    time_code,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.core import minimize, value
from pyomo.core.expr import identify_variables

cplex, cplex_available = attempt_import('cplex')


class LazyOACallback_cplex(
    cplex.callbacks.LazyConstraintCallback if cplex_available else object
):
    """Inherent class in CPLEX to call Lazy callback."""

    def copy_lazy_var_list_values(
        self,
        opt,
        from_list,
        to_list,
        config,
        skip_stale=False,
        skip_fixed=True,
        ignore_integrality=False,
    ):
        """This function copies variable values from one list to another.

        Rounds to Binary/Integer if necessary.
        Sets to zero for NonNegativeReals if necessary.

        Parameters
        ----------
        opt : SolverFactory
            The cplex_persistent solver.
        from_list : list
            The variables that provides the values to copy from.
        to_list : list
            The variables that need to set value.
        config : ConfigBlock
            The specific configurations for MindtPy.
        skip_stale : bool, optional
            Whether to skip the stale variables, by default False.
        skip_fixed : bool, optional
            Whether to skip the fixed variables, by default True.
        ignore_integrality : bool, optional
            Whether to ignore the integrality of integer variables, by default False.
        """
        for v_from, v_to in zip(from_list, to_list):
            if skip_stale and v_from.stale:
                continue  # Skip stale variable values.
            if skip_fixed and v_to.is_fixed():
                continue  # Skip fixed variables.
            v_val = self.get_values(opt._pyomo_var_to_solver_var_map[v_from])
            try:
                # We don't want to trigger the reset of the global stale
                # indicator, so we will set this variable to be "stale",
                # knowing that set_value will switch it back to "not
                # stale"
                v_to.stale = True
                # NOTE: PEP 2180 changes the var behavior so that domain
                # / bounds violations no longer generate exceptions (and
                # instead log warnings).  This means that the following
                # will always succeed and the ValueError should never be
                # raised.
                v_to.set_value(v_val, skip_validation=True)
            except ValueError as e:
                # Snap the value to the bounds
                config.logger.error(e)
                if (
                    v_to.has_lb()
                    and v_val < v_to.lb
                    and v_to.lb - v_val <= config.variable_tolerance
                ):
                    v_to.set_value(v_to.lb, skip_validation=True)
                elif (
                    v_to.has_ub()
                    and v_val > v_to.ub
                    and v_val - v_to.ub <= config.variable_tolerance
                ):
                    v_to.set_value(v_to.ub, skip_validation=True)
                # ... or the nearest integer
                elif v_to.is_integer():
                    rounded_val = int(round(v_val))
                    if (
                        ignore_integrality
                        or abs(v_val - rounded_val) <= config.integer_tolerance
                    ) and rounded_val in v_to.domain:
                        v_to.set_value(rounded_val, skip_validation=True)
                else:
                    raise

    def add_lazy_oa_cuts(
        self,
        target_model,
        dual_values,
        mindtpy_solver,
        config,
        opt,
        linearize_active=True,
        linearize_violated=True,
    ):
        """Linearizes nonlinear constraints; add the OA cuts through CPLEX inherent function self.add()
        For nonconvex problems, turn on 'config.add_slack'. Slack variables will always be used for
        nonlinear equality constraints.

        Parameters
        ----------
        target_model : Pyomo model
            The MIP main problem.
        dual_values : list
            The value of the duals for each constraint.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        linearize_active : bool, optional
            Whether to linearize the active nonlinear constraints, by default True.
        linearize_violated : bool, optional
            Whether to linearize the violated nonlinear constraints, by default True.
        """
        config.logger.debug('Adding OA cuts')
        with time_code(mindtpy_solver.timing, 'OA cut generation'):
            for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
                if (
                    constr.body.polynomial_degree()
                    in mindtpy_solver.mip_constraint_polynomial_degree
                ):
                    continue

                constr_vars = list(identify_variables(constr.body))
                jacs = mindtpy_solver.jacobians

                # Equality constraint (makes the problem nonconvex)
                if (
                    constr.has_ub()
                    and constr.has_lb()
                    and value(constr.lower) == value(constr.upper)
                ):
                    sign_adjust = (
                        -1 if mindtpy_solver.objective_sense == minimize else 1
                    )
                    rhs = constr.lower

                    # Since CPLEX requires the lazy cuts in CPLEX type,
                    # we need to transform the pyomo expression into CPLEX expression.
                    pyomo_expr = copysign(1, sign_adjust * dual_values[index]) * (
                        sum(
                            value(jacs[constr][var]) * (var - value(var))
                            for var in EXPR.identify_variables(constr.body)
                        )
                        + value(constr.body)
                        - rhs
                    )
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    self.add(
                        constraint=cplex.SparsePair(
                            ind=cplex_expr.variables, val=cplex_expr.coefficients
                        ),
                        sense='L',
                        rhs=cplex_rhs,
                    )
                    if (
                        self.get_solution_source()
                        == cplex.callbacks.SolutionSource.mipstart_solution
                    ):
                        mindtpy_solver.mip_start_lazy_oa_cuts.append(
                            [
                                cplex.SparsePair(
                                    ind=cplex_expr.variables,
                                    val=cplex_expr.coefficients,
                                ),
                                'L',
                                cplex_rhs,
                            ]
                        )
                else:  # Inequality constraint (possibly two-sided)
                    if (
                        constr.has_ub()
                        and (
                            linearize_active
                            and abs(constr.uslack()) < config.zero_tolerance
                        )
                        or (linearize_violated and constr.uslack() < 0)
                        or (config.linearize_inactive and constr.uslack() > 0)
                    ) or (
                        'MindtPy_utils.objective_constr' in constr.name
                        and constr.has_ub()
                    ):
                        pyomo_expr = sum(
                            value(jacs[constr][var]) * (var - var.value)
                            for var in constr_vars
                        ) + value(constr.body)
                        cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                        cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                        self.add(
                            constraint=cplex.SparsePair(
                                ind=cplex_expr.variables, val=cplex_expr.coefficients
                            ),
                            sense='L',
                            rhs=value(constr.upper) + cplex_rhs,
                        )
                        if (
                            self.get_solution_source()
                            == cplex.callbacks.SolutionSource.mipstart_solution
                        ):
                            mindtpy_solver.mip_start_lazy_oa_cuts.append(
                                [
                                    cplex.SparsePair(
                                        ind=cplex_expr.variables,
                                        val=cplex_expr.coefficients,
                                    ),
                                    'L',
                                    value(constr.upper) + cplex_rhs,
                                ]
                            )
                    if (
                        constr.has_lb()
                        and (
                            linearize_active
                            and abs(constr.lslack()) < config.zero_tolerance
                        )
                        or (linearize_violated and constr.lslack() < 0)
                        or (config.linearize_inactive and constr.lslack() > 0)
                    ) or (
                        'MindtPy_utils.objective_constr' in constr.name
                        and constr.has_lb()
                    ):
                        pyomo_expr = sum(
                            value(jacs[constr][var])
                            * (
                                var
                                - self.get_values(opt._pyomo_var_to_solver_var_map[var])
                            )
                            for var in constr_vars
                        ) + value(constr.body)
                        cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                        cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                        self.add(
                            constraint=cplex.SparsePair(
                                ind=cplex_expr.variables, val=cplex_expr.coefficients
                            ),
                            sense='G',
                            rhs=value(constr.lower) + cplex_rhs,
                        )
                        if (
                            self.get_solution_source()
                            == cplex.callbacks.SolutionSource.mipstart_solution
                        ):
                            mindtpy_solver.mip_start_lazy_oa_cuts.append(
                                [
                                    cplex.SparsePair(
                                        ind=cplex_expr.variables,
                                        val=cplex_expr.coefficients,
                                    ),
                                    'G',
                                    value(constr.lower) + cplex_rhs,
                                ]
                            )

    def add_lazy_affine_cuts(self, mindtpy_solver, config, opt):
        """Adds affine cuts using MCPP.

        Add affine cuts through CPLEX inherent function self.add().

        Parameters
        ----------
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
        with time_code(mindtpy_solver.timing, 'Affine cut generation'):
            m = mindtpy_solver.mip
            config.logger.debug('Adding affine cuts')
            counter = 0

            for constr in m.MindtPy_utils.nonlinear_constraint_list:
                vars_in_constr = list(identify_variables(constr.body))
                if any(var.value is None for var in vars_in_constr):
                    continue  # a variable has no values

                # mcpp stuff
                try:
                    mc_eqn = mc(constr.body)
                except MCPP_Error as e:
                    config.logger.debug(
                        'Skipping constraint %s due to MCPP error %s'
                        % (constr.name, str(e))
                    )
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

                ub_int = (
                    min(value(constr.upper), mc_eqn.upper())
                    if constr.has_ub()
                    else mc_eqn.upper()
                )
                lb_int = (
                    max(value(constr.lower), mc_eqn.lower())
                    if constr.has_lb()
                    else mc_eqn.lower()
                )

                if concave_cut_valid:
                    pyomo_concave_cut = (
                        sum(
                            ccSlope[var] * (var - var.value)
                            for var in vars_in_constr
                            if not var.fixed
                        )
                        + ccStart
                    )
                    cplex_concave_rhs = generate_standard_repn(
                        pyomo_concave_cut
                    ).constant
                    cplex_concave_cut, _ = opt._get_expr_from_pyomo_expr(
                        pyomo_concave_cut
                    )
                    self.add(
                        constraint=cplex.SparsePair(
                            ind=cplex_concave_cut.variables,
                            val=cplex_concave_cut.coefficients,
                        ),
                        sense='G',
                        rhs=lb_int - cplex_concave_rhs,
                    )
                    counter += 1
                if convex_cut_valid:
                    pyomo_convex_cut = (
                        sum(
                            cvSlope[var] * (var - var.value)
                            for var in vars_in_constr
                            if not var.fixed
                        )
                        + cvStart
                    )
                    cplex_convex_rhs = generate_standard_repn(pyomo_convex_cut).constant
                    cplex_convex_cut, _ = opt._get_expr_from_pyomo_expr(
                        pyomo_convex_cut
                    )
                    self.add(
                        constraint=cplex.SparsePair(
                            ind=cplex_convex_cut.variables,
                            val=cplex_convex_cut.coefficients,
                        ),
                        sense='L',
                        rhs=ub_int - cplex_convex_rhs,
                    )
                    counter += 1

            config.logger.debug('Added %s affine cuts' % counter)

    def add_lazy_no_good_cuts(
        self, var_values, mindtpy_solver, config, opt, feasible=False
    ):
        """Adds no-good cuts.

        Add the no-good cuts through Cplex inherent function self.add().

        Parameters
        ----------
        var_values : list
            The variable values of the incumbent solution, used to generate the cut.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        feasible : bool, optional
            Whether the integer combination yields a feasible or infeasible NLP, by default False.

        Raises
        ------
        ValueError
            The value of binary variable is not 0 or 1.
        """
        if not config.add_no_good_cuts:
            return

        config.logger.debug('Adding no-good cuts')
        with time_code(mindtpy_solver.timing, 'No-good cut generation'):
            m = mindtpy_solver.mip
            MindtPy = m.MindtPy_utils
            int_tol = config.integer_tolerance

            binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]

            # copy variable values over
            for var, val in zip(MindtPy.variable_list, var_values):
                if not var.is_binary():
                    continue
                # We don't want to trigger the reset of the global stale
                # indicator, so we will set this variable to be "stale",
                # knowing that set_value will switch it back to "not
                # stale"
                var.stale = True
                var.set_value(val, skip_validation=True)

            # check to make sure that binary variables are all 0 or 1
            for v in binary_vars:
                if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                    raise ValueError(
                        'Binary {} = {} is not 0 or 1'.format(v.name, value(v))
                    )

            if not binary_vars:  # if no binary variables, skip
                return

            pyomo_no_good_cut = sum(
                1 - v for v in binary_vars if value(abs(v - 1)) <= int_tol
            ) + sum(v for v in binary_vars if value(abs(v)) <= int_tol)
            cplex_no_good_rhs = generate_standard_repn(pyomo_no_good_cut).constant
            cplex_no_good_cut, _ = opt._get_expr_from_pyomo_expr(pyomo_no_good_cut)

            self.add(
                constraint=cplex.SparsePair(
                    ind=cplex_no_good_cut.variables, val=cplex_no_good_cut.coefficients
                ),
                sense='G',
                rhs=1 - cplex_no_good_rhs,
            )

    def handle_lazy_main_feasible_solution(self, main_mip, mindtpy_solver, config, opt):
        """This function is called during the branch and bound of main mip, more
        exactly when a feasible solution is found and LazyCallback is activated.
        Copy the result to working model and update upper or lower bound.
        In LP-NLP, upper or lower bound are updated during solving the main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
        # proceed. Just need integer values

        # this value copy is useful since we need to fix subproblem based on the solution of the main problem
        self.copy_lazy_var_list_values(
            opt,
            main_mip.MindtPy_utils.variable_list,
            mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list,
            config,
            skip_fixed=False,
        )
        mindtpy_solver.update_dual_bound(self.get_best_objective_value())
        config.logger.info(
            mindtpy_solver.log_formatter.format(
                mindtpy_solver.mip_iter,
                'restrLP',
                self.get_objective_value(),
                mindtpy_solver.primal_bound,
                mindtpy_solver.dual_bound,
                mindtpy_solver.rel_gap,
                get_main_elapsed_time(mindtpy_solver.timing),
            )
        )

    def handle_lazy_subproblem_optimal(self, fixed_nlp, mindtpy_solver, config, opt):
        """This function copies the optimal solution of the fixed NLP subproblem to the MIP
        main problem(explanation see below), updates bound, adds OA and no-good cuts,
        stores incumbent solution if it has been improved.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.tmp_duals:
                if fixed_nlp.dual.get(c, None) is None:
                    fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
                elif (
                    config.nlp_solver == 'cyipopt'
                    and mindtpy_solver.objective_sense == minimize
                ):
                    # TODO: recover the opposite dual when cyipopt issue #2831 is solved.
                    fixed_nlp.dual[c] = -fixed_nlp.dual[c]
            dual_values = list(
                fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list
            )
        else:
            dual_values = None
        main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
        mindtpy_solver.update_primal_bound(value(main_objective.expr))
        if mindtpy_solver.primal_bound_improved:
            mindtpy_solver.best_solution_found = fixed_nlp.clone()
            mindtpy_solver.best_solution_found_time = get_main_elapsed_time(
                mindtpy_solver.timing
            )
            if config.add_no_good_cuts or config.use_tabu_list:
                mindtpy_solver.stored_bound.update(
                    {mindtpy_solver.primal_bound: mindtpy_solver.dual_bound}
                )
        config.logger.info(
            mindtpy_solver.fixed_nlp_log_formatter.format(
                '*' if mindtpy_solver.primal_bound_improved else ' ',
                mindtpy_solver.nlp_iter,
                'Fixed NLP',
                value(main_objective.expr),
                mindtpy_solver.primal_bound,
                mindtpy_solver.dual_bound,
                mindtpy_solver.rel_gap,
                get_main_elapsed_time(mindtpy_solver.timing),
            )
        )

        # In OA algorithm, OA cuts are generated based on the solution of the subproblem
        # We need to first copy the value of variables from the subproblem and then add cuts
        # since value(constr.body), value(jacs[constr][var]), value(var) are used in self.add_lazy_oa_cuts()
        copy_var_list_values(
            fixed_nlp.MindtPy_utils.variable_list,
            mindtpy_solver.mip.MindtPy_utils.variable_list,
            config,
        )
        if config.strategy == 'OA':
            self.add_lazy_oa_cuts(
                mindtpy_solver.mip, dual_values, mindtpy_solver, config, opt
            )
            if config.add_regularization is not None:
                add_oa_cuts(
                    mindtpy_solver.mip,
                    dual_values,
                    mindtpy_solver.jacobians,
                    mindtpy_solver.objective_sense,
                    mindtpy_solver.mip_constraint_polynomial_degree,
                    mindtpy_solver.mip_iter,
                    config,
                    mindtpy_solver.timing,
                )
        elif config.strategy == 'GOA':
            self.add_lazy_affine_cuts(mindtpy_solver, config, opt)
        if config.add_no_good_cuts:
            var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            self.add_lazy_no_good_cuts(var_values, mindtpy_solver, config, opt)

    def handle_lazy_subproblem_infeasible(self, fixed_nlp, mindtpy_solver, config, opt):
        """Solves feasibility NLP subproblem and adds cuts according to the specified strategy.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        mindtpy_solver.nlp_infeasible_counter += 1
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.MindtPy_utils.constraint_list:
                rhs = (0 if c.upper is None else c.upper) + (
                    0 if c.lower is None else c.lower
                )
                sign_adjust = 1 if c.upper is None else -1
                fixed_nlp.dual[c] = sign_adjust * max(
                    0, sign_adjust * (rhs - value(c.body))
                )
            dual_values = list(
                fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list
            )
        else:
            dual_values = None

        config.logger.info('Solving feasibility problem')
        (
            feas_subproblem,
            feas_subproblem_results,
        ) = mindtpy_solver.solve_feasibility_subproblem()
        # In OA algorithm, OA cuts are generated based on the solution of the subproblem
        # We need to first copy the value of variables from the subproblem and then add cuts
        copy_var_list_values(
            feas_subproblem.MindtPy_utils.variable_list,
            mindtpy_solver.mip.MindtPy_utils.variable_list,
            config,
        )
        if config.strategy == 'OA':
            self.add_lazy_oa_cuts(
                mindtpy_solver.mip, dual_values, mindtpy_solver, config, opt
            )
            if config.add_regularization is not None:
                add_oa_cuts(
                    mindtpy_solver.mip,
                    dual_values,
                    mindtpy_solver.jacobians,
                    mindtpy_solver.objective_sense,
                    mindtpy_solver.mip_constraint_polynomial_degree,
                    mindtpy_solver.mip_iter,
                    config,
                    mindtpy_solver.timing,
                )
        elif config.strategy == 'GOA':
            self.add_lazy_affine_cuts(mindtpy_solver, config, opt)
        if config.add_no_good_cuts:
            var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            self.add_lazy_no_good_cuts(var_values, mindtpy_solver, config, opt)

    def handle_lazy_subproblem_other_termination(
        self, fixed_nlp, termination_condition, mindtpy_solver, config
    ):
        """Handles the result of the latest iteration of solving the NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        termination_condition : Pyomo TerminationCondition
            The termination condition of the fixed NLP subproblem.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the fixed NLP subproblem.
        """
        if termination_condition is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial value?
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.'
            )
            var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(termination_condition)
            )

    def __call__(self):
        """This is an inherent function in LazyConstraintCallback in CPLEX.

        This function is called whenever an integer solution is found during the branch and bound process.
        """
        mindtpy_solver = self.mindtpy_solver
        config = self.config
        opt = self.opt
        main_mip = self.main_mip
        mindtpy_solver = self.mindtpy_solver

        # Reference: https://www.ibm.com/docs/en/icos/22.1.1?topic=SSSA5P_22.1.1/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.SolutionSource-class.htm
        # Another solution source is user_solution = 118, but it will not be encountered in LazyConstraintCallback.
        config.logger.debug(
            "Solution source: %s (111 node_solution, 117 heuristic_solution, 119 mipstart_solution)".format(
                self.get_solution_source()
            )
        )

        # The solution found in MIP start process might be revisited in branch and bound.
        # Lazy constraints separated when processing a MIP start will be discarded after that MIP start has been processed.
        # This means that the callback may have to separate the same constraint again for the next MIP start or for a solution that is found later in the solution process.
        # https://www.ibm.com/docs/en/icos/22.1.1?topic=SSSA5P_22.1.1/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.LazyConstraintCallback-class.htm
        if (
            self.get_solution_source()
            != cplex.callbacks.SolutionSource.mipstart_solution
            and len(mindtpy_solver.mip_start_lazy_oa_cuts) > 0
        ):
            for constraint, sense, rhs in mindtpy_solver.mip_start_lazy_oa_cuts:
                self.add(constraint, sense, rhs)
            mindtpy_solver.mip_start_lazy_oa_cuts = []

        if mindtpy_solver.should_terminate:
            self.abort()
            return
        self.handle_lazy_main_feasible_solution(main_mip, mindtpy_solver, config, opt)
        if config.add_cuts_at_incumbent:
            self.copy_lazy_var_list_values(
                opt,
                main_mip.MindtPy_utils.variable_list,
                mindtpy_solver.mip.MindtPy_utils.variable_list,
                config,
            )
            if config.strategy == 'OA':
                # The solution obtained from mip start might be infeasible and even introduce a math domain error, like log(-1).
                try:
                    self.add_lazy_oa_cuts(
                        mindtpy_solver.mip, None, mindtpy_solver, config, opt
                    )
                except ValueError as e:
                    config.logger.error(
                        str(e)
                        + "\nUsually this error is caused by the MIP start solution causing a math domain error. "
                        "We will skip it."
                    )
                    return

        # regularization is activated after the first feasible solution is found.
        if (
            config.add_regularization is not None
            and mindtpy_solver.best_solution_found is not None
        ):
            # The main problem might be unbounded, regularization is activated only when a valid bound is provided.
            if (
                not mindtpy_solver.dual_bound_improved
                and not mindtpy_solver.primal_bound_improved
            ):
                config.logger.debug(
                    'The bound and the best found solution have neither been improved.'
                    'We will skip solving the regularization problem and the Fixed-NLP subproblem'
                )
                mindtpy_solver.primal_bound_improved = False
                return
            if mindtpy_solver.dual_bound != mindtpy_solver.dual_bound_progress[0]:
                mindtpy_solver.add_regularization()
        if (
            abs(mindtpy_solver.primal_bound - mindtpy_solver.dual_bound)
            <= config.absolute_bound_tolerance
        ):
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                '|Primal Bound: {} - Dual Bound: {}| <= (absolute tolerance {})  \n'.format(
                    mindtpy_solver.primal_bound,
                    mindtpy_solver.dual_bound,
                    config.absolute_bound_tolerance,
                )
            )
            mindtpy_solver.results.solver.termination_condition = tc.optimal
            self.abort()
            return

        # check if the same integer combination is obtained.
        mindtpy_solver.curr_int_sol = get_integer_solution(
            mindtpy_solver.fixed_nlp, string_zero=True
        )

        if mindtpy_solver.curr_int_sol in set(mindtpy_solver.integer_list):
            config.logger.debug(
                'This integer combination has been explored. '
                'We will skip solving the Fixed-NLP subproblem.'
            )
            mindtpy_solver.primal_bound_improved = False
            if config.strategy == 'GOA':
                if config.add_no_good_cuts:
                    var_values = list(
                        v.value
                        for v in mindtpy_solver.working_model.MindtPy_utils.variable_list
                    )
                    self.add_lazy_no_good_cuts(var_values, mindtpy_solver, config, opt)
                return
            elif config.strategy == 'OA':
                return
        else:
            mindtpy_solver.integer_list.append(mindtpy_solver.curr_int_sol)

        # solve subproblem
        # The constraint linearization happens in the handlers
        fixed_nlp, fixed_nlp_result = mindtpy_solver.solve_subproblem()
        # add oa cuts
        if fixed_nlp_result.solver.termination_condition in {
            tc.optimal,
            tc.locallyOptimal,
            tc.feasible,
        }:
            self.handle_lazy_subproblem_optimal(fixed_nlp, mindtpy_solver, config, opt)
            if (
                abs(mindtpy_solver.primal_bound - mindtpy_solver.dual_bound)
                <= config.absolute_bound_tolerance
            ):
                config.logger.info(
                    'MindtPy exiting on bound convergence. '
                    '|Primal Bound: {} - Dual Bound: {}| <= (absolute tolerance {})  \n'.format(
                        mindtpy_solver.primal_bound,
                        mindtpy_solver.dual_bound,
                        config.absolute_bound_tolerance,
                    )
                )
                mindtpy_solver.results.solver.termination_condition = tc.optimal
                return
        elif fixed_nlp_result.solver.termination_condition in {
            tc.infeasible,
            tc.noSolution,
        }:
            self.handle_lazy_subproblem_infeasible(
                fixed_nlp, mindtpy_solver, config, opt
            )
        else:
            self.handle_lazy_subproblem_other_termination(
                fixed_nlp,
                fixed_nlp_result.solver.termination_condition,
                mindtpy_solver,
                config,
            )


# Gurobi


def LazyOACallback_gurobi(cb_m, cb_opt, cb_where, mindtpy_solver, config):
    """This is a Gurobi callback function defined for LP/NLP based B&B algorithm.

    Parameters
    ----------
    cb_m : Pyomo model
        The MIP main problem.
    cb_opt : SolverFactory
        The gurobi_persistent solver.
    cb_where : int
        An enum member of gurobipy.GRB.Callback.
    mindtpy_solver : object
        The mindtpy solver class.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if cb_where == gurobipy.GRB.Callback.MIPSOL:
        # gurobipy.GRB.Callback.MIPSOL means that an integer solution is found during the branch and bound process
        if mindtpy_solver.should_terminate:
            cb_opt._solver_model.terminate()
            return
        cb_opt.cbGetSolution(vars=cb_m.MindtPy_utils.variable_list)
        handle_lazy_main_feasible_solution_gurobi(cb_m, cb_opt, mindtpy_solver, config)

        if config.add_cuts_at_incumbent:
            if config.strategy == 'OA':
                add_oa_cuts(
                    mindtpy_solver.mip,
                    None,
                    mindtpy_solver.jacobians,
                    mindtpy_solver.objective_sense,
                    mindtpy_solver.mip_constraint_polynomial_degree,
                    mindtpy_solver.mip_iter,
                    config,
                    mindtpy_solver.timing,
                    cb_opt=cb_opt,
                )

        # Regularization is activated after the first feasible solution is found.
        if (
            config.add_regularization is not None
            and mindtpy_solver.best_solution_found is not None
        ):
            # The main problem might be unbounded, regularization is activated only when a valid bound is provided.
            if (
                not mindtpy_solver.dual_bound_improved
                and not mindtpy_solver.primal_bound_improved
            ):
                config.logger.debug(
                    'The bound and the best found solution have neither been improved.'
                    'We will skip solving the regularization problem and the Fixed-NLP subproblem'
                )
                mindtpy_solver.primal_bound_improved = False
                return
            if mindtpy_solver.dual_bound != mindtpy_solver.dual_bound_progress[0]:
                mindtpy_solver.add_regularization()

        if (
            abs(mindtpy_solver.primal_bound - mindtpy_solver.dual_bound)
            <= config.absolute_bound_tolerance
        ):
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                '|Primal Bound: {} - Dual Bound: {}| <= (absolute tolerance {})  \n'.format(
                    mindtpy_solver.primal_bound,
                    mindtpy_solver.dual_bound,
                    config.absolute_bound_tolerance,
                )
            )
            mindtpy_solver.results.solver.termination_condition = tc.optimal
            cb_opt._solver_model.terminate()
            return

        # check if the same integer combination is obtained.
        mindtpy_solver.curr_int_sol = get_integer_solution(
            mindtpy_solver.fixed_nlp, string_zero=True
        )

        if mindtpy_solver.curr_int_sol in set(mindtpy_solver.integer_list):
            config.logger.debug(
                'This integer combination has been explored. '
                'We will skip solving the Fixed-NLP subproblem.'
            )
            mindtpy_solver.primal_bound_improved = False
            if config.strategy == 'GOA':
                if config.add_no_good_cuts:
                    var_values = list(
                        v.value
                        for v in mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list
                    )
                    add_no_good_cuts(
                        mindtpy_solver.mip,
                        var_values,
                        config,
                        mindtpy_solver.timing,
                        mip_iter=mindtpy_solver.mip_iter,
                        cb_opt=cb_opt,
                    )
                return
            elif config.strategy == 'OA':
                return
        else:
            mindtpy_solver.integer_list.append(mindtpy_solver.curr_int_sol)

        # solve subproblem
        # The constraint linearization happens in the handlers
        fixed_nlp, fixed_nlp_result = mindtpy_solver.solve_subproblem()

        mindtpy_solver.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, cb_opt)


def handle_lazy_main_feasible_solution_gurobi(cb_m, cb_opt, mindtpy_solver, config):
    """This function is called during the branch and bound of main MIP problem,
    more exactly when a feasible solution is found and LazyCallback is activated.

    Copy the solution to working model and update upper or lower bound.
    In LP-NLP, upper or lower bound are updated during solving the main problem.

    Parameters
    ----------
    cb_m : Pyomo model
        The MIP main problem.
    cb_opt : SolverFactory
        The gurobi_persistent solver.
    mindtpy_solver : object
        The mindtpy solver class.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    # proceed. Just need integer values
    cb_opt.cbGetSolution(vars=cb_m.MindtPy_utils.variable_list)
    # this value copy is useful since we need to fix subproblem based on the solution of the main problem
    copy_var_list_values(
        cb_m.MindtPy_utils.variable_list,
        mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list,
        config,
        skip_fixed=False,
    )
    copy_var_list_values(
        cb_m.MindtPy_utils.variable_list,
        mindtpy_solver.mip.MindtPy_utils.variable_list,
        config,
    )
    mindtpy_solver.update_dual_bound(cb_opt.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJBND))
    config.logger.info(
        mindtpy_solver.log_formatter.format(
            mindtpy_solver.mip_iter,
            'restrLP',
            cb_opt.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ),
            mindtpy_solver.primal_bound,
            mindtpy_solver.dual_bound,
            mindtpy_solver.rel_gap,
            get_main_elapsed_time(mindtpy_solver.timing),
        )
    )
