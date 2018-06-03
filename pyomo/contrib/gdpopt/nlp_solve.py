"""Functions for solving the nonlinear subproblem
in Logic-based outer approximation.
"""
from __future__ import division

from math import fabs

from pyomo.core import TerminationCondition as tc
from pyomo.core import (Block, Constraint, SolverFactory,
                        TransformationFactory, Var, value, minimize)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverStatus


def _solve_NLP_subproblem(solve_data, config):
    # print('Clone working model for NLP')
    m = solve_data.working_model.clone()
    GDPopt = m.GDPopt_utils
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries and '
                       'logical realizations.'
                       % (solve_data.nlp_iter,))
    # Fix binary variables
    binary_vars = [
        v for v in m.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v.is_binary() and not v.fixed]
    for v in binary_vars:
        if v.value is None:
            config.logger.warning(
                'No value is defined for binary variable %s'
                ' for the NLP subproblem.' % (v.name,))
        else:
            # round the integer variable values so that they are exactly 0
            # or 1
            if config.round_NLP_binaries:
                v.set_value(round(v.value))
        v.fix()

    # Deactivate the OA and PSC cuts
    for constr in m.component_objects(ctype=Constraint, active=True,
                                      descend_into=(Block, Disjunct)):
        if (constr.local_name == 'GDPopt_OA_cuts' or
                constr.local_name == 'psc_cuts'):
            constr.deactivate()

    # Activate or deactivate disjuncts according to the value of their
    # indicator variable
    for disj in m.component_data_objects(
            ctype=Disjunct, descend_into=(Block, Disjunct)):
        if (fabs(value(disj.indicator_var) - 1)
                <= config.integer_tolerance):
            # Disjunct is active. Convert to Block.
            disj.parent_block().reclassify_component_type(disj, Block)
        elif (fabs(value(disj.indicator_var))
                <= config.integer_tolerance):
            disj.deactivate()
        else:
            raise ValueError(
                'Non-binary value of disjunct indicator variable '
                'for %s: %s' % (disj.name, value(disj.indicator_var)))

    for d in m.component_data_objects(Disjunction, active=True):
        d.deactivate()

    # Propagate variable bounds
    TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)
    # Detect fixed variables
    TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
    # Propagate fixed variables
    TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
    # Remove zero terms in linear expressions
    TransformationFactory('contrib.remove_zero_terms').apply_to(m)
    # Remove terms in equal to zero summations
    TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
    # Transform bound constraints
    TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
    # Detect fixed variables
    TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
    # Remove terms in equal to zero summations
    TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
    # Remove trivial constraints
    TransformationFactory(
        'contrib.deactivate_trivial_constraints').apply_to(m)

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

    config.subprob_presolve(m, solve_data)

    # Solve the NLP
    nlp_solver = SolverFactory(config.nlp)
    if not nlp_solver.available():
        raise RuntimeError("NLP solver %s is not available." % config.nlp)
    results = nlp_solver.solve(
        m, load_solutions=False,
        **config.nlp_options)
    solve_data.solve_results = results

    solnFeasible = False

    def process_feasible_solution():
        self._copy_values(m, solve_data.working_model, config)
        self._copy_dual_suffixes(m, solve_data.working_model)
        if GDPopt.objective.sense == minimize:
            solve_data.UB = min(
                value(GDPopt.objective.expr), solve_data.UB)
            solve_data.solution_improved = (
                solve_data.UB < solve_data.UB_progress[-1])
        else:
            solve_data.LB = max(
                value(GDPopt.objective.expr), solve_data.LB)
            solve_data.solution_improved = (
                solve_data.LB > solve_data.LB_progress[-1])
        config.logger.info(
            'NLP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.nlp_iter,
               value(GDPopt.objective.expr),
               solve_data.LB, solve_data.UB))
        if solve_data.solution_improved:
            # print('Clone model for best_solution_found')
            solve_data.best_solution_found = m.clone()

        # Add the linear cut
        if solve_data.current_strategy == 'LOA':
            self._add_oa_cut(m, solve_data, config)

        # This adds an integer cut to the GDPopt_feasible_integer_cuts
        # ConstraintList, which is not activated by default. However, it
        # may be activated as needed in certain situations or for certain
        # values of option flags.
        self._add_int_cut(solve_data, config, feasible=True)

        config.subprob_postfeas(m, solve_data)

    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        m.solutions.load_from(results)
        process_feasible_solution()
        solnFeasible = True
    elif subprob_terminate_cond is tc.infeasible:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        results.solver.status = SolverStatus.ok
        m.solutions.load_from(results)
        self._copy_values(m, solve_data.working_model)
        self._copy_dual_suffixes(m, solve_data.working_model)
        # Add an integer cut to exclude this discrete option
        self._add_int_cut(solve_data)
    elif subprob_terminate_cond is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        results.solver.status = SolverStatus.ok
        m.solutions.load_from(results)
        if self._is_feasible(m):
            config.logger.info(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
            process_feasible_solution()
            solnFeasible = True
        else:
            # Add an integer cut to exclude this discrete option
            self._add_int_cut(solve_data)
    else:
        raise ValueError(
            'GDPopt unable to handle NLP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

    if GDPopt.objective.sense == minimize:
        solve_data.UB_progress.append(solve_data.UB)
    else:
        solve_data.LB_progress.append(solve_data.LB)

    # Call the NLP post-solve callback
    config.subprob_postsolve(m, solve_data)
    return solnFeasible
