#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Functions for solving the nonlinear subproblem."""
from __future__ import division

from math import fabs

from pyomo.common.collections import ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.gdpopt.data_class import SubproblemResult
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning,
                                       is_feasible, get_main_elapsed_time)
from pyomo.core import Constraint, TransformationFactory, minimize, value, Objective
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc


def solve_disjunctive_subproblem(mip_result, solve_data, config):
    """Set up and solve the disjunctive subproblem."""
    if config.force_subproblem_nlp:
        if config.strategy == "LOA":
            return solve_local_NLP(mip_result.var_values, solve_data, config)
        elif config.strategy == 'GLOA':
            return solve_global_subproblem(mip_result, solve_data, config)
    else:
        if config.strategy == "LOA":
            return solve_local_subproblem(mip_result, solve_data, config)
        elif config.strategy == 'GLOA':
            return solve_global_subproblem(mip_result, solve_data, config)


def solve_linear_subproblem(mip_model, solve_data, config):
    GDPopt = mip_model.GDPopt_utils

    initialize_subproblem(mip_model, solve_data)

    # Callback immediately before solving NLP subproblem
    config.call_before_subproblem_solve(mip_model, solve_data)

    mip_solver = SolverFactory(config.mip_solver)
    if not mip_solver.available():
        raise RuntimeError("MIP solver %s is not available." % config.mip_solver)
    with SuppressInfeasibleWarning():
        mip_args = dict(config.mip_solver_args)
        elapsed = get_main_elapsed_time(solve_data.timing)
        remaining = max(config.time_limit - elapsed, 1)
        if config.mip_solver == 'gams':
            mip_args['add_options'] = mip_args.get('add_options', [])
            mip_args['add_options'].append('option reslim=%s;' % remaining)
        elif config.mip_solver == 'multisolve':
            mip_args['time_limit'] = min(mip_args.get('time_limit', float('inf')), remaining)
        results = mip_solver.solve(mip_model, **mip_args)

    subprob_result = SubproblemResult()
    subprob_result.feasible = True
    subprob_result.var_values = list(v.value for v in GDPopt.variable_list)
    subprob_result.pyomo_results = results
    subprob_result.dual_values = list(mip_model.dual.get(c, None) for c in GDPopt.constraint_list)

    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        pass
    elif subprob_terminate_cond is tc.infeasible:
        config.logger.info('MIP subproblem was infeasible.')
        subprob_result.feasible = False
    else:
        raise ValueError(
            'GDPopt unable to handle MIP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(mip_model, solve_data)

    # if feasible, call the NLP post-feasible callback
    if subprob_result.feasible:
        config.call_after_subproblem_feasible(mip_model, solve_data)

    return subprob_result


def solve_NLP(nlp_model, solve_data, config):
    """Solve the NLP subproblem."""
    config.logger.info(
        'Solving nonlinear subproblem for '
        'fixed binaries and logical realizations.')

    # Error checking for unfixed discrete variables
    unfixed_discrete_vars = detect_unfixed_discrete_vars(nlp_model)
    assert len(unfixed_discrete_vars) == 0, \
        "Unfixed discrete variables exist on the NLP subproblem: {0}".format(
        list(v.name for v in unfixed_discrete_vars))

    GDPopt = nlp_model.GDPopt_utils

    initialize_subproblem(nlp_model, solve_data)

    # Callback immediately before solving NLP subproblem
    config.call_before_subproblem_solve(nlp_model, solve_data)

    nlp_solver = SolverFactory(config.nlp_solver)
    if not nlp_solver.available():
        raise RuntimeError("NLP solver %s is not available." %
                           config.nlp_solver)
    with SuppressInfeasibleWarning():
        try:
            nlp_args = dict(config.nlp_solver_args)
            elapsed = get_main_elapsed_time(solve_data.timing)
            remaining = max(config.time_limit - elapsed, 1)
            if config.nlp_solver == 'gams':
                nlp_args['add_options'] = nlp_args.get('add_options', [])
                nlp_args['add_options'].append('option reslim=%s;' % remaining)
            elif config.nlp_solver == 'multisolve':
                nlp_args['time_limit'] = min(nlp_args.get('time_limit', float('inf')), remaining)
            results = nlp_solver.solve(nlp_model, **nlp_args)
        except ValueError as err:
            if 'Cannot load a SolverResults object with bad status: error' in str(err):
                results = SolverResults()
                results.solver.termination_condition = tc.error
                results.solver.message = str(err)
            else:
                raise

    nlp_result = SubproblemResult()
    nlp_result.feasible = True
    nlp_result.var_values = list(v.value for v in GDPopt.variable_list)
    nlp_result.pyomo_results = results
    nlp_result.dual_values = list(
        nlp_model.dual.get(c, None)
        for c in GDPopt.constraint_list)

    term_cond = results.solver.termination_condition
    if any(term_cond == cond for cond in (tc.optimal, tc.locallyOptimal, tc.feasible)):
        pass
    elif term_cond == tc.infeasible:
        config.logger.info('NLP subproblem was infeasible.')
        nlp_result.feasible = False
    elif term_cond == tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        if is_feasible(nlp_model, config):
            config.logger.info(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
        else:
            nlp_result.feasible = False
    elif term_cond == tc.internalSolverError:
        # Possible that IPOPT had a restoration failure
        config.logger.info(
            "NLP solver had an internal failure: %s" % results.solver.message)
        nlp_result.feasible = False
    elif (term_cond == tc.other and
          "Too few degrees of freedom" in str(results.solver.message)):
        # Possible IPOPT degrees of freedom error
        config.logger.info(
            "IPOPT has too few degrees of freedom: %s" %
            results.solver.message)
        nlp_result.feasible = False
    elif term_cond == tc.other:
        config.logger.info(
            "NLP solver had a termination condition of 'other': %s" %
            results.solver.message)
        nlp_result.feasible = False
    elif term_cond == tc.error:
        config.logger.info("NLP solver had a termination condition of 'error': %s" % results.solver.message)
        nlp_result.feasible = False
    elif term_cond == tc.maxTimeLimit:
        config.logger.info("NLP solver ran out of time. Assuming infeasible for now.")
        nlp_result.feasible = False
    else:
        raise ValueError(
            'GDPopt unable to handle NLP subproblem termination '
            'condition of %s. Results: %s'
            % (term_cond, results))

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(nlp_model, solve_data)

    # if feasible, call the NLP post-feasible callback
    if nlp_result.feasible:
        config.call_after_subproblem_feasible(nlp_model, solve_data)

    return nlp_result


def solve_MINLP(model, solve_data, config):
    """Solve the MINLP subproblem."""
    config.logger.info(
        "Solving MINLP subproblem for fixed logical realizations."
    )

    GDPopt = model.GDPopt_utils

    initialize_subproblem(model, solve_data)

    # Callback immediately before solving MINLP subproblem
    config.call_before_subproblem_solve(model, solve_data)

    minlp_solver = SolverFactory(config.minlp_solver)
    if not minlp_solver.available():
        raise RuntimeError("MINLP solver %s is not available." %
                           config.minlp_solver)
    with SuppressInfeasibleWarning():
        minlp_args = dict(config.minlp_solver_args)
        elapsed = get_main_elapsed_time(solve_data.timing)
        remaining = max(config.time_limit - elapsed, 1)
        if config.minlp_solver == 'gams':
            minlp_args['add_options'] = minlp_args.get('add_options', [])
            minlp_args['add_options'].append('option reslim=%s;' % remaining)
        elif config.minlp_solver == 'multisolve':
            minlp_args['time_limit'] = min(minlp_args.get('time_limit', float('inf')), remaining)
        results = minlp_solver.solve(model, **minlp_args)

    subprob_result = SubproblemResult()
    subprob_result.feasible = True
    subprob_result.var_values = list(v.value for v in GDPopt.variable_list)
    subprob_result.pyomo_results = results
    subprob_result.dual_values = list(
        model.dual.get(c, None)
        for c in GDPopt.constraint_list)

    term_cond = results.solver.termination_condition
    if any(term_cond == cond for cond in (tc.optimal, tc.locallyOptimal, tc.feasible)):
        pass
    elif term_cond == tc.infeasible:
        config.logger.info('MINLP subproblem was infeasible.')
        subprob_result.feasible = False
    elif term_cond == tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info(
            'MINLP subproblem failed to converge within iteration limit.')
        if is_feasible(model, config):
            config.logger.info(
                'MINLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
        else:
            subprob_result.feasible = False
    elif term_cond == tc.maxTimeLimit:
        config.logger.info('MINLP subproblem failed to converge within time limit.')
        if is_feasible(model, config):
            config.logger.info(
                'MINLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
        else:
            subprob_result.feasible = False
    elif term_cond == tc.intermediateNonInteger:
        config.logger.info(
            "MINLP solver could not find feasible integer solution: %s" % results.solver.message)
        subprob_result.feasible = False
    else:
        raise ValueError(
            'GDPopt unable to handle MINLP subproblem termination '
            'condition of %s. Results: %s'
            % (term_cond, results))

    # Call the subproblem post-solve callback
    config.call_after_subproblem_solve(model, solve_data)

    # if feasible, call the subproblem post-feasible callback
    if subprob_result.feasible:
        config.call_after_subproblem_feasible(model, solve_data)

    return subprob_result


def detect_unfixed_discrete_vars(model):
    """Detect unfixed discrete variables in use on the model."""
    var_set = ComponentSet()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=True):
        var_set.update(
            v for v in EXPR.identify_variables(
                constr.body, include_fixed=False)
            if not v.is_continuous())
    for obj in model.component_data_objects(Objective, active=True):
        var_set.update(v for v in EXPR.identify_variables(obj.expr, include_fixed=False)
                       if not v.is_continuous())
    return var_set


def preprocess_subproblem(m, config):
    """Applies preprocessing transformations to the model."""
    # fbbt(m, integer_tol=config.integer_tolerance)
    xfrm = TransformationFactory
    xfrm('contrib.propagate_eq_var_bounds').apply_to(m)
    xfrm('contrib.detect_fixed_vars').apply_to(
        m, tolerance=config.variable_tolerance)
    xfrm('contrib.propagate_fixed_vars').apply_to(m)
    xfrm('contrib.remove_zero_terms').apply_to(m)
    xfrm('contrib.propagate_zero_sum').apply_to(m)
    xfrm('contrib.constraints_to_var_bounds').apply_to(
        m, tolerance=config.variable_tolerance)
    xfrm('contrib.detect_fixed_vars').apply_to(
        m, tolerance=config.variable_tolerance)
    xfrm('contrib.propagate_zero_sum').apply_to(m)
    xfrm('contrib.deactivate_trivial_constraints').apply_to(
        m, tolerance=config.constraint_tolerance)


def initialize_subproblem(model, solve_data):
    """Perform initialization of the subproblem.

    Presently, this just restores the continuous variables to the original model values.

    """
    # restore original continuous variable values
    for var, old_value in zip(model.GDPopt_utils.variable_list,
                              solve_data.initial_var_values):
        if not var.fixed and var.is_continuous():
            if old_value is not None:
                # Adjust value if it falls outside the bounds
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                # Set the value
                var.set_value(old_value)


def update_subproblem_progress_indicators(solved_model, solve_data, config):
    """Update the progress indicators for the subproblem."""
    GDPopt = solved_model.GDPopt_utils
    objective = next(solved_model.component_data_objects(Objective, active=True))
    if objective.sense == minimize:
        old_UB = solve_data.UB
        solve_data.UB = min(value(objective.expr), solve_data.UB)
        solve_data.feasible_solution_improved = (solve_data.UB < old_UB)
    else:
        old_LB = solve_data.LB
        solve_data.LB = max(value(objective.expr), solve_data.LB)
        solve_data.feasible_solution_improved = (solve_data.LB > old_LB)
    solve_data.iteration_log[
        (solve_data.master_iteration,
         solve_data.mip_iteration,
         solve_data.nlp_iteration)
    ] = (
        value(objective.expr),
        value(objective.expr),
        [v.value for v in GDPopt.variable_list]
    )

    if solve_data.feasible_solution_improved:
        solve_data.best_solution_found = solved_model.clone()

    improvement_tag = (
        "(IMPROVED) " if solve_data.feasible_solution_improved else "")
    lb_improved, ub_improved = (
        ("", improvement_tag)
        if objective.sense == minimize
        else (improvement_tag, ""))
    config.logger.info(
        'ITER {:d}.{:d}.{:d}-NLP: OBJ: {:.10g}  LB: {:.10g} {:s} UB: {:.10g} {:s}'.format(
            solve_data.master_iteration,
            solve_data.mip_iteration,
            solve_data.nlp_iteration,
            value(objective.expr),
            solve_data.LB, lb_improved,
            solve_data.UB, ub_improved))


def solve_local_NLP(mip_var_values, solve_data, config):
    """Set up and solve the local LOA subproblem."""
    nlp_model = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1
    # copy in the discrete variable values
    for var, val in zip(nlp_model.GDPopt_utils.variable_list, mip_var_values):
        if val is None:
            continue
        if var.is_continuous():
            var.value = val
        elif ((fabs(val) > config.integer_tolerance and
               fabs(val - 1) > config.integer_tolerance)):
            raise ValueError(
                "Binary variable %s value %s is not "
                "within tolerance %s of 0 or 1." %
                (var.name, var.value, config.integer_tolerance))
        else:
            # variable is binary and within tolerances
            if config.round_discrete_vars:
                var.fix(int(round(val)))
            else:
                var.fix(val)
    TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)

    nlp_result = solve_NLP(nlp_model, solve_data, config)
    if nlp_result.feasible:  # NLP is feasible
        update_subproblem_progress_indicators(nlp_model, solve_data, config)
    return nlp_result


def solve_local_subproblem(mip_result, solve_data, config):
    """Set up and solve the local MINLP or NLP subproblem."""
    subprob = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1

    # TODO also copy over the variable values?

    for disj, val in zip(subprob.GDPopt_utils.disjunct_list,
                         mip_result.disjunct_values):
        rounded_val = int(round(val))
        if (fabs(val - rounded_val) > config.integer_tolerance or
                rounded_val not in (0, 1)):
            raise ValueError(
                "Disjunct %s indicator value %s is not "
                "within tolerance %s of 0 or 1." %
                (disj.name, val.value, config.integer_tolerance)
            )
        else:
            if config.round_discrete_vars:
                disj.indicator_var.fix(rounded_val)
            else:
                disj.indicator_var.fix(val)

    if config.force_subproblem_nlp:
        # We also need to copy over the discrete variable values
        for var, val in zip(subprob.GDPopt_utils.variable_list,
                            mip_result.var_values):
            if var.is_continuous():
                continue
            rounded_val = int(round(val))
            if fabs(val - rounded_val) > config.integer_tolerance:
                raise ValueError(
                    "Discrete variable %s value %s is not "
                    "within tolerance %s of %s." %
                    (var.name, var.value, config.integer_tolerance, rounded_val))
            else:
                # variable is binary and within tolerances
                if config.round_discrete_vars:
                    var.fix(rounded_val)
                else:
                    var.fix(val)

    TransformationFactory('gdp.fix_disjuncts').apply_to(subprob)

    # for disj in subprob.component_data_objects(Disjunct, active=True):
    #     disj.deactivate()  # TODO this is a HACK for something that isn't happening correctly in fix_disjuncts

    if config.subproblem_presolve:
        try:
            preprocess_subproblem(subprob, config)
        except InfeasibleConstraintException:
            return get_infeasible_result_object(
                subprob, "Preprocessing determined problem to be infeasible.")

    if not any(constr.body.polynomial_degree() not in (1, 0)
               for constr in subprob.component_data_objects(Constraint, active=True)):
        subprob_result = solve_linear_subproblem(subprob, solve_data, config)
    else:
        unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
        if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
            raise RuntimeError("Unfixed discrete variables found on the NLP subproblem.")
        elif len(unfixed_discrete_vars) == 0:
            subprob_result = solve_NLP(subprob, solve_data, config)
        else:
            subprob_result = solve_MINLP(subprob, solve_data, config)

    if subprob_result.feasible:  # subproblem is feasible
        update_subproblem_progress_indicators(subprob, solve_data, config)
    return subprob_result


def solve_global_subproblem(mip_result, solve_data, config):
    subprob = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1

    # copy in the discrete variable values
    for disj, val in zip(subprob.GDPopt_utils.disjunct_list,
                         mip_result.disjunct_values):
        rounded_val = int(round(val))
        if (fabs(val - rounded_val) > config.integer_tolerance or
                rounded_val not in (0, 1)):
            raise ValueError(
                "Disjunct %s indicator value %s is not "
                "within tolerance %s of 0 or 1." %
                (disj.name, val.value, config.integer_tolerance)
            )
        else:
            if config.round_discrete_vars:
                disj.indicator_var.fix(rounded_val)
            else:
                disj.indicator_var.fix(val)

    if config.force_subproblem_nlp:
        # We also need to copy over the discrete variable values
        for var, val in zip(subprob.GDPopt_utils.variable_list,
                            mip_result.var_values):
            if var.is_continuous():
                continue
            rounded_val = int(round(val))
            if fabs(val - rounded_val) > config.integer_tolerance:
                raise ValueError(
                    "Discrete variable %s value %s is not "
                    "within tolerance %s of %s." %
                    (var.name, var.value, config.integer_tolerance, rounded_val))
            else:
                # variable is binary and within tolerances
                if config.round_discrete_vars:
                    var.fix(rounded_val)
                else:
                    var.fix(val)

    TransformationFactory('gdp.fix_disjuncts').apply_to(subprob)
    subprob.dual.deactivate()  # global solvers may not give dual info

    if config.subproblem_presolve:
        try:
            preprocess_subproblem(subprob, config)
        except InfeasibleConstraintException as e:
            # FBBT found the problem to be infeasible
            return get_infeasible_result_object(
                subprob, "Preprocessing determined problem to be infeasible.")

    unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
    if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
        raise RuntimeError("Unfixed discrete variables found on the NLP subproblem.")
    elif len(unfixed_discrete_vars) == 0:
        subprob_result = solve_NLP(subprob, solve_data, config)
    else:
        subprob_result = solve_MINLP(subprob, solve_data, config)
    if subprob_result.feasible:  # NLP is feasible
        update_subproblem_progress_indicators(subprob, solve_data, config)
    return subprob_result


def get_infeasible_result_object(model, message=""):
    infeas_result = SubproblemResult()
    infeas_result.feasible = False
    infeas_result.var_values = list(v.value for v in model.GDPopt_utils.variable_list)
    infeas_result.pyomo_results = SolverResults()
    infeas_result.pyomo_results.solver.termination_condition = tc.infeasible
    infeas_result.pyomo_results.message = message
    infeas_result.dual_values = list(None for _ in model.GDPopt_utils.constraint_list)
    return infeas_result
