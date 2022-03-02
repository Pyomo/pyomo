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
from math import fabs

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning,
                                       is_feasible, get_main_elapsed_time)
from pyomo.core import (Constraint, TransformationFactory, minimize, value,
                        Objective, Block)
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.fbbt.fbbt import fbbt

def configure_and_call_solver(model, solver, args, problem_type, timing,
                              time_limit):
    opt = SolverFactory(solver)
    if not opt.available():
        raise RuntimeError("%s solver %s is not available." % (problem_type,
                                                               solver))
    with SuppressInfeasibleWarning():
        solver_args = dict(args)
        elapsed = get_main_elapsed_time(timing)
        remaining = max(time_limit - elapsed, 1)
        if solver == 'gams':
            solver_args['add_options'] = solver_args.get('add_options', [])
            solver_args['add_options'].append('option reslim=%s;' % remaining)
        elif solver == 'multisolve':
            solver_args['time_limit'] = min(solver_args.get('time_limit',
                                                            float('inf')),
                                            remaining)
        try:
            results = opt.solve(model, **solver_args)
        except ValueError as err:
            if 'Cannot load a SolverResults object with bad status: error' in \
               str(err):
                results = SolverResults()
                results.solver.termination_condition = tc.error
                results.solver.message = str(err)
            else:
                raise
    return results

def process_nonlinear_problem_results(results, model, problem_type, config):
    logger = config.logger
    term_cond = results.solver.termination_condition
    if any(term_cond == cond for cond in (tc.optimal, tc.locallyOptimal,
                                          tc.feasible)):
        return True
    elif term_cond == tc.infeasible:
        logger.info('%s subproblem was infeasible.' % problem_type)
        return False
    elif term_cond == tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        logger.info( '%s subproblem failed to converge within iteration limit.'
                     % problem_type)
        if is_feasible(model, config):
            logger.info(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
            return True
        return False
    elif term_cond == tc.internalSolverError:
        # Possible that IPOPT had a restoration failure
        logger.info( "%s solver had an internal failure: %s" % 
                     (problem_type, results.solver.message))
        return False
    elif (term_cond == tc.other and
          "Too few degrees of freedom" in str(results.solver.message)):
        # Possible IPOPT degrees of freedom error
        logger.info(
            "Perhaps the subproblem solver has too few degrees of freedom: %s" %
            results.solver.message)
        return False
    elif term_cond == tc.other:
        logger.info(
            "%s solver had a termination condition of 'other': %s" %
            (problem_type, results.solver.message))
        return False
    elif term_cond == tc.error:
        logger.info("%s solver had a termination condition of 'error': "
                    "%s" % (problem_type, results.solver.message))
        return False
    elif term_cond == tc.maxTimeLimit:
        logger.info("%s subproblem failed to converge within time "
                    "limit." % problem_type)
        if is_feasible(model, config):
            config.logger.info(
                '%s solution is still feasible. '
                'Using potentially suboptimal feasible solution.' % 
                problem_type)
            return True
        return False
    elif term_cond == tc.intermediateNonInteger:
        config.logger.info( "%s solver could not find feasible integer"
                            " solution: %s" % (problem_type, 
                                               results.solver.message))
        return False
    else:
        raise ValueError( 'GDPopt unable to handle %s subproblem termination '
                          'condition of %s. Results: %s' % (problem_type,
                                                            term_cond, results))

def solve_linear_subproblem(subproblem, config, timing):
    results = configure_and_call_solver(subproblem, config.mip_solver,
                                        config.mip_solver_args, 'MIP', timing,
                                        config.time_limit)
    
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        return True
    elif subprob_terminate_cond is tc.infeasible:
        config.logger.info('MIP subproblem was infeasible.')
        return False
    else:
        raise ValueError(
            'GDPopt unable to handle MIP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

def solve_NLP(nlp_model, config, timing):
    """Solve the NLP subproblem."""
    config.logger.info(
        'Solving nonlinear subproblem for '
        'fixed binaries and logical realizations.')

    results = configure_and_call_solver(nlp_model, config.nlp_solver,
                                        config.nlp_solver_args, 'NLP', timing,
                                        config.time_limit)

    return process_nonlinear_problem_results(results, nlp_model, 'NLP', config)

def solve_MINLP(model, util_block, config, timing):
    """Solve the MINLP subproblem."""
    config.logger.info(
        "Solving MINLP subproblem for fixed logical realizations."
    )
    # TODO: make this a callback, which probably means calling it somewhere else
    # because it should have access to the master problem as well.
    initialize_subproblem(util_block)

    # Callback immediately before solving MINLP subproblem
    config.call_before_subproblem_solve(model)

    minlp_solver = SolverFactory(config.minlp_solver)
    if not minlp_solver.available():
        raise RuntimeError("MINLP solver %s is not available." %
                           config.minlp_solver)
                                    
    results = configure_and_call_solver(model, config.minlp_solver,
                                        config.minlp_solver_args, 'MINLP',
                                        timing, config.time_limit)
    subprob_feasible = process_nonlinear_problem_results(results, model,
                                                         'MINLP', config)
    
    # Call the subproblem post-solve callback
    config.call_after_subproblem_solve(model)

    # if feasible, call the subproblem post-feasible callback
    if subprob_feasible:
        config.call_after_subproblem_feasible(model)

    return subprob_feasible

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
        var_set.update(v for v in EXPR.identify_variables(obj.expr,
                                                          include_fixed=False)
                       if not v.is_continuous())
    return var_set

def preprocess_subproblem(m, config):
    """Applies preprocessing transformations to the model."""
    if not config.tighten_nlp_var_bounds:
        original_bounds = ComponentMap()
        # TODO: Switch this to the general utility function, but I hid it in
        # #2221
        for cons in m.component_data_objects(Constraint, active=True,
                                             descend_into=Block):
            for v in EXPR.identify_variables(cons.expr):
                if v not in original_bounds.keys():
                    original_bounds[v] = (v.lb, v.ub)
        # We could miss if there is a variable that only appears in the
        # objective, but its bounds are not going to get changed anyway if
        # that's the case.

    # First do FBBT
    fbbt(m, integer_tol=config.integer_tolerance,
         feasibility_tol=config.constraint_tolerance,
         max_iter=config.max_fbbt_iterations)
    xfrm = TransformationFactory
    # Now that we've tightened bounds, see if any variables are fixed because
    # their lb is equal to the ub (within tolerance)
    xfrm('contrib.detect_fixed_vars').apply_to(
         m, tolerance=config.variable_tolerance)

    # Restore the original bounds because the NLP solver might like that better
    # and because, if deactivate_trivial_constraints ever gets fancier, this
    # could change what is and is not trivial.
    if not config.tighten_nlp_var_bounds:
        for v, (lb, ub) in original_bounds.items():
            v.setlb(lb)
            v.setub(ub)

    # Now, if something got fixed to 0, we might have 0*var terms to remove
    xfrm('contrib.remove_zero_terms').apply_to(m)
    # Last, check if any constraints are now trivial and deactivate them
    xfrm('contrib.deactivate_trivial_constraints').apply_to(
        m, tolerance=config.constraint_tolerance)

def initialize_subproblem(util_block):
    """Perform initialization of the subproblem.

    Presently, this just restores the continuous variables to the original 
    model values.
    """
    # restore original continuous variable values
    for var, old_value in util_block.initial_var_values.items():
        if not var.fixed and var.is_continuous():
            if old_value is not None:
                # Adjust value if it falls outside the bounds
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                # Set the value
                var.set_value(old_value)

# ESJ TODO: YOU ARE HERE. I would rather return this info and process it
# somewhere else, I think. But I need to think about this since it comes up in
# initialization too.
def update_subproblem_progress_indicators(solved_model, solve_data, config):
    """Update the progress indicators for the subproblem."""
    GDPopt = solved_model.GDPopt_utils
    objective = next(solved_model.component_data_objects(Objective,
                                                         active=True))
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
        'ITER {:d}.{:d}.{:d}-NLP: OBJ: {:.10g}  LB: {:.10g} {:s} UB: {:.10g} '
        '{:s}'.format(
            solve_data.master_iteration,
            solve_data.mip_iteration,
            solve_data.nlp_iteration,
            value(objective.expr),
            solve_data.LB, lb_improved,
            solve_data.UB, ub_improved))

def solve_subproblem(subprob_util_block, config, timing, solve_globally=False):
    """Set up and solve the local MINLP or NLP subproblem."""
    # ESJ: do we need this/do we need to track it here?
    #solve_data.nlp_iteration += 1
    subprob = subprob_util_block.model()

    if config.subproblem_presolve:
        try:
            preprocess_subproblem(subprob, config)
        except InfeasibleConstraintException as e:
            config.logger.info("NLP subproblem determined to be infeasible "
                               "during preprocessing.")
            config.logger.debug("Message from preprocessing: %s" % e)
            return get_infeasible_result_object(
                subprob,
                "Preprocessing determined problem to be infeasible.")

    # TODO: If this is really here, then we need to have some very
    # strongly-worded documentation thatabout how modifying the subproblem model
    # could have very serious consequences... I don't really want to expose it
    # at all, honestly...
    config.call_before_subproblem_solve(subprob)

    # Is the subproblem linear?
    if not any(constr.body.polynomial_degree() not in (1, 0) for constr in
               subprob.component_data_objects(Constraint, active=True)):
        subprob_feasible = solve_linear_subproblem(subprob, config, timing)
    else:
        # Does it have any discrete variables, and is that allowed?
        unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
        if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
            raise RuntimeError("Unfixed discrete variables found on the NLP "
                               "subproblem.")
        elif len(unfixed_discrete_vars) == 0:
            subprob_feasible = solve_NLP(subprob, config, timing)
        else:
            subprob_feasible = solve_MINLP(subprob, subprob_util_block, config,
                                           timing)

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(subprob)

    # if feasible, call the NLP post-feasible callback
    if subprob_feasible:
        config.call_after_subproblem_feasible(subprob)

    return subprob_feasible

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
                disj.indicator_var.fix(bool(rounded_val))
            else:
                disj.indicator_var.fix(bool(val))

    if config.force_subproblem_nlp:
        # We also need to copy over the discrete variable values
        for var, val in zip(subprob.GDPopt_utils.variable_list,
                            mip_result.var_values):
            if var.is_continuous():
                continue
            rounded_val = int(round(val))
            if fabs(val - rounded_val) > config.integer_tolerance:
                raise ValueError( "Discrete variable %s value %s is not "
                                  "within tolerance %s of %s." % 
                                  (var.name, var.value, 
                                   config.integer_tolerance, rounded_val))
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
        except InfeasibleConstraintException:
            # Preprocessing found the problem to be infeasible
            return get_infeasible_result_object(
                subprob, "Preprocessing determined problem to be infeasible.")

    unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
    if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
        raise RuntimeError("Unfixed discrete variables found on the NLP "
                           "subproblem.")
    elif len(unfixed_discrete_vars) == 0:
        subprob_result = solve_NLP(subprob, solve_data, config)
    else:
        subprob_result = solve_MINLP(subprob, config)
    if subprob_result.feasible:  # NLP is feasible
        update_subproblem_progress_indicators(subprob, solve_data, config)
    return subprob_result

def get_infeasible_result_object(model, message=""):
    infeas_result = SubproblemResult()
    infeas_result.feasible = False
    infeas_result.var_values = list(v.value for v in
                                    model.GDPopt_utils.variable_list)
    infeas_result.pyomo_results = SolverResults()
    infeas_result.pyomo_results.solver.termination_condition = tc.infeasible
    infeas_result.pyomo_results.message = message
    infeas_result.dual_values = list(None for _ in
                                     model.GDPopt_utils.constraint_list)
    return infeas_result
