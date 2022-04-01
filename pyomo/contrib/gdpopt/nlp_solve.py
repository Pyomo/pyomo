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
from contextlib import contextmanager
from math import fabs

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException, DeveloperError
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning,
                                       is_feasible, get_main_elapsed_time)
from pyomo.core import Constraint, TransformationFactory, Objective, Block
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc

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
    """Processes the results object returns from the nonlinear solver.
    Returns one of TerminationCondition.optimal (for locally optimal or
    globally optimal since people use this as a heuristic),
    TerminationCondition.feasible (we have a solution with no guarantees),
    TerminationCondition.noSolution (we have no solution, with not guarantees
    of infeasibility), or TerminationCondition.infeasible.
    """
    logger = config.logger
    term_cond = results.solver.termination_condition
    if any(term_cond == cond for cond in (tc.optimal, tc.locallyOptimal,
                                          tc.globallyOptimal)):
        # Since we let people use local solvers and settle for the heuristic, we
        # just let all these by.
        return tc.optimal
    elif term_cond == tc.feasible:
        return tc.feasible
    elif term_cond == tc.infeasible:
        logger.debug('%s subproblem was infeasible.' % problem_type)
        return tc.infeasible
    elif term_cond == tc.maxIterations:
        # TODO try something else? Reinitialize with different initial
        # value?
        logger.debug( '%s subproblem failed to converge within iteration limit.'
                     % problem_type)
        if is_feasible(model, config):
            logger.debug(
                'NLP solution is still feasible. '
                'Using potentially suboptimal feasible solution.')
            return tc.feasible
        return False
    elif term_cond == tc.internalSolverError:
        # Possible that IPOPT had a restoration failure
        logger.debug( "%s solver had an internal failure: %s" %
                     (problem_type, results.solver.message))
        return tc.noSolution
    elif (term_cond == tc.other and
          "Too few degrees of freedom" in str(results.solver.message)):
        # Possible IPOPT degrees of freedom error
        logger.debug(
            "Perhaps the subproblem solver has too few degrees of freedom: %s" %
            results.solver.message)
        return tc.infeasible
    elif term_cond == tc.other:
        logger.debug(
            "%s solver had a termination condition of 'other': %s" %
            (problem_type, results.solver.message))
        return tc.noSolution
    elif term_cond == tc.error:
        logger.debug("%s solver had a termination condition of 'error': "
                    "%s" % (problem_type, results.solver.message))
        return tc.noSolution
    elif term_cond == tc.maxTimeLimit:
        logger.debug("%s subproblem failed to converge within time "
                    "limit." % problem_type)
        if is_feasible(model, config):
            config.logger.debug(
                '%s solution is still feasible. '
                'Using potentially suboptimal feasible solution.' %
                problem_type)
            return tc.feasible
        return tc.noSolution
    elif term_cond == tc.intermediateNonInteger:
        config.logger.debug( "%s solver could not find feasible integer"
                            " solution: %s" % (problem_type,
                                               results.solver.message))
        return tc.noSolution
    elif term_cond == tc.unbounded:
        config.logger.debug("The NLP subproblem is unbounded, meaning that "
                           "the GDP is unbounded.")
        return tc.unbounded
    else:
        # This isn't the user's fault, but we give up--we don't know what's
        # going on.
        raise DeveloperError(
            'GDPopt unable to handle %s subproblem termination '
            'condition of %s. Results: %s' % (problem_type, term_cond, results))

def solve_linear_subproblem(subproblem, config, timing):
    results = configure_and_call_solver(subproblem, config.mip_solver,
                                        config.mip_solver_args, 'MIP', timing,
                                        config.time_limit)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        return tc.optimal
    elif subprob_terminate_cond is tc.infeasible:
        config.logger.debug('MILP subproblem was infeasible.')
        return tc.infeasible
    elif subprob_terminate_cond is tc.unbounded:
        config.logger.debug('MILP subproblem was unbounded.')
        return tc.unbounded
    else:
        raise ValueError(
            'GDPopt unable to handle MIP subproblem termination '
            'condition of %s. Results: %s'
            % (subprob_terminate_cond, results))

def solve_NLP(nlp_model, config, timing):
    """Solve the NLP subproblem."""
    config.logger.debug(
        'Solving nonlinear subproblem for '
        'fixed binaries and logical realizations.')

    results = configure_and_call_solver(nlp_model, config.nlp_solver,
                                        config.nlp_solver_args, 'NLP', timing,
                                        config.time_limit)

    return process_nonlinear_problem_results(results, nlp_model, 'NLP', config)

def solve_MINLP(util_block, config, timing):
    """Solve the MINLP subproblem."""
    config.logger.debug(
        "Solving MINLP subproblem for fixed logical realizations."
    )
    model = util_block.model()
    # TODO: make this a callback, which probably means calling it somewhere else
    # because it should have access to the master problem as well.
    initialize_subproblem(util_block)

    minlp_solver = SolverFactory(config.minlp_solver)
    if not minlp_solver.available():
        raise RuntimeError("MINLP solver %s is not available." %
                           config.minlp_solver)

    results = configure_and_call_solver(model, config.minlp_solver,
                                        config.minlp_solver_args, 'MINLP',
                                        timing, config.time_limit)
    subprob_termination = process_nonlinear_problem_results(results, model,
                                                            'MINLP', config)

    return subprob_termination

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

@contextmanager
def preprocess_subproblem(util_block, config):
    """Applies preprocessing transformations to the model."""
    m = util_block.model()

    # Save bounds so we can restore them
    original_bounds = ComponentMap()
    unfixed_vars = []
    for cons in m.component_data_objects(Constraint, active=True,
                                         descend_into=Block):
        for v in EXPR.identify_variables(cons.expr):
            if v not in original_bounds.keys():
                original_bounds[v] = (v.lb, v.ub)
                if not v.fixed:
                    unfixed_vars.append(v)
    # We could miss if there is a variable that only appears in the
    # objective, but its bounds are not going to get changed anyway if
    # that's the case.
    constraints_modified = {}
    constraints_deactivated = []
    not_infeas = True

    try:
        # First do FBBT
        fbbt(m, integer_tol=config.integer_tolerance,
             feasibility_tol=config.constraint_tolerance,
             max_iter=config.max_fbbt_iterations)
        xfrm = TransformationFactory
        # Now that we've tightened bounds, see if any variables are fixed
        # because their lb is equal to the ub (within tolerance)
        xfrm('contrib.detect_fixed_vars').apply_to(
             m, tolerance=config.variable_tolerance)

        # Restore the original bounds because the NLP solver might like that
        # better and because, if deactivate_trivial_constraints ever gets
        # fancier, this could change what is and is not trivial.
        if not config.tighten_nlp_var_bounds:
            for v, (lb, ub) in original_bounds.items():
                v.setlb(lb)
                v.setub(ub)

        # Now, if something got fixed to 0, we might have 0*var terms to remove
        xfrm('contrib.remove_zero_terms').apply_to(
            m, constraints_modified=constraints_modified)
        # Last, check if any constraints are now trivial and deactivate them
        xfrm('contrib.deactivate_trivial_constraints').apply_to(
            m, tolerance=config.constraint_tolerance,
            return_trivial=constraints_deactivated)

    except InfeasibleConstraintException as e:
            config.logger.debug("NLP subproblem determined to be infeasible "
                                "during preprocessing. Message: %s" % e)
            not_infeas = False

    yield not_infeas

    # restore the bounds if we found the problem infeasible or if we didn't do
    # it above
    if not not_infeas or config.tighten_nlp_var_bounds:
        for v, (lb, ub) in original_bounds.items():
            v.setlb(lb)
            v.setub(ub)

    # ESJ TODO: I think fbbt is messing with Vars that aren't its beeswax, but
    # it tightens the bounds on the fixed Boolean vars, so restore the bounds
    # here for now.
    for disj in util_block.disjunct_list:
        disj.binary_indicator_var.setlb(0)
        disj.binary_indicator_var.setub(1)
    for bool_var in util_block.non_indicator_boolean_variable_list:
        bool_var.get_associated_binary().setlb(0)
        bool_var.get_associated_binary().setub(1)

    # reactivate constraints
    for cons in constraints_deactivated:
        cons.activate()

    for cons, (orig, modified) in constraints_modified.items():
        cons.set_value(orig)

    # unfix variables: TODO: Maybe I should modify preprocessing so we only have
    # to go through the relevant ones here too
    for v in unfixed_vars:
        v.unfix()

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

def call_appropriate_subproblem_solver(subprob_util_block, config, timing):
    subprob = subprob_util_block.model()
    # TODO: If this is really here, then we need to have some very
    # strongly-worded documentation about how modifying the subproblem model
    # could have very serious consequences... I don't really want to expose it
    # at all, honestly...
    config.call_before_subproblem_solve(subprob)

    # Is the subproblem linear?
    if not any(constr.body.polynomial_degree() not in (1, 0) for constr in
               subprob.component_data_objects(Constraint, active=True)):
        subprob_termination = solve_linear_subproblem(subprob, config, timing)
    else:
        # Does it have any discrete variables, and is that allowed?
        unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
        if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
            raise RuntimeError("Unfixed discrete variables found on the NLP "
                               "subproblem.")
        elif len(unfixed_discrete_vars) == 0:
            subprob_termination = solve_NLP(subprob, config, timing)
        else:
            config.logger.debug(
                "The following discrete variables are unfixed: %s"
                "\nProceeding by solving the subproblem as a MINLP."
                % ", ".join([v.name for v in unfixed_discrete_vars]))
            subprob_termination = solve_MINLP(subprob_util_block, config,
                                              timing)

    # Call the NLP post-solve callback
    config.call_after_subproblem_solve(subprob)

    # if feasible, call the NLP post-feasible callback
    if subprob_termination in {tc.optimal, tc.feasible}:
        config.call_after_subproblem_feasible(subprob)

    return subprob_termination

def solve_subproblem(subprob_util_block, config, timing):
    """Set up and solve the local MINLP or NLP subproblem."""
    # ESJ: do we need this/do we need to track it here?
    #solve_data.nlp_iteration += 1
    if config.subproblem_presolve:
        with preprocess_subproblem(subprob_util_block, config) as call_solver:
            if call_solver:
                return call_appropriate_subproblem_solver(subprob_util_block,
                                                          config, timing)
            else:
                return tc.infeasible

    return call_appropriate_subproblem_solver(subprob_util_block, config,
                                              timing)
