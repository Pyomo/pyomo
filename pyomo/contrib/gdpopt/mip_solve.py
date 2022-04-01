"""Functions for solving the master problem."""
from copy import deepcopy

from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning, _DoNothing,
                                       get_main_elapsed_time)
from pyomo.core import Objective, Constraint
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

# TODO: I'd really like too change the name of this function. lienar_GDP is
# super confusing, since this is a MIP at this point!
def solve_linear_GDP(util_block, config, timing):
    """Solves the linear GDP model and attempts to resolve solution issues.  
    Returns one of TerminationCondition.optimal, TerminationCondition.feasible,
    TerminationCondition.infeasible, or TerminationCondition.unbounded.
    """
    m = util_block.model()

    if config.mip_presolve:
        try:
            # ESJ: deactivating satisfied constraints here is risky since it can
            # result in sending no constraints to the solver and hence not
            # actually filling in the values of the variables that got
            # implicitly fixed by these bounds. We can fix this by calling some
            # contrib.preprocessing transformations, but for now I'm just
            # leaving the constraints in.
            fbbt(m, integer_tol=config.integer_tolerance,
                 deactivate_satisfied_constraints=False)
            # [ESJ 1/28/22]: Despite being a little scary, the tightened bounds
            # are okay to leave in because if you tighten the bounds now, they
            # could only get tighter in later iterations, since you are
            # tightening this relaxation
        except InfeasibleConstraintException:
            config.logger.debug("MIP preprocessing detected infeasibility.")
            return tc.infeasible

    # Deactivate extraneous IMPORT/EXPORT suffixes 
    # ESJ TODO: Do we need to do this? Is this our problem? And, if you give 
    # a mouse a cookie... Would we have to account for other Suffixes?
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    # Create solver, check availability
    if not SolverFactory(config.mip_solver).available():
        raise RuntimeError(
            "MIP solver %s is not available." % config.mip_solver)

    # Callback immediately before solving MIP master problem
    # TODO: Should this have other arguments?
    config.call_before_master_solve(m)

    try:
        with SuppressInfeasibleWarning():
            mip_args = dict(config.mip_solver_args)
            elapsed = get_main_elapsed_time(timing)
            remaining = max(config.time_limit - elapsed, 1)
            if config.mip_solver == 'gams':
                mip_args['add_options'] = mip_args.get('add_options', [])
                mip_args['add_options'].append('option reslim=%s;' % remaining)
            elif config.mip_solver == 'multisolve':
                mip_args['time_limit'] = min(mip_args.get( 'time_limit',
                                                           float('inf')),
                                             remaining)
            results = SolverFactory(config.mip_solver).solve(m, **mip_args)

    except RuntimeError as e:
        # ESJ TODO: How come GAMS is special? Doesn't seem safe to assume
        # infeasibility here if the error could be something else...?
        if 'GAMS encountered an error during solve.' in str(e):
            config.logger.warning(
                "GAMS encountered an error in solve. Treating as infeasible.")
            return tc.infeasible
        else:
            raise
    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, terminate_cond = distinguish_mip_infeasible_or_unbounded(
            m, config)
    if terminate_cond is tc.unbounded:
        # Solution is unbounded. This occurs when the objective is
        # nonlinear. The nonlinear objective is moved to the constraints, and
        # deactivated for the linear master problem. We will generate an
        # arbitrary discrete solution by bounding the objective and re-solving,
        # in hopes that the cuts we generate later bound this problem.

        obj_bound = 1E15
        config.logger.warning(
            'Master problem was unbounded. '
            'Re-solving with arbitrary bound values of (-{0:.10g}, {0:.10g}) '
            'on the objective, in order to get a discrete solution. '
            'Check your initialization routine.'.format(obj_bound))
        main_objective = next(m.component_data_objects(Objective, active=True))
        util_block.objective_bound = Constraint(
            expr=(-obj_bound, main_objective.expr, obj_bound))
        with SuppressInfeasibleWarning():
            results = SolverFactory(config.mip_solver).solve(
                m, **config.mip_solver_args)
        if results.solver.termination_condition in {tc.optimal, tc.feasible,
                                                    tc.locallyOptimal,
                                                    tc.globallyOptimal}:
            # we found a solution, that's all we need to keep going.
            return tc.unbounded
        else:
            set_trace()
            raise NotImplementedError(
                "TODO: I guess we can increase that bound?")

    if terminate_cond is tc.optimal:
        return tc.optimal
    elif terminate_cond in {tc.locallyOptimal, tc.feasible}:
        return tc.feasible
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'Linear GDP is now infeasible. '
            'GDPopt has finished exploring feasible discrete configurations.')
        return tc.infeasible
    elif terminate_cond is tc.maxTimeLimit:
        if len(results.solution) > 0:
            config.logger.info(
                'Unable to optimize linear GDP problem within time limit. '
                'Using current solver feasible solution.')
            return tc.feasible
        else:
            config.logger.info(
                'Unable to optimize linear GDP problem within time limit. '
                'No solution found. Treating as infeasible, but there are no '
                'guarantees.')
            return tc.infeasible
    elif (terminate_cond is tc.other and
          results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'Linear GDP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        return tc.feasible
    else:
        raise ValueError(
            'GDPopt unable to handle linear GDP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))

def distinguish_mip_infeasible_or_unbounded(m, config):
    """Distinguish between an infeasible or unbounded solution.

    Linear solvers will sometimes tell me that a problem is infeasible or
    unbounded during presolve, but not distinguish between the two cases. We
    address this by solving again with a solver option flag on.

    """
    tmp_args = deepcopy(config.mip_solver_args)
    if config.mip_solver == 'gurobi':
        # This solver option is specific to Gurobi.
        tmp_args['options'] = tmp_args.get('options', {})
        tmp_args['options']['DualReductions'] = 0
    mipopt = SolverFactory(config.mip_solver)
    # ESJ TODO: Why does this only happen here? Shouldn't it be everywhere or
    # nowhere? Oh, or is that what the callbacks are for? But how would the user
    # know what changed?
    if isinstance(mipopt, PersistentSolver):
        mipopt.set_instance(m)
    with SuppressInfeasibleWarning():
        results = mipopt.solve(m, **tmp_args)
    termination_condition = results.solver.termination_condition
    return results, termination_condition
