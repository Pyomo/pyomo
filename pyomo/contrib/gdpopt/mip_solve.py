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

def solve_linear_GDP(util_block, config, timing):
    """Solves the linear GDP model and attempts to resolve solution issues."""
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
            return False

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
    # ESJ: Should we expose alg_info here? I guess we should but I need to make
    # it safer in terms of getters and setters I think.
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
            return False
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
        # Solution is unbounded. Add an arbitrary bound to the objective and
        # resolve.  This occurs when the objective is nonlinear. The nonlinear
        # objective is moved to the constraints, and deactivated for the linear
        # master problem.
        # ESJ TODO: This is terrifying! I don't think we should do this... How
        # about just check your initialization routine? Or bound your variables?
        obj_bound = 1E15
        config.logger.warning(
            'Linear GDP was unbounded. '
            'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) '
            'on the objective. '
            'Check your initialization routine.'.format(obj_bound))
        main_objective = next(m.component_data_objects(Objective, active=True))
        GDPopt.objective_bound = Constraint(
            expr=(-obj_bound, main_objective.expr, obj_bound))
        with SuppressInfeasibleWarning():
            results = SolverFactory(config.mip_solver).solve(
                m, **config.mip_solver_args)
        terminate_cond = results.solver.termination_condition

    mip_feasible = True

    if terminate_cond in {tc.optimal, tc.locallyOptimal, tc.feasible}:
        pass
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'Linear GDP is now infeasible. '
            'GDPopt has finished exploring feasible discrete configurations.')
        mip_feasible = False
    elif terminate_cond is tc.maxTimeLimit:
        # TODO check that status is actually ok and everything is feasible
        config.logger.info(
            'Unable to optimize linear GDP problem within time limit. '
            'Using current solver feasible solution.')
    elif (terminate_cond is tc.other and
          results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'Linear GDP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
    else:
        raise ValueError(
            'GDPopt unable to handle linear GDP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))

    return mip_feasible

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
