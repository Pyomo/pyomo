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
"""Functions for solving the discrete problem."""
from copy import deepcopy

from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (
    SuppressInfeasibleWarning,
    _DoNothing,
    get_main_elapsed_time,
)
from pyomo.core import Objective, Constraint
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver


def solve_MILP_discrete_problem(util_block, solver, config):
    """Solves the linear GDP model and attempts to resolve solution issues.
    Returns one of TerminationCondition.optimal, TerminationCondition.feasible,
    TerminationCondition.infeasible, or TerminationCondition.unbounded.
    """
    timing = solver.timing
    m = util_block.parent_block()

    if config.mip_presolve:
        try:
            # ESJ: deactivating satisfied constraints here is risky since it can
            # result in sending no constraints to the solver and hence not
            # actually filling in the values of the variables that got
            # implicitly fixed by these bounds. We can fix this by calling some
            # contrib.preprocessing transformations, but for now I'm just
            # leaving the constraints in.
            fbbt(
                m,
                integer_tol=config.integer_tolerance,
                deactivate_satisfied_constraints=False,
            )
            # [ESJ 1/28/22]: Despite being a little scary, the tightened bounds
            # are okay to leave in because if you tighten the bounds now, they
            # could only get tighter in later iterations, since you are
            # tightening this relaxation
        except InfeasibleConstraintException as e:
            config.logger.debug(
                "MIP preprocessing detected infeasibility:\n\t%s" % str(e)
            )
            return tc.infeasible

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    # Create solver, check availability
    if not SolverFactory(config.mip_solver).available():
        raise RuntimeError("MIP solver %s is not available." % config.mip_solver)

    # Callback immediately before solving MIP discrete problem
    config.call_before_discrete_problem_solve(solver, m, util_block)
    if config.call_before_master_solve is not _DoNothing:
        deprecation_warning(
            "The 'call_before_master_solve' argument is deprecated. "
            "Please use the 'call_before_discrete_problem_solve' option "
            "to specify the callback.",
            version="6.4.2",
        )

    with SuppressInfeasibleWarning():
        mip_args = dict(config.mip_solver_args)
        if config.time_limit is not None:
            elapsed = get_main_elapsed_time(timing)
            remaining = max(config.time_limit - elapsed, 1)
            if config.mip_solver == 'gams':
                mip_args['add_options'] = mip_args.get('add_options', [])
                mip_args['add_options'].append('option reslim=%s;' % remaining)
            elif config.mip_solver == 'multisolve':
                mip_args['time_limit'] = min(
                    mip_args.get('time_limit', float('inf')), remaining
                )
        results = SolverFactory(config.mip_solver).solve(m, **mip_args)

    config.call_after_discrete_problem_solve(solver, m, util_block)
    if config.call_after_master_solve is not _DoNothing:
        deprecation_warning(
            "The 'call_after_master_solve' argument is deprecated. "
            "Please use the 'call_after_discrete_problem_solve' option to "
            "specify the callback.",
            version="6.4.2",
        )

    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, terminate_cond = distinguish_mip_infeasible_or_unbounded(m, config)
    if terminate_cond is tc.unbounded:
        # Solution is unbounded. This occurs when the objective is
        # nonlinear. The nonlinear objective is moved to the constraints, and
        # deactivated for the linear discrete problem. We will generate an
        # arbitrary discrete solution by bounding the objective and re-solving,
        # in hopes that the cuts we generate later bound this problem.

        obj_bound = 1e15
        config.logger.warning(
            'Discrete problem was unbounded. '
            'Re-solving with arbitrary bound values of (-{0:.10g}, {0:.10g}) '
            'on the objective, in order to get a discrete solution. '
            'Check your initialization routine.'.format(obj_bound)
        )
        discrete_objective = next(m.component_data_objects(Objective, active=True))
        util_block.objective_bound = Constraint(
            expr=(-obj_bound, discrete_objective.expr, obj_bound)
        )
        with SuppressInfeasibleWarning():
            results = SolverFactory(config.mip_solver).solve(
                m, **config.mip_solver_args
            )
        # get rid of the made-up constraint
        del util_block.objective_bound
        if results.solver.termination_condition in {
            tc.optimal,
            tc.feasible,
            tc.locallyOptimal,
            tc.globallyOptimal,
        }:
            # we found a solution, that's all we need to keep going.
            return tc.unbounded
        else:
            raise RuntimeError(
                "Unable to find a feasible solution for the "
                "unbounded MILP discrete problem by bounding "
                "the objective. Either check your "
                "discrete problem initialization, or add a "
                "bound on the discrete problem objective value "
                "that admits a feasible solution."
            )

    if terminate_cond is tc.optimal:
        return tc.optimal
    elif terminate_cond in {tc.locallyOptimal, tc.feasible}:
        return tc.feasible
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'MILP discrete problem is now infeasible. GDPopt has explored or '
            'cut off all feasible discrete configurations.'
        )
        return tc.infeasible
    elif terminate_cond is tc.maxTimeLimit:
        if len(results.solution) > 0:
            config.logger.info(
                'Unable to optimize MILP discrete problem within time limit. '
                'Using current solver feasible solution.'
            )
            return tc.feasible
        else:
            config.logger.info(
                'Unable to optimize MILP discrete problem within time limit. '
                'No solution found. Treating as infeasible, but there are no '
                'guarantees.'
            )
            return tc.infeasible
    elif (
        terminate_cond is tc.other
        and results.solution.status is SolutionStatus.feasible
    ):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'MIP solver reported feasible solution to MILP discrete problem, '
            'but it is not guaranteed to be optimal.'
        )
        return tc.feasible
    else:
        raise ValueError(
            'GDPopt unable to handle MILP discrete problem '
            'termination condition '
            'of %s. Solver message: %s' % (terminate_cond, results.solver.message)
        )


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
    # gdpopt no longer supports non-auto persistent solvers, but mindtpy does,
    # and it uses this function.
    if isinstance(mipopt, PersistentSolver):
        mipopt.set_instance(m)
    with SuppressInfeasibleWarning():
        results = mipopt.solve(m, load_solutions=False, **tmp_args)
        if len(results.solution) > 0:
            m.solutions.load_from(results)
    termination_condition = results.solver.termination_condition
    return results, termination_condition
