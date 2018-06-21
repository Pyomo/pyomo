"""Functions for solving the master problem."""

from __future__ import division

from copy import deepcopy

from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import TransformationFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory, SolverStatus


def solve_linear_GDP(linear_GDP_model, solve_data, config):
    m = linear_GDP_model
    GDPopt = m.GDPopt_utils
    # Transform disjunctions
    TransformationFactory('gdp.bigm').apply_to(m)

    preprocessing_transformations = [
        # Propagate variable bounds
        'contrib.propagate_eq_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Propagate fixed variables
        'contrib.propagate_fixed_vars',
        # Remove zero terms in linear expressions
        'contrib.remove_zero_terms',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Transform bound constraints
        'contrib.constraints_to_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Remove trivial constraints
        'contrib.deactivate_trivial_constraints']
    for xfrm in preprocessing_transformations:
        TransformationFactory(xfrm).apply_to(m)

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    # Load solutions is false because otherwise an annoying error message
    # appears for infeasible models.
    mip_solver = SolverFactory(config.mip)
    if not mip_solver.available():
        raise RuntimeError("MIP solver %s is not available." % config.mip)
    results = mip_solver.solve(
        m, load_solutions=False,
        **config.mip_options)
    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        tmp_options = deepcopy(config.mip_options)
        # TODO This solver option is specific to Gurobi.
        tmp_options['DualReductions'] = 0
        results = mip_solver.solve(
            m, load_solutions=False,
            **tmp_options)
        terminate_cond = results.solver.termination_condition

    if terminate_cond is tc.optimal:
        m.solutions.load_from(results)
        return True, list(v.value for v in GDPopt.working_var_list)
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'Linear GDP is infeasible. '
            'Problem may have no more feasible discrete configurations.')
        return False
    elif terminate_cond is tc.maxTimeLimit:
        # TODO check that status is actually ok and everything is feasible
        config.logger.info(
            'Unable to optimize linear GDP problem within time limit. '
            'Using current solver feasible solution.')
        results.solver.status = SolverStatus.ok
        m.solutions.load_from(results)
        return True, list(v.value for v in GDPopt.working_var_list)
    elif (terminate_cond is tc.other and
          results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'Linear GDP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        results.solver.status = SolverStatus.ok
        m.solutions.load_from(results)
        return True, list(v.value for v in GDPopt.working_var_list)
    else:
        raise ValueError(
            'GDPopt unable to handle linear GDP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))
