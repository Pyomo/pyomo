"""Functions for initializing the master problem
in Logic-based outer approximation.
"""

from __future__ import division

from copy import deepcopy
from math import fabs

from pyomo.core import TerminationCondition as tc
from pyomo.core import (Block, ComponentSet, Constraint, Objective,
                        SolverFactory, TransformationFactory, Var, maximize,
                        minimize, value, ComponentMap)
from pyomo.gdp import Disjunct


def solve_linear_GDP():
    pass


def _init_custom_disjuncts(self, solve_data, config):
    """Initialize by using user-specified custom disjuncts."""
    # TODO error checking to make sure that the user gave proper disjuncts
    for active_disjunct_set in config.custom_init_disjuncts:
        # custom_init_disjuncts contains a list of sets, giving the disjuncts
        # active at each initialization iteration
        fixed_disjs = ComponentSet()
        for disj in active_disjunct_set:
            if not disj.indicator_var.fixed:
                disj.indicator_var.fix(1)
                fixed_disjs.add(disj)
        self._solve_init_MIP(solve_data)
        for disj in fixed_disjs:
            disj.indicator_var.unfix()
        self._solve_NLP_subproblem(solve_data)


def _init_max_binaries(self, solve_data, config):
    """Initialize by maximizing binary variables and disjuncts.

    This function activates as many binary variables and disjucts as
    feasible. The user would usually want to call _solve_NLP_subproblem()
    after an invocation of this function.

    """
    solve_data.mip_subiter += 1
    # print('Clone working model for init max binaries')
    m = solve_data.working_model.clone()
    config.logger.info(
        "MIP %s.%s: maximize value of binaries" %
        (solve_data.mip_iter, solve_data.mip_subiter))
    nonlinear_constraints = (
        c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=(Block, Disjunct))
        if c.body.polynomial_degree() not in (0, 1))
    for c in nonlinear_constraints:
        c.deactivate()
    m.GDPopt_utils.objective.deactivate()
    binary_vars = (
        v for v in m.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v.is_binary() and not v.fixed)
    m.GDPopt_utils.max_binary_obj = Objective(
        expr=sum(v for v in binary_vars), sense=maximize)
    TransformationFactory('gdp.bigm').apply_to(m)
    # TODO: chull too?
    mip_solver = SolverFactory(config.mip)
    if not mip_solver.available():
        raise RuntimeError("MIP solver %s is not available." % config.mip)
    results = mip_solver.solve(
        m, **config.mip_options)
    # m.display()
    solve_terminate_cond = results.solver.termination_condition
    if solve_terminate_cond is tc.optimal:
        # Transfer variable values back to main working model
        self._copy_values(m, solve_data.working_model, config)
    elif solve_terminate_cond is tc.infeasible:
        raise ValueError('Linear relaxation is infeasible. '
                         'Problem is infeasible.')
    else:
        raise ValueError('Cannot handle termination condition %s'
                         % (solve_terminate_cond,))


def _init_set_covering(self, solve_data, config, iterlim=8):
    """Initialize by solving problems to cover the set of all disjuncts.

    The purpose of this initialization is to generate linearizations
    corresponding to each of the disjuncts.

    This work is based upon prototyping work done by Eloy Fernandez at
    Carnegie Mellon University.

    """
    m = solve_data.working_model
    GDPopt = m.GDPopt_utils
    GDPopt.nonlinear_disjuncts = list(
        disj for disj in m.component_data_objects(
            ctype=Disjunct, active=True, descend_into=(Block, Disjunct))
        if any(constr.body.polynomial_degree() not in (0, 1)
               for constr in disj.component_data_objects(
                   ctype=Constraint, active=True,
                   # TODO should this descend into both Block and Disjunct,
                   # or just Block?
                   descend_into=(Block, Disjunct))))
    GDPopt.covered_disjuncts = ComponentSet()
    GDPopt.not_covered_disjuncts = ComponentSet(GDPopt.nonlinear_disjuncts)
    iter_count = 1
    GDPopt.feasible_integer_cuts.activate()
    while GDPopt.not_covered_disjuncts and iter_count <= iterlim:
        # Solve set covering MIP
        if not self._solve_set_cover_MIP(solve_data, config):
            # problem is infeasible. break
            return False
        # solve local NLP
        if self._solve_NLP_subproblem(solve_data, config):
            # if successful, updated sets
            active_disjuncts = ComponentSet(
                disj for disj in m.component_data_objects(
                    ctype=Disjunct, active=True,
                    descend_into=(Block, Disjunct))
                if fabs(value(disj.indicator_var) - 1)
                <= config.integer_tolerance)
            GDPopt.covered_disjuncts.update(active_disjuncts)
            GDPopt.not_covered_disjuncts -= active_disjuncts
        iter_count += 1
        # m.GDPopt_utils.integer_cuts.pprint()
    GDPopt.feasible_integer_cuts.deactivate()
    if GDPopt.not_covered_disjuncts:
        # Iteration limit was hit without a full covering of all nonlinear
        # disjuncts
        logger.warn('Iteration limit reached for set covering '
                    'initialization.')
        return False
    return True


def _solve_set_cover_MIP(self, solve_data, config):
    m = solve_data.working_model.clone()
    GDPopt = m.GDPopt_utils
    covered_disjuncts = GDPopt.covered_disjuncts

    # Set up set covering objective
    weights = ComponentMap((disj, 1) for disj in covered_disjuncts)
    weights.update(ComponentMap(
        (disj, len(covered_disjuncts) + 1)
        for disj in GDPopt.not_covered_disjuncts))
    GDPopt.objective.deactivate()
    GDPopt.set_cover_obj = Objective(
        expr=sum(weights[disj] * disj.indicator_var
                 for disj in weights),
        sense=maximize)

    # deactivate nonlinear constraints
    nonlinear_constraints = (
        c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=(Block, Disjunct))
        if c.body.polynomial_degree() not in (0, 1))
    for c in nonlinear_constraints:
        c.deactivate()

    # Deactivate potentially non-rigorous generated cuts
    for constr in m.component_objects(ctype=Constraint, active=True,
                                      descend_into=(Block, Disjunct)):
        if (constr.local_name == 'GDPopt_OA_cuts'):
            constr.deactivate()

    # Transform disjunctions
    TransformationFactory('gdp.bigm').apply_to(m)

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
        old_options = deepcopy(mip_solver.options)
        # This solver option is specific to Gurobi.
        mip_solver.options['DualReductions'] = 0
        results = mip_solver.solve(
            m, load_solutions=False,
            **config.mip_options)
        terminate_cond = results.solver.termination_condition
        mip_solver.options.update(old_options)

    if terminate_cond is tc.optimal:
        m.solutions.load_from(results)
        self._copy_values(m, solve_data.working_model, config)
        config.logger.info('Solved set covering MIP')
        return True
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'Set covering problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
        if GDPopt.mip_iter <= 1:
            config.logger.warn(
                'Set covering problem was infeasible. '
                'Check your linear and logical constraints '
                'for contradictions.')
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
        return False
    else:
        raise ValueError(
            'GDPopt unable to handle set covering MILP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))


def _solve_init_MIP(self, solve_data, config):
    """Solves the initialization MIP corresponding to the passed model.

    Intended to consolidate some MIP solution code.

    """
    # print('Clone working model for init MIP')
    m = solve_data.working_model.clone()
    GDPopt = m.GDPopt_utils

    # deactivate nonlinear constraints
    nonlinear_constraints = (
        c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=(Block, Disjunct))
        if c.body.polynomial_degree() not in (0, 1))
    for c in nonlinear_constraints:
        c.deactivate()

    # Transform disjunctions
    TransformationFactory('gdp.bigm').apply_to(m)

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

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    results = solve_data.mip_solver.solve(
        m, load_solutions=False,
        **solve_data.mip_solver_kwargs)
    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        old_options = deepcopy(solve_data.mip_solver.options)
        # This solver option is specific to Gurobi.
        solve_data.mip_solver.options['DualReductions'] = 0
        results = solve_data.mip_solver.solve(
            m, load_solutions=False,
            **solve_data.mip_solver_kwargs)
        terminate_cond = results.solver.termination_condition
        solve_data.mip_solver.options.update(old_options)

    if terminate_cond is tc.optimal:
        m.solutions.load_from(results)
        self._copy_values(m, solve_data.working_model)
        logger.info('Solved set covering MIP')
        return True
    elif terminate_cond is tc.infeasible:
        logger.info('Set covering problem is infeasible. '
                    'Problem may have no more feasible '
                    'binary configurations.')
        if solve_data.mip_iter <= 1:
            logger.warn('Problem was infeasible. '
                        'Check your linear and logical constraints '
                        'for contradictions.')
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
        return False
    else:
        raise ValueError(
            'GDPopt unable to handle set covering MILP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))
