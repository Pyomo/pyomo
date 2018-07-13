"""Initialization functions."""
from __future__ import division

from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing
from pyomo.contrib.mindtpy.cut_generation import (add_ecp_cut, add_gbd_cut,
                                                  add_oa_cut,
                                                  add_objective_linearization,
                                                  add_psc_cut)
from pyomo.contrib.mindtpy.nlp_solve import solve_NLP_subproblem
from pyomo.contrib.mindtpy.util import (calc_jacobians, copy_values,
                                        detect_nonlinear_vars)
from pyomo.core import (ConstraintList, Objective, Suffix,
                        TransformationFactory, maximize, minimize, value)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory


def MindtPy_initialize_master(solve_data, config):
    """Initialize the decomposition algorithm.
    This includes generating the initial cuts require to build the master
    problem.
    """
    m = solve_data.mip = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils

    m.dual.activate()

    if config.strategy == 'OA':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.MindtPy_linear_cuts.oa_cuts = ConstraintList(
            doc='Outer approximation cuts')
    elif config.strategy == 'ECP':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.MindtPy_linear_cuts.ecp_cuts = ConstraintList(
            doc='Extended Cutting Planes')
    elif config.strategy == 'PSC':
        detect_nonlinear_vars(solve_data, config)
        MindtPy.MindtPy_linear_cuts.psc_cuts = ConstraintList(
            doc='Partial surrogate cuts')
    elif config.strategy == 'GBD':
        MindtPy.MindtPy_linear_cuts.gbd_cuts = ConstraintList(
            doc='Generalized Benders cuts')

    # Set default initialization_strategy
    if config.init_strategy is None:
        if config.strategy == 'OA':
            config.init_strategy = 'rNLP'
        else:
            config.init_strategy = 'max_binary'
    # Do the initialization
    elif config.init_strategy == 'rNLP':
        init_rNLP(solve_data, config)
    elif config.init_strategy == 'max_binary':
        init_max_binaries(solve_data, config)
        if config.strategy == 'ECP':
            add_ecp_cut(solve_data, config)
        else:
            solve_NLP_subproblem(solve_data, config)


def init_rNLP(solve_data, config):
    """Initialize by solving the rNLP (relaxed binary variables)."""
    solve_data.nlp_iter += 1
    m = solve_data.working_model.clone()
    config.logger.info(
        "NLP %s: Solve relaxed integrality" % (solve_data.nlp_iter,))
    MindtPy = m.MindtPy_utils
    TransformationFactory('core.relax_integrality').apply_to(m)
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            m, **config.nlp_solver_args)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        nlp_solution_values = list(v.value for v in MindtPy.var_list)
        dual_values = list(m.dual[c] for c in MindtPy.constraints)
        # Add OA cut
        if MindtPy.objective.sense == minimize:
            solve_data.LB = value(MindtPy.objective.expr)
        else:
            solve_data.UB = value(MindtPy.objective.expr)
        config.logger.info(
            'NLP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.nlp_iter, value(MindtPy.objective.expr),
               solve_data.LB, solve_data.UB))
        if config.strategy == 'OA':
            add_oa_cut(nlp_solution_values, dual_values, solve_data, config)
        elif config.strategy == 'PSC':
            add_psc_cut(solve_data, config)
        elif config.strategy == 'GBD':
            add_gbd_cut(solve_data, config)
        elif config.strategy == 'ECP':
            add_ecp_cut(solve_data, config)
            add_objective_linearization(m, solve_data, config)
    elif subprob_terminate_cond is tc.infeasible:
        # TODO fail? try something else?
        config.logger.info(
            'Initial relaxed NLP problem is infeasible. '
            'Problem may be infeasible.')
    else:
        raise ValueError(
            'MindtPy unable to handle relaxed NLP termination condition '
            'of %s. Solver message: %s' %
            (subprob_terminate_cond, results.solver.message))


def init_max_binaries(solve_data, config):
    """Initialize by turning on as many binary variables as possible.

    The user would usually want to call _solve_NLP_subproblem after an
    invocation of this function.

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    solve_data.mip_subiter += 1
    config.logger.info(
        "MILP %s: maximize value of binaries" %
        (solve_data.mip_iter))
    for c in MindtPy.nonlinear_constraints:
        c.deactivate()
    MindtPy.obj.deactivate()
    MindtPy.MindtPy_max_binary_obj = Objective(
        expr=sum(v for v in MindtPy.binary_vars), sense=maximize)

    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    results = solve_data.mip_solver.solve(m, options=config.mip_solver_args)

    getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

    MindtPy.MindtPy_max_binary_obj.deactivate()

    MindtPy.obj.activate()
    for c in MindtPy.nonlinear_constraints:
        c.activate()
    solve_terminate_cond = results.solver.termination_condition
    if solve_terminate_cond is tc.optimal:
        pass  # good
    elif solve_terminate_cond is tc.infeasible:
        raise ValueError(
            'MILP master problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of %s. Solver message: %s' %
            (solve_terminate_cond, results.solver.message))
