#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Functions for construction and solution of the PyROS master problem.
"""

import os

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import TransformationFactory
from pyomo.core.base import (
    ConcreteModel,
    Block,
    Var,
    Objective,
    Constraint,
)
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr import value
from pyomo.opt import (
    check_optimal_termination,
    SolverResults,
    TerminationCondition as tc,
)
from pyomo.repn.standard_repn import generate_standard_repn

from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.contrib.pyros.util import (
    call_solver,
    enforce_dr_degree,
    get_dr_expression,
    get_main_elapsed_time,
    check_time_limit_reached,
    ObjectiveType,
    process_termination_condition_master_problem,
    pyrosTerminationCondition,
    selective_clone,
    TIC_TOC_SOLVE_TIME_ATTR,
)


def construct_initial_master_problem(model_data, config):
    """
    Construct the initial master problem model object
    from the preprocessed working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object,
        containing the preprocessed working model.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    master_model : ConcreteModel
        Initial master problem model object.
        Contains a single scenario block fully cloned from
        the working model.
    """
    master_model = m = ConcreteModel()
    m.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)
    add_scenario_block_to_master_problem(
        master_model=master_model,
        scenario_idx=(0, 0),
        param_realization=config.nominal_uncertain_param_vals,
        from_block=model_data.working_model,
        clone_first_stage_components=True,
    )

    # epigraph Objective was not added during preprocessing,
    # as we wanted to add it to the root block of the master
    # model rather than to the model to prevent
    # duplication across scenario sub-blocks
    master_model.epigraph_obj = Objective(
        expr=master_model.scenarios[0, 0].epigraph_var,
    )

    return master_model


def add_scenario_block_to_master_problem(
        master_model,
        scenario_idx,
        param_realization,
        from_block,
        clone_first_stage_components,
        ):
    """
    Add new scenario block to the master model.

    Parameters
    ----------
    master_model : ConcreteModel
        Master model.
    scenario_idx : tuple
        Index of ``master_model.scenarios`` for the new block.
    param_realization : Iterable of numeric type
        Uncertain parameter realization for new block.
    from_block : BlockData
        Block from which to transfer attributes.
        This can be an existing scenario block, or a block
        with the same hierarchical structure as the
        preprocessed working model.
    clone_first_stage_components : bool
        True to clone first-stage variables
        when transferring attributes to the new block
        to the new block (as opposed to using the objects as
        they are in `from_block`), False otherwise.
    """
    # Note for any of the Vars not copied:
    # - if Var is not a member of an indexed var, then
    #   the 'name' attribute changes from
    #   '{from_block.name}.{var.name}'
    #   to 'scenarios[{scenario_idx}].{var.name}'
    # - otherwise, the name stays the same
    memo = dict()
    if not clone_first_stage_components:
        nonadjustable_comps = from_block.all_nonadjustable_variables
        memo = {id(comp): comp for comp in nonadjustable_comps}

        # we will clone the first-stage constraints
        # (mostly to prevent symbol map name clashes).
        # the duplicate constraints are redundant.
        # consider deactivating these constraints in the
        # off-nominal blocks?

    new_block = from_block.clone(memo=memo)
    master_model.scenarios[scenario_idx].transfer_attributes_from(new_block)

    # update uncertain parameter values in new block
    new_uncertain_params = (
        master_model.scenarios[scenario_idx].uncertain_params
    )
    for param, val in zip(new_uncertain_params, param_realization):
        param.set_value(val)


def new_construct_master_feasibility_problem(master_data, config):
    """
    Construct slack variable minimization problem from the master
    model.

    Slack variables are added only to the performance constraints
    of the blocks added for the current PyROS iteration.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    slack_model : ConcreteModel
        Slack variable model.
    """
    # to prevent use of find_component when copying variable values
    # from the slack model to the master problem later, we will
    # map corresponding variables before/during slack model construction
    varmap_name = unique_component_name(master_data.master_model, 'pyros_var_map')
    setattr(
        master_data.master_model,
        varmap_name,
        list(master_data.master_model.component_data_objects(Var)),
    )

    slack_model = master_data.master_model.clone()

    # construct the variable mapping
    master_data.feasibility_problem_varmap = list(
        zip(
            getattr(master_data.master_model, varmap_name),
            getattr(slack_model, varmap_name),
        )
    )
    delattr(master_data.master_model, varmap_name)
    delattr(slack_model, varmap_name)

    for obj in slack_model.component_data_objects(Objective):
        obj.deactivate()
    iteration = master_data.iteration

    # add slacks only to performance inequality constraints for the newest
    # master block. these should be the only constraints which
    # may have been violated by the previous master and separation
    # solution(s)
    targets = []
    for blk in slack_model.scenarios[iteration, :]:
        targets.extend(blk.effective_performance_inequality_cons)

    # retain original constraint expressions before adding slacks
    # (to facilitate slack initialization and scaling)
    pre_slack_con_exprs = ComponentMap((con, con.body - con.upper) for con in targets)

    # add slack variables and objective
    # inequalities g(v) <= b become g(v) - s^- <= b
    TransformationFactory("core.add_slack_variables").apply_to(
        slack_model,
        targets=targets,
    )
    slack_vars = ComponentSet(
        slack_model
        ._core_add_slack_variables
        .component_data_objects(Var, descend_into=True)
    )

    # initialize slack variables
    for con in pre_slack_con_exprs:
        # get mapping from slack variables to their (linear)
        # coefficients (+/-1) in the updated constraint expressions
        repn = generate_standard_repn(con.body)
        slack_var_coef_map = ComponentMap()
        for idx in range(len(repn.linear_vars)):
            var = repn.linear_vars[idx]
            if var in slack_vars:
                slack_var_coef_map[var] = repn.linear_coefs[idx]

        # use this dict if we elect custom scaling in future
        # slack_substitution_map = dict()
        for slack_var in slack_var_coef_map:
            # coefficient determines whether the slack
            # is a +ve or -ve slack
            if slack_var_coef_map[slack_var] == -1:
                con_slack = max(0, value(pre_slack_con_exprs[con]))
            else:
                con_slack = max(0, -value(pre_slack_con_exprs[con]))

            # initialize slack variable, evaluate scaling coefficient
            slack_var.set_value(con_slack)

            # (we will probably want to change scaling later)
            # # update expression replacement map for slack scaling
            # scaling_coeff = 1  # we may want to change scaling later
            # slack_substitution_map[id(slack_var)] = scaling_coeff * slack_var
            # slack_substitution_map[id(slack_var)] = slack_var

        # # finally, scale slack(s)
        # con.set_value(
        #     (
        #         replace_expressions(con.lower, slack_substitution_map),
        #         replace_expressions(con.body, slack_substitution_map),
        #         replace_expressions(con.upper, slack_substitution_map),
        #     )
        # )

    return slack_model


def solve_master_feasibility_problem(master_data, config):
    """
    Solve a slack variable-based feasibility model derived
    from the master problem. Initialize the master problem
    to the  solution found by the optimizer if solved successfully,
    or to the initial point provided to the solver otherwise.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Solver results.
    """
    model = new_construct_master_feasibility_problem(master_data, config)

    active_obj = next(model.component_data_objects(Objective, active=True))

    config.progress_logger.debug("Solving master feasibility problem")
    config.progress_logger.debug(
        f" Initial objective (total slack): {value(active_obj)}"
    )

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    results = call_solver(
        model=model,
        solver=solver,
        config=config,
        timing_obj=master_data.timing,
        timer_name="main.master_feasibility",
        err_msg=(
            f"Optimizer {repr(solver)} encountered exception "
            "attempting to solve master feasibility problem in iteration "
            f"{master_data.iteration}."
        ),
    )

    feasible_terminations = {
        tc.optimal,
        tc.locallyOptimal,
        tc.globallyOptimal,
        tc.feasible,
    }
    if results.solver.termination_condition in feasible_terminations:
        model.solutions.load_from(results)
        config.progress_logger.debug(
            f" Final objective (total slack): {value(active_obj)}"
        )
        config.progress_logger.debug(
            f" Termination condition: {results.solver.termination_condition}"
        )
        config.progress_logger.debug(
            f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
        )
    else:
        config.progress_logger.warning(
            "Could not successfully solve master feasibility problem "
            f"of iteration {master_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unoptimized point for master problem initialization."
        )

    # load master feasibility point to master model
    for master_var, feas_var in master_data.feasibility_problem_varmap:
        master_var.set_value(feas_var.value, skip_validation=True)

    return results


def new_construct_dr_polishing_problem(master_data, config):
    """
    Construct DR polishing problem from the master problem.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    polishing_model : ConcreteModel
        Polishing model.

    Note
    ----
    Polishing problem is to minimize the L1-norm of the vector of
    all decision rule polynomial terms, subject to the original
    master problem constraints, with all first-stage variables
    (including epigraph) fixed. Optimality of the polished
    DR with respect to the master objective is also enforced.
    """
    master_model = master_data.master_model
    polishing_model = master_model.clone()
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    nominal_eff_var_partitioning = nominal_polishing_block.effective_var_partitioning

    nondr_nonadjustable_vars = (
        nominal_eff_var_partitioning.first_stage_variables
        # fixing epigraph variable constrains the problem
        # to the optimal master problem solution set
        + [nominal_polishing_block.epigraph_var]
    )
    for var in nondr_nonadjustable_vars:
        var.fix()

    # we will add the polishing objective later
    polishing_model.epigraph_obj.deactivate()

    decision_rule_vars = nominal_polishing_block.decision_rule_vars
    nominal_polishing_block.polishing_vars = polishing_vars = []
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        # declare auxiliary 'polishing' variables.
        # these are meant to represent the absolute values
        # of the terms of DR polynomial; we need these for the
        # L1-norm
        indexed_polishing_var = Var(
            list(indexed_dr_var.keys()), domain=NonNegativeReals
        )
        nominal_polishing_block.add_component(
            unique_component_name(nominal_polishing_block, f"dr_polishing_var_{idx}"),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # we need the DR expressions to set up the
    # absolute value constraints and initialize the
    # auxiliary polishing variables
    eff_ss_var_to_dr_expr_pairs = [
        (ss_var, get_dr_expression(nominal_polishing_block, ss_var))
        for ss_var in nominal_eff_var_partitioning.second_stage_variables
    ]

    dr_eq_var_zip = zip(
        polishing_vars,
        eff_ss_var_to_dr_expr_pairs,
    )
    nominal_polishing_block.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (indexed_polishing_var, (ss_var, dr_expr)) in enumerate(dr_eq_var_zip):
        # set up absolute value constraint components
        polishing_absolute_value_lb_cons = Constraint(indexed_polishing_var.index_set())
        polishing_absolute_value_ub_cons = Constraint(indexed_polishing_var.index_set())

        # add indexed constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(polishing_model, f"polishing_abs_val_lb_con_{idx}"),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(polishing_model, f"polishing_abs_val_ub_con_{idx}"),
            polishing_absolute_value_ub_cons,
        )

        # update list of absolute value cons
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        for dr_monomial in dr_expr.args:
            if dr_monomial.is_expression_type():
                # degree > 1 monomial expression of form
                # (product of uncertain params) * dr variable
                dr_var_in_term = dr_monomial.args[-1]
            else:
                # the static term (intercept)
                dr_var_in_term = dr_monomial

            # we want the DR variable and corresponding polishing
            # variable to have the same index
            dr_var_in_term_idx = dr_var_in_term.index()
            polishing_var = indexed_polishing_var[dr_var_in_term_idx]

            # add polishing constraints
            polishing_absolute_value_lb_cons[dr_var_in_term_idx] = (
                -polishing_var - dr_monomial <= 0
            )
            polishing_absolute_value_ub_cons[dr_var_in_term_idx] = (
                dr_monomial - polishing_var <= 0
            )

            # some DR variables may be fixed in the earlier
            # PyROS iterations for efficiency purposes
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[dr_var_in_term_idx].deactivate()
                polishing_absolute_value_ub_cons[dr_var_in_term_idx].deactivate()

            # ensure the polishing constraints
            # are satisfied (to equality) at the initial point
            polishing_var.set_value(abs(value(dr_monomial)))

    # finally, the 1-norm objective
    polishing_model.polishing_obj = Objective(
        expr=sum(sum(polishing_var.values()) for polishing_var in polishing_vars)
    )

    return polishing_model


def minimize_dr_vars(master_data, config):
    """
    Polish decision rule of most recent master problem solution.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.
    """
    # create polishing NLP
    polishing_model = new_construct_dr_polishing_problem(master_data, config)

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    config.progress_logger.debug("Solving DR polishing problem")

    # polishing objective should be consistent with value of sum
    # of absolute values of polynomial DR terms provided
    # auxiliary variables initialized correctly
    polishing_obj = polishing_model.polishing_obj
    config.progress_logger.debug(f" Initial DR norm: {value(polishing_obj)}")

    # === Solve the polishing model
    results = call_solver(
        model=polishing_model,
        solver=solver,
        config=config,
        timing_obj=master_data.timing,
        timer_name="main.dr_polishing",
        err_msg=(
            f"Optimizer {repr(solver)} encountered an exception "
            "attempting to solve decision rule polishing problem "
            f"in iteration {master_data.iteration}"
        ),
    )

    # interested in the time and termination status for debugging
    # purposes
    config.progress_logger.debug(" Done solving DR polishing problem")
    config.progress_logger.debug(
        f"  Termination condition: {results.solver.termination_condition} "
    )
    config.progress_logger.debug(
        f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
    )

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}
    if results.solver.termination_condition not in acceptable:
        # continue with "unpolished" master model solution
        config.progress_logger.warning(
            "Could not successfully solve DR polishing problem "
            f"of iteration {master_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unpolished master problem solution."
        )
        return results, False

    # update master model second-stage, state, and decision rule
    # variables to polishing model solution
    polishing_model.solutions.load_from(results)

    # update master problem variable values
    for idx, blk in master_data.master_model.scenarios.items():
        master_adjustable_vars = blk.all_adjustable_variables
        polishing_adjustable_vars = (
            polishing_model.scenarios[idx].all_adjustable_variables
        )
        adjustable_vars_zip = zip(master_adjustable_vars, polishing_adjustable_vars)
        for master_var, polish_var in adjustable_vars_zip:
            master_var.set_value(value(polish_var))
        dr_var_zip = zip(
            blk.decision_rule_vars,
            polishing_model.scenarios[idx].decision_rule_vars,
        )
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)

    config.progress_logger.debug(f" Optimized DR norm: {value(polishing_obj)}")
    config.progress_logger.debug(" Polished master objective:")

    # print breakdown of objective value of polished master solution
    if config.objective_focus == ObjectiveType.worst_case:
        eval_obj_blk_idx = max(
            master_data.master_model.scenarios.keys(),
            key=lambda idx: value(
                master_data.master_model.scenarios[idx].second_stage_objective
            ),
        )
    else:
        eval_obj_blk_idx = (0, 0)

    # debugging: summarize objective breakdown
    eval_obj_blk = master_data.master_model.scenarios[eval_obj_blk_idx]
    config.progress_logger.debug(
        "  First-stage objective: " f"{value(eval_obj_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        "  Second-stage objective: " f"{value(eval_obj_blk.second_stage_objective)}"
    )
    polished_master_obj = value(
        eval_obj_blk.first_stage_objective + eval_obj_blk.second_stage_objective
    )
    config.progress_logger.debug(f"  Objective: {polished_master_obj}")

    return results, True


def add_p_robust_constraint(master_data, config):
    """
    p-robustness--adds constraints to the master problem ensuring that the
    optimal k-th iteration solution is within (1+rho) of the nominal
    objective. The parameter rho is specified by the user and should be between.
    """
    rho = config.p_robustness['rho']
    model = master_data.master_model
    block_0 = model.scenarios[0, 0]
    frac_nom_cost = (1 + rho) * (
        block_0.first_stage_objective + block_0.second_stage_objective
    )

    for block_k in model.scenarios[master_data.iteration, :]:
        model.p_robust_constraints.add(
            block_k.first_stage_objective + block_k.second_stage_objective
            <= frac_nom_cost
        )
    return


def get_master_dr_degree(master_data, config):
    """
    Determine DR polynomial degree to enforce based on
    the iteration number.

    Currently, the degree is set to:

    - 0 if iteration number is 0
    - min(1, config.decision_rule_order) if iteration number
      otherwise does not exceed number of uncertain parameters
    - min(2, config.decision_rule_order) otherwise.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    int
        DR order, or polynomial degree, to enforce.
    """
    if master_data.iteration == 0:
        return 0
    elif master_data.iteration <= len(config.uncertain_params):
        return min(1, config.decision_rule_order)
    else:
        return min(2, config.decision_rule_order)


def higher_order_decision_rule_efficiency(master_data, config):
    """
    Enforce DR coefficient variable efficiencies for
    master problem-like formulation.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver options.

    Note
    ----
    The DR coefficient variable efficiencies consist of
    setting the degree of the DR polynomial expressions
    by fixing the appropriate variables to 0. The degree
    to be set depends on the iteration number;
    see ``get_master_dr_degree``.
    """
    order_to_enforce = get_master_dr_degree(master_data, config)
    enforce_dr_degree(
        blk=master_data.master_model.scenarios[0, 0],
        config=config,
        degree=order_to_enforce,
    )


def log_master_solve_results(master_model, config, results):
    """
    Log master problem solve results.
    """
    if config.objective_focus == ObjectiveType.worst_case:
        eval_obj_blk_idx = max(
            master_model.scenarios.keys(),
            key=lambda idx: value(
                master_model.scenarios[idx].second_stage_objective
            ),
        )
    else:
        eval_obj_blk_idx = (0, 0)

    eval_obj_blk = master_model.scenarios[eval_obj_blk_idx]
    config.progress_logger.debug(" Optimized master objective breakdown:")
    config.progress_logger.debug(
        f"  First-stage objective: {value(eval_obj_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        f"  Second-stage objective: {value(eval_obj_blk.second_stage_objective)}"
    )
    master_obj = (
        eval_obj_blk.first_stage_objective + eval_obj_blk.second_stage_objective
    )
    config.progress_logger.debug(f"  Objective: {value(master_obj)}")
    config.progress_logger.debug(
        f" Termination condition: {results.solver.termination_condition}"
    )
    config.progress_logger.debug(
        f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
    )


def solver_call_master(master_data, config, solve_data):
    """
    Invoke subsolver(s) on PyROS master problem.

    Parameters
    ----------
    master_data : MasterProblemData
        Container for current master problem and related data.
    config : ConfigDict
        PyROS solver settings.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the master problem. This may be a local or global
        NLP solver.
    solve_data : MasterResult
        Master problem results object. May be empty or contain
        master feasibility problem results.

    Returns
    -------
    master_soln : MasterResult
        Master problem results object, containing master
        model and subsolver results.
    """
    master_model = master_data.master_model
    master_soln = solve_data
    solver_term_cond_dict = {}

    if config.solve_master_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers

    solve_mode = "global" if config.solve_master_globally else "local"
    config.progress_logger.debug("Solving master problem")

    nominal_block = master_model.scenarios[0, 0]
    higher_order_decision_rule_efficiency(master_data, config)

    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(
                f"Invoking backup solver {opt!r} "
                f"(solver {idx + 1} of {len(solvers)}) for "
                f"master problem of iteration {master_data.iteration}."
            )
        results = call_solver(
            model=master_model,
            solver=opt,
            config=config,
            timing_obj=master_data.timing,
            timer_name="main.master",
            err_msg=(
                f"Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) "
                "encountered exception attempting to "
                f"solve master problem in iteration {master_data.iteration}"
            ),
        )

        optimal_termination = check_optimal_termination(results)
        infeasible = results.solver.termination_condition == tc.infeasible

        if optimal_termination:
            master_model.solutions.load_from(results)

        # record master problem termination conditions
        # for this particular subsolver
        # pyros termination condition is determined later in the
        # algorithm
        solver_term_cond_dict[str(opt)] = str(results.solver.termination_condition)
        master_soln.termination_condition = results.solver.termination_condition
        master_soln.pyros_termination_condition = None
        (try_backup, _) = master_soln.master_subsolver_results = (
            process_termination_condition_master_problem(config=config, results=results)
        )

        master_soln.nominal_block = nominal_block
        master_soln.results = results
        master_soln.master_model = master_model

        # if model was solved successfully, update/record the results
        # (nominal block DOF variable and objective values)
        if not try_backup and not infeasible:
            # debugging: log breakdown of master objective
            log_master_solve_results(master_model, config, results)

            master_soln.nominal_block = master_model.scenarios[0, 0]
            master_soln.results = results
            master_soln.master_model = master_model

        # if PyROS time limit exceeded, exit loop and return solution
        if check_time_limit_reached(master_data.timing, config):
            try_backup = False
            master_soln.master_subsolver_results = (
                None,
                pyrosTerminationCondition.time_out,
            )
            master_soln.pyros_termination_condition = (
                pyrosTerminationCondition.time_out
            )

        if not try_backup:
            return master_soln

    # all solvers have failed to return an acceptable status.
    # we will terminate PyROS with subsolver error status.
    # at this point, export subproblem to file, if desired.
    # NOTE: subproblem is written with variables set to their
    #       initial values (not the final subsolver iterate)
    save_dir = config.subproblem_file_directory
    serialization_msg = ""
    if save_dir and config.keepfiles:
        output_problem_path = os.path.join(
            save_dir,
            (
                config.uncertainty_set.type
                + "_"
                + master_data.original_model_name
                + "_master_"
                + str(master_data.iteration)
                + ".bar"
            ),
        )
        master_model.write(
            output_problem_path, io_options={'symbolic_solver_labels': True}
        )
        serialization_msg = (
            " For debugging, problem has been serialized to the file "
            f"{output_problem_path!r}."
        )

    deterministic_model_qual = (
        " (i.e., the deterministic model)" if master_data.iteration == 0 else ""
    )
    deterministic_msg = (
        (
            " Please ensure your deterministic model "
            f"is solvable by at least one of the subordinate {solve_mode} "
            "optimizers provided."
        )
        if master_data.iteration == 0
        else ""
    )
    master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
    config.progress_logger.warning(
        f"Could not successfully solve master problem of iteration "
        f"{master_data.iteration}{deterministic_model_qual} with any of the "
        f"provided subordinate {solve_mode} optimizers. "
        f"(Termination statuses: "
        f"{[term_cond for term_cond in solver_term_cond_dict.values()]}.)"
        f"{deterministic_msg}"
        f"{serialization_msg}"
    )

    return master_soln


def solve_master(master_data, config):
    """
    Solve the master problem
    """
    master_soln = MasterResult()

    # no master feas problem for iteration 0
    if master_data.iteration > 0:
        results = solve_master_feasibility_problem(master_data, config)
        master_soln.feasibility_problem_results = results

        # if pyros time limit reached, load time out status
        # to master results and return to caller
        if check_time_limit_reached(master_data.timing, config):
            # load master model
            master_soln.master_model = master_data.master_model
            master_soln.nominal_block = master_data.master_model.scenarios[0, 0]

            # empty results object, with master solve time of zero
            master_soln.results = SolverResults()
            setattr(master_soln.results.solver, TIC_TOC_SOLVE_TIME_ATTR, 0)

            # PyROS time out status
            master_soln.pyros_termination_condition = (
                pyrosTerminationCondition.time_out
            )
            master_soln.master_subsolver_results = (
                None,
                pyrosTerminationCondition.time_out,
            )
            return master_soln

    return solver_call_master(
        master_data=master_data, config=config, solve_data=master_soln
    )


class NewMasterProblemData:
    """
    Container for objects pertaining to the PyROS master problem.
    """
    def __init__(self, model_data, config):
        """Initialize self (see docstring).

        """
        self.master_model = construct_initial_master_problem(model_data, config)
        # we track the original model name for serialization purposes
        self.original_model_name = model_data.original_model.name
        self.iteration = 0
        self.timing = model_data.timing
        self.config = config

    def solve_master(self):
        """
        Solve the master problem.
        """
        return solve_master(self, self.config)

    def solve_dr_polishing(self):
        """
        Solve the DR polishing problem.
        """
        return minimize_dr_vars(self, self.config)
