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
Methods for constructing and solving PyROS separation problems
and related objects.
"""

from itertools import product
import os

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.core.base import (
    Block,
    Constraint,
    maximize,
    Objective,
    value,
    Var,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
    replace_expressions,
    identify_mutable_parameters,
)

from pyomo.contrib.pyros.solve_data import (
    DiscreteSeparationSolveCallResults,
    SeparationSolveCallResults,
    SeparationLoopResults,
    SeparationResults,
)
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
    ABS_CON_CHECK_FEAS_TOL,
    are_param_vars_fixed_by_bounds,
    call_solver,
    check_time_limit_reached,
)


def add_uncertainty_set_constraints(separation_model, config):
    """
    Add to the separation model constraints restricting
    the uncertain parameter proxy variables to the user-provided
    uncertainty set. Note that inferred interval enclosures
    on the uncertain parameters are also imposed as bounds
    specified on the proxy variables.
    """
    separation_model.uncertainty = Block()
    separation_model.uncertainty.uncertain_param_indexed_var = Var(
        range(config.uncertainty_set.dim),
        initialize={
            idx: nom_val
            for idx, nom_val in enumerate(config.nominal_uncertain_param_vals)
        },
    )
    indexed_param_var = separation_model.uncertainty.uncertain_param_indexed_var
    uncertainty_quantification = (
        config.uncertainty_set.set_as_constraint(
            uncertain_params=indexed_param_var,
            block=separation_model.uncertainty,
        )
    )

    # facilitate retrieval later
    _, uncertainty_cons, param_var_list, aux_vars = uncertainty_quantification
    separation_model.uncertainty.uncertain_param_var_list = param_var_list
    separation_model.uncertainty.auxiliary_var_list = aux_vars
    separation_model.uncertainty.uncertainty_cons_list = uncertainty_cons

    config.uncertainty_set._add_bounds_on_uncertain_parameters(
        uncertain_param_vars=param_var_list,
        global_solver=config.global_solver,
    )

    # preprocess uncertain parameters which have been fixed by bounds
    # in order to simplify the separation problems
    param_var_certain_nomval_zip = zip(
        param_var_list,
        are_param_vars_fixed_by_bounds(param_var_list),
        config.nominal_uncertain_param_vals,
    )
    for idx, (param_var, is_certain, nomval) in enumerate(param_var_certain_nomval_zip):
        if is_certain:
            param_var.fix(nomval)


def construct_separation_problem(model_data, config):
    """
    Construct the separation problem model from the fully preprocessed
    working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    separation_model : ConcreteModel
        Separation problem model.
    """
    separation_model = model_data.working_model.clone()

    # fix/deactivate all nonadjustable components
    for var in separation_model.all_nonadjustable_variables:
        var.fix()
    nonadjustable_cons = (
        separation_model.effective_first_stage_inequality_cons
        + separation_model.effective_first_stage_equality_cons
    )
    for nadjcon in nonadjustable_cons:
        nadjcon.deactivate()

    # add block for the uncertainty set quantification
    add_uncertainty_set_constraints(separation_model, config)

    # the uncertain params function as decision variables
    # in the separation problems.
    # note: expression replacement is performed only for
    #       the active constraints
    uncertain_params = separation_model.uncertain_params
    uncertain_param_vars = separation_model.uncertainty.uncertain_param_var_list
    param_id_to_var_map = {
        id(param): var
        for param, var in zip(uncertain_params, uncertain_param_vars)
    }
    uncertain_params_set = ComponentSet(uncertain_params)
    adjustable_cons = (
        separation_model.effective_performance_inequality_cons
        + separation_model.effective_performance_equality_cons
        + separation_model.decision_rule_eqns
    )
    for adjcon in adjustable_cons:
        uncertain_params_in_con = ComponentSet(
            identify_mutable_parameters(adjcon.expr)
        ) & uncertain_params_set
        if uncertain_params_in_con:
            adjcon.set_value(
                replace_expressions(adjcon.expr, substitution_map=param_id_to_var_map)
            )

    # performance inequality constraint expressions
    # become maximization objectives in the separation problems
    separation_model.perf_ineq_con_to_obj_map = ComponentMap()
    perf_ineq_cons = separation_model.effective_performance_inequality_cons
    for idx, perf_con in enumerate(perf_ineq_cons):
        perf_con.deactivate()
        separation_obj = Objective(expr=perf_con.body - perf_con.upper, sense=maximize)
        separation_model.add_component(
            f"separation_obj_{idx}",
            separation_obj,
        )
        separation_model.perf_ineq_con_to_obj_map[perf_con] = separation_obj
        separation_obj.deactivate()

    return separation_model


def get_sep_objective_values(separation_data, config, perf_cons):
    """
    Evaluate performance constraint functions at current
    separation solution.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    perf_cons : list of Constraint
        Performance constraints to be evaluated.

    Returns
    -------
    violations : ComponentMap
        Mapping from performance constraints to violation values.
    """
    con_to_obj_map = separation_data.separation_model.perf_ineq_con_to_obj_map
    violations = ComponentMap()

    user_var_partitioning = separation_data.separation_model.user_var_partitioning
    first_stage_variables = user_var_partitioning.first_stage_variables
    second_stage_variables = user_var_partitioning.second_stage_variables

    for perf_con in perf_cons:
        obj = con_to_obj_map[perf_con]
        try:
            violations[perf_con] = value(obj.expr)
        except ValueError:
            for v in first_stage_variables:
                config.progress_logger.info(v.name + " " + str(v.value))
            for v in second_stage_variables:
                config.progress_logger.info(v.name + " " + str(v.value))
            raise ArithmeticError(
                f"Evaluation of performance constraint {perf_con.name} "
                f"(separation objective {obj.name}) "
                "led to a math domain error. "
                "Does the performance constraint expression "
                "contain log(x) or 1/x functions "
                "or others with tricky domains?"
            )

    return violations


def get_argmax_sum_violations(solver_call_results_map, perf_cons_to_evaluate):
    """
    Get key of entry of `solver_call_results_map` which contains
    separation problem solution with maximal sum of performance
    constraint violations over a specified sequence of performance
    constraints.

    Parameters
    ----------
    solver_call_results : ComponentMap
        Mapping from performance constraints to corresponding
        separation solver call results.
    perf_cons_to_evaluate : list of Constraints
        Performance constraints to consider for evaluating
        maximal sum.

    Returns
    -------
    worst_perf_con : None or Constraint
        Performance constraint corresponding to solver call
        results object containing solution with maximal sum
        of violations across all performance constraints.
        If ``found_violation`` attribute of all value entries of
        `solver_call_results_map` is False, then `None` is
        returned, as this means none of the performance constraints
        were found to be violated.
    """
    # get indices of performance constraints for which violation found
    idx_to_perf_con_map = {
        idx: perf_con for idx, perf_con in enumerate(solver_call_results_map)
    }
    idxs_of_violated_cons = [
        idx
        for idx, perf_con in idx_to_perf_con_map.items()
        if solver_call_results_map[perf_con].found_violation
    ]

    num_violated_cons = len(idxs_of_violated_cons)

    if num_violated_cons == 0:
        return None

    # assemble square matrix (2D array) of constraint violations.
    # matrix size: number of constraints for which violation was found
    # each row corresponds to a performance constraint
    # each column corresponds to a separation problem solution
    violations_arr = np.zeros(shape=(num_violated_cons, num_violated_cons))
    idxs_product = product(
        enumerate(idxs_of_violated_cons), enumerate(idxs_of_violated_cons)
    )
    for (row_idx, viol_con_idx), (col_idx, viol_param_idx) in idxs_product:
        violations_arr[row_idx, col_idx] = max(
            0,
            (
                # violation of this row's performance constraint
                # by this column's separation solution
                # if separation problems were solved globally,
                # then diagonal entries should be the largest in each row
                solver_call_results_map[
                    idx_to_perf_con_map[viol_param_idx]
                ].scaled_violations[idx_to_perf_con_map[viol_con_idx]]
            ),
        )

    worst_col_idx = np.argmax(np.sum(violations_arr, axis=0))

    return idx_to_perf_con_map[idxs_of_violated_cons[worst_col_idx]]


def solve_separation_problem(separation_data, master_data, config):
    """
    Solve PyROS separation problems.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    pyros.solve_data.SeparationResults
        Separation problem solve results.
    """
    run_local = not config.bypass_local_separation
    run_global = config.bypass_local_separation

    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    )

    if run_local:
        local_separation_loop_results = perform_separation_loop(
            separation_data=separation_data,
            master_data=master_data,
            config=config,
            solve_globally=False,
        )
        run_global = not (
            local_separation_loop_results.found_violation
            or uncertainty_set_is_discrete
            or local_separation_loop_results.subsolver_error
            or local_separation_loop_results.time_out
            or config.bypass_global_separation
        )
    else:
        local_separation_loop_results = None

    if run_global:
        global_separation_loop_results = perform_separation_loop(
            separation_data=separation_data,
            master_data=master_data,
            config=config,
            solve_globally=True,
        )
    else:
        global_separation_loop_results = None

    return SeparationResults(
        local_separation_loop_results=local_separation_loop_results,
        global_separation_loop_results=global_separation_loop_results,
    )


def evaluate_violations_by_nominal_master(
        separation_data,
        master_data,
        performance_cons,
        ):
    """
    Evaluate violation of performance constraints by
    variables in nominal block of most recent master
    problem.

    Returns
    -------
    nom_perf_con_violations : dict
        Mapping from performance constraint names
        to floats equal to violations by nominal master
        problem variables.
    """
    nom_perf_con_violations = ComponentMap()
    for perf_con in performance_cons:
        nom_violation = value(
            master_data.master_model.scenarios[0, 0].find_component(perf_con)
        )
        nom_perf_con_violations[perf_con] = nom_violation

    return nom_perf_con_violations


def group_performance_constraints_by_priority(separation_data, config):
    """
    Group model performance constraints by separation priority.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        User-specified PyROS solve options.

    Returns
    -------
    dict
        Mapping from an int to a list of performance constraints
        (Constraint objects),
        for which the int is equal to the specified priority.
        Keys are sorted in descending order
        (i.e. highest priority first).
    """
    all_perf_cons = (
        separation_data.separation_model.effective_performance_inequality_cons
    )
    separation_priority_groups = dict()
    config_sep_priority_dict = config.separation_priority_order
    for perf_con in all_perf_cons:
        # by default, priority set to 0
        priority = config_sep_priority_dict.get(perf_con.name, 0)
        cons_with_same_priority = separation_priority_groups.setdefault(priority, [])
        cons_with_same_priority.append(perf_con)

    # sort separation priority groups
    return {
        priority: perf_cons
        for priority, perf_cons in sorted(
            separation_priority_groups.items(), reverse=True
        )
    }


def get_worst_discrete_separation_solution(
    performance_constraint,
    config,
    perf_cons_to_evaluate,
    discrete_solve_results,
):
    """
    Determine separation solution (and therefore worst-case
    uncertain parameter realization) with maximum violation
    of specified performance constraint.

    Parameters
    ----------
    performance_constraint : Constraint
        Performance constraint of interest.
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        User-specified PyROS solver settings.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints for which to report violations
        by separation solution.
    discrete_solve_results : DiscreteSeparationSolveCallResults
        Separation problem solutions corresponding to the
        uncertain parameter scenarios listed in
        ``config.uncertainty_set.scenarios``.

    Returns
    -------
    SeparationSolveCallResult
        Solver call result for performance constraint of interest.
    """
    # violation of specified performance constraint by separation
    # problem solutions for all scenarios
    violations_of_perf_con = [
        solve_call_res.scaled_violations[performance_constraint]
        for solve_call_res in discrete_solve_results.solver_call_results.values()
    ]

    list_of_scenario_idxs = list(discrete_solve_results.solver_call_results.keys())

    # determine separation solution for which scaled violation of this
    # performance constraint is the worst
    worst_case_res = discrete_solve_results.solver_call_results[
        list_of_scenario_idxs[np.argmax(violations_of_perf_con)]
    ]
    worst_case_violation = np.max(violations_of_perf_con)
    assert worst_case_violation in worst_case_res.scaled_violations.values()

    # evaluate violations for specified performance constraints
    eval_perf_con_scaled_violations = ComponentMap(
        (perf_con, worst_case_res.scaled_violations[perf_con])
        for perf_con in perf_cons_to_evaluate
    )

    # discrete separation solutions were obtained by optimizing
    # just one performance constraint, as an efficiency.
    # if the constraint passed to this routine is the same as the
    # constraint used to obtain the solutions, then we bundle
    # the separation solve call results into a single list.
    # otherwise, we return an empty list, as we did not need to call
    # subsolvers for the other performance constraints
    is_optimized_performance_con = (
        performance_constraint is discrete_solve_results.performance_constraint
    )
    if is_optimized_performance_con:
        results_list = [
            res
            for solve_call_results
            in discrete_solve_results.solver_call_results.values()
            for res in solve_call_results.results_list
        ]
    else:
        results_list = []

    return SeparationSolveCallResults(
        solved_globally=worst_case_res.solved_globally,
        results_list=results_list,
        scaled_violations=eval_perf_con_scaled_violations,
        violating_param_realization=worst_case_res.violating_param_realization,
        variable_values=worst_case_res.variable_values,
        found_violation=(worst_case_violation > config.robust_feasibility_tolerance),
        time_out=False,
        subsolver_error=False,
        discrete_set_scenario_index=worst_case_res.discrete_set_scenario_index,
    )


def get_con_name_repr(separation_model, con, with_obj_name=True):
    """
    Get string representation of performance constraint
    and the objective to which it has
    been mapped.

    Parameters
    ----------
    separation_model : ConcreteModel
        Separation model.
    con : ScalarConstraint or ConstraintData
        Constraint for which to get the representation.
    with_obj_name : bool, optional
        Include name of separation model objective to which
        constraint is mapped. Applicable only to performance
        constraints of the separation problem.

    Returns
    -------
    str
        Constraint name representation.
    """
    qual_str = ""
    if with_obj_name:
        objectives_map = separation_model.perf_ineq_con_to_obj_map
        separation_obj = objectives_map[con]
        qual_str = f" (mapped to objective {separation_obj.name!r})"

    return f"{con.name!r}{qual_str}"


def perform_separation_loop(separation_data, master_data, config, solve_globally):
    """
    Loop through, and solve, PyROS separation problems to
    desired optimality condition.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve separation problems locally.

    Returns
    -------
    pyros.solve_data.SeparationLoopResults
        Separation problem solve results.
    """
    all_performance_constraints = (
        separation_data.separation_model.effective_performance_inequality_cons
    )
    if not all_performance_constraints:
        # robustness certified: no separation problems to solve
        return SeparationLoopResults(
            solver_call_results=ComponentMap(),
            solved_globally=solve_globally,
            worst_case_perf_con=None,
        )

    # needed for normalizing separation solution constraint violations
    separation_data.nom_perf_con_violations = evaluate_violations_by_nominal_master(
        separation_data=separation_data,
        master_data=master_data,
        performance_cons=all_performance_constraints,
    )
    sorted_priority_groups = group_performance_constraints_by_priority(
        separation_data, config
    )
    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    )

    if uncertainty_set_is_discrete:
        all_scenarios_exhausted = len(separation_data.idxs_of_master_scenarios) == len(
            config.uncertainty_set.scenarios
        )
        if all_scenarios_exhausted:
            # robustness certified: entire uncertainty set already
            # accounted for in master
            return SeparationLoopResults(
                solver_call_results=ComponentMap(),
                solved_globally=solve_globally,
                worst_case_perf_con=None,
                all_discrete_scenarios_exhausted=True,
            )

        perf_con_to_maximize = sorted_priority_groups[
            max(sorted_priority_groups.keys())
        ][0]

        # efficiency: evaluate all separation problem solutions in
        # advance of entering loop
        discrete_sep_results = discrete_solve(
            separation_data=separation_data,
            master_data=master_data,
            config=config,
            solve_globally=solve_globally,
            perf_con_to_maximize=perf_con_to_maximize,
            perf_cons_to_evaluate=all_performance_constraints,
        )

        termination_not_ok = (
            discrete_sep_results.time_out or discrete_sep_results.subsolver_error
        )
        if termination_not_ok:
            single_solver_call_res = ComponentMap()
            results_list = [
                res
                for solve_call_results
                in discrete_sep_results.solver_call_results.values()
                for res in solve_call_results.results_list
            ]
            single_solver_call_res[perf_con_to_maximize] = (
                # not the neatest assembly,
                # but should maintain accuracy of total solve times
                # and overall outcome
                SeparationSolveCallResults(
                    solved_globally=solve_globally,
                    results_list=results_list,
                    time_out=discrete_sep_results.time_out,
                    subsolver_error=discrete_sep_results.subsolver_error,
                )
            )
            return SeparationLoopResults(
                solver_call_results=single_solver_call_res,
                solved_globally=solve_globally,
                worst_case_perf_con=None,
            )

    all_solve_call_results = ComponentMap()
    priority_groups_enum = enumerate(sorted_priority_groups.items())
    for group_idx, (priority, perf_constraints) in priority_groups_enum:
        priority_group_solve_call_results = ComponentMap()
        for idx, perf_con in enumerate(perf_constraints):
            # log progress of separation loop
            solve_adverb = "Globally" if solve_globally else "Locally"
            config.progress_logger.debug(
                f"{solve_adverb} separating performance constraint "
                f"{get_con_name_repr(separation_data.separation_model, perf_con)} "
                f"(priority {priority}, priority group {group_idx + 1} of "
                f"{len(sorted_priority_groups)}, "
                f"constraint {idx + 1} of {len(perf_constraints)} "
                "in priority group, "
                f"{len(all_solve_call_results) + idx + 1} of "
                f"{len(all_performance_constraints)} total)"
            )

            # solve separation problem for this performance constraint
            if uncertainty_set_is_discrete:
                solve_call_results = get_worst_discrete_separation_solution(
                    performance_constraint=perf_con,
                    config=config,
                    perf_cons_to_evaluate=all_performance_constraints,
                    discrete_solve_results=discrete_sep_results,
                )
            else:
                solve_call_results = solver_call_separation(
                    separation_data=separation_data,
                    master_data=master_data,
                    config=config,
                    solve_globally=solve_globally,
                    perf_con_to_maximize=perf_con,
                    perf_cons_to_evaluate=all_performance_constraints,
                )

            priority_group_solve_call_results[perf_con] = solve_call_results

            termination_not_ok = (
                solve_call_results.time_out or solve_call_results.subsolver_error
            )
            if termination_not_ok:
                all_solve_call_results.update(priority_group_solve_call_results)
                return SeparationLoopResults(
                    solver_call_results=all_solve_call_results,
                    solved_globally=solve_globally,
                    worst_case_perf_con=None,
                )

        all_solve_call_results.update(priority_group_solve_call_results)

        # there may be multiple separation problem solutions
        # found to have violated a performance constraint.
        # we choose just one for master problem of next iteration
        worst_case_perf_con = get_argmax_sum_violations(
            solver_call_results_map=all_solve_call_results,
            perf_cons_to_evaluate=perf_constraints,
        )
        if worst_case_perf_con is not None:
            # take note of chosen separation solution
            worst_case_res = all_solve_call_results[worst_case_perf_con]
            if uncertainty_set_is_discrete:
                separation_data.idxs_of_master_scenarios.append(
                    worst_case_res.discrete_set_scenario_index
                )

            # # auxiliary log messages
            violated_con_names = "\n ".join(
                get_con_name_repr(separation_data.separation_model, con)
                for con, res in all_solve_call_results.items()
                if res.found_violation
            )
            config.progress_logger.debug(
                f"Violated constraints:\n {violated_con_names} "
            )
            config.progress_logger.debug(
                "Worst-case constraint: "
                f"{get_con_name_repr(separation_data.separation_model, worst_case_perf_con)} "
                "under realization "
                f"{worst_case_res.violating_param_realization}."
            )
            config.progress_logger.debug(
                f"Maximal scaled violation "
                f"{worst_case_res.scaled_violations[worst_case_perf_con]} "
                "from this constraint "
                "exceeds the robust feasibility tolerance "
                f"{config.robust_feasibility_tolerance}"
            )

            # violating separation problem solution now chosen.
            # exit loop
            break
        else:
            config.progress_logger.debug("No violated performance constraints found.")

    return SeparationLoopResults(
        solver_call_results=all_solve_call_results,
        solved_globally=solve_globally,
        worst_case_perf_con=worst_case_perf_con,
    )


def evaluate_performance_constraint_violations(
    separation_data, config, perf_con_to_maximize, perf_cons_to_evaluate
):
    """
    Evaluate the inequality constraint function violations
    of the current separation model solution, and store the
    results in a given `SeparationResult` object.
    Also, determine whether the separation solution violates
    the inequality constraint whose body is the model's
    active objective.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Object containing the separation model.
    config : ConfigDict
        PyROS solver settings.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to
        be evaluated at the current separation problem
        solution.
        Exactly one of these constraints should be mapped
        to an active Objective in the separation model.

    Returns
    -------
    violating_param_realization : list of float
        Uncertain parameter realization corresponding to maximum
        constraint violation.
    scaled_violations : ComponentMap
        Mapping from performance constraints to be evaluated
        to their violations by the separation problem solution.
    constraint_violated : bool
        True if performance constraint mapped to active
        separation model Objective is violated (beyond tolerance),
        False otherwise

    Raises
    ------
    ValueError
        If `perf_cons_to_evaluate` does not contain exactly
        1 entry which can be mapped to an active Objective
        of ``model_data.separation_model``.
    """
    # parameter realization for current separation problem solution
    uncertain_param_vars = (
        separation_data.separation_model.uncertainty.uncertain_param_var_list
    )
    violating_param_realization = list(
        param_var.value for param_var in uncertain_param_vars
    )

    # evaluate violations for all performance constraints provided
    violations_by_sep_solution = get_sep_objective_values(
        separation_data=separation_data, config=config, perf_cons=perf_cons_to_evaluate
    )

    # normalize constraint violation: i.e. divide by
    # absolute value of constraint expression evaluated at
    # nominal master solution (if expression value is large enough)
    scaled_violations = ComponentMap()
    for perf_con, sep_sol_violation in violations_by_sep_solution.items():
        scaled_violation = sep_sol_violation / max(
            1, abs(separation_data.nom_perf_con_violations[perf_con])
        )
        scaled_violations[perf_con] = scaled_violation
        if perf_con is perf_con_to_maximize:
            scaled_active_obj_violation = scaled_violation

    constraint_violated = (
        scaled_active_obj_violation > config.robust_feasibility_tolerance
    )

    return (violating_param_realization, scaled_violations, constraint_violated)


def initialize_separation(perf_con_to_maximize, separation_data, master_data, config):
    """
    Initialize separation problem variables using the solution
    to the most recent master problem.

    Parameters
    ----------
    perf_con_to_maximize : ConstraintData
        Performance constraint whose violation is to be maximized
        for the separation problem of interest.
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.

    Note
    ----
    The point to which the separation model is initialized should,
    in general, be feasible, provided the set does not have a
    discrete geometry (as there is no master model block corresponding
    to any of the remaining discrete scenarios against which we
    separate). If the uncertainty set constraints involve
    auxiliary variables, then some uncertainty set constraints
    may be violated.
    """
    master_model = master_data.master_model
    sep_model = separation_data.separation_model

    def eval_master_violation(scenario_idx):
        """
        Evaluate violation of `perf_con` by variables of
        specified master block.
        """
        master_con = (
            master_model.scenarios[scenario_idx].find_component(perf_con_to_maximize)
        )
        return value(master_con)

    # initialize from master block with max violation of the
    # performance constraint of interest. This gives the best known
    # feasible solution (for case of non-discrete uncertainty sets).
    worst_master_block_idx = max(
        master_model.scenarios.keys(),
        key=eval_master_violation,
    )
    worst_case_master_blk = master_model.scenarios[worst_master_block_idx]
    for sep_var in sep_model.all_variables:
        master_var = worst_case_master_blk.find_component(sep_var)
        sep_var.set_value(value(master_var, exception=False))

    # for discrete uncertainty sets, the uncertain parameters
    # have already been addressed
    if config.uncertainty_set.geometry != Geometry.DISCRETE_SCENARIOS:
        param_vars = sep_model.uncertainty.uncertain_param_var_list
        param_values = separation_data.points_added_to_master[
            worst_master_block_idx
        ]
        for param_var, val in zip(param_vars, param_values):
            param_var.set_value(val)

    # confirm the initial point is feasible for cases where
    # we expect it to be (i.e. non-discrete uncertainty sets).
    # otherwise, log the violated constraints
    # NOTE: some uncertainty set constraints may be violated
    #       at the initial point if there are auxiliary variables
    #       (e.g. factor model, cardinality sets).
    #       revisit initialization of auxiliary uncertainty set
    #       variables later
    tol = ABS_CON_CHECK_FEAS_TOL
    perf_con_name_repr = get_con_name_repr(
        separation_model=sep_model,
        con=perf_con_to_maximize,
        with_obj_name=True,
    )
    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry is Geometry.DISCRETE_SCENARIOS
    )
    for con in sep_model.component_data_objects(Constraint, active=True):
        lslack, uslack = con.lslack(), con.uslack()
        if (lslack < -tol or uslack < -tol) and not uncertainty_set_is_discrete:
            con_name_repr = get_con_name_repr(
                separation_model=sep_model,
                con=con,
                with_obj_name=False,
            )
            config.progress_logger.debug(
                f"Initial point for separation of performance constraint "
                f"{perf_con_name_repr} violates the model constraint "
                f"{con_name_repr} by more than {tol}. "
                f"(lslack={con.lslack()}, uslack={con.uslack()})"
            )


locally_acceptable = {tc.optimal, tc.locallyOptimal, tc.globallyOptimal}
globally_acceptable = {tc.optimal, tc.globallyOptimal}


def solver_call_separation(
    separation_data, master_data, config,
    solve_globally, perf_con_to_maximize, perf_cons_to_evaluate
):
    """
    Invoke subordinate solver(s) on separation problem.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve locally.
    perf_con_to_maximize : Constraint
        Performance constraint for which to solve separation problem.
        Informs the objective (constraint violation) to maximize.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the separation problem solution
        obtained.

    Returns
    -------
    solve_call_results : pyros.solve_data.SeparationSolveCallResults
        Solve results for separation problem of interest.
    """
    # prepare the problem
    separation_model = separation_data.separation_model
    objectives_map = separation_data.separation_model.perf_ineq_con_to_obj_map
    separation_obj = objectives_map[perf_con_to_maximize]
    initialize_separation(perf_con_to_maximize, separation_data, master_data, config)
    separation_obj.activate()

    # get name of constraint for loggers
    con_name_repr = get_con_name_repr(
        separation_model=separation_model,
        con=perf_con_to_maximize,
        with_obj_name=True,
    )

    # keep track of solver statuses for output logging
    solve_mode = "global" if solve_globally else "local"
    solver_status_dict = {}
    if solve_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers
    solve_mode_adverb = "globally" if solve_globally else "locally"

    solve_call_results = SeparationSolveCallResults(
        solved_globally=solve_globally,
        time_out=False,
        results_list=[],
        found_violation=False,
        subsolver_error=False,
    )
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(
                f"Invoking backup solver {opt!r} "
                f"(solver {idx + 1} of {len(solvers)}) for {solve_mode} "
                f"separation of performance constraint {con_name_repr} "
                f"in iteration {separation_data.iteration}."
            )
        results = call_solver(
            model=separation_model,
            solver=opt,
            config=config,
            timing_obj=separation_data.timing,
            timer_name=f"main.{solve_mode}_separation",
            err_msg=(
                f"Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) "
                f"encountered exception attempting "
                f"to {solve_mode_adverb} solve separation problem for constraint "
                f"{con_name_repr} in iteration {separation_data.iteration}."
            ),
        )

        # record termination condition for this particular solver
        solver_status_dict[str(opt)] = results.solver.termination_condition
        solve_call_results.results_list.append(results)

        # has PyROS time limit been reached?
        if check_time_limit_reached(separation_data.timing, config):
            solve_call_results.time_out = True
            separation_obj.deactivate()
            return solve_call_results

        # if separation problem solved to optimality, record results
        # and exit
        acceptable_conditions = (
            globally_acceptable if solve_globally else locally_acceptable
        )
        optimal_termination = solve_call_results.termination_acceptable(
            acceptable_conditions
        )
        if optimal_termination:
            separation_model.solutions.load_from(results)

            # record second-stage and state variable values
            solve_call_results.variable_values = ComponentMap()
            for var in separation_model.all_adjustable_variables:
                solve_call_results.variable_values[var] = value(var)

            # record uncertain parameter realization
            #   and constraint violations
            (
                solve_call_results.violating_param_realization,
                solve_call_results.scaled_violations,
                solve_call_results.found_violation,
            ) = evaluate_performance_constraint_violations(
                separation_data, config, perf_con_to_maximize, perf_cons_to_evaluate
            )

            separation_obj.deactivate()

            return solve_call_results
        else:
            config.progress_logger.debug(
                f"Solver {opt} ({idx + 1} of {len(solvers)}) "
                f"failed for {solve_mode} separation of performance "
                f"constraint {con_name_repr} in iteration "
                f"{separation_data.iteration}. Termination condition: "
                f"{results.solver.termination_condition!r}."
            )
            config.progress_logger.debug(f"Results:\n{results.solver}")

    # All subordinate solvers failed to optimize model to appropriate
    # termination condition. PyROS will terminate with subsolver
    # error. At this point, export model if desired
    solve_call_results.subsolver_error = True
    save_dir = config.subproblem_file_directory
    serialization_msg = ""
    if save_dir and config.keepfiles:
        objective = separation_obj.name
        output_problem_path = os.path.join(
            save_dir,
            (
                config.uncertainty_set.type
                + "_"
                + separation_model.name
                + "_separation_"
                + str(separation_data.iteration)
                + "_obj_"
                + objective
                + ".bar"
            ),
        )
        separation_model.write(
            output_problem_path, io_options={'symbolic_solver_labels': True}
        )
        serialization_msg = (
            " For debugging, problem has been serialized to the file "
            f"{output_problem_path!r}."
        )
    solve_call_results.message = (
        "Could not successfully solve separation problem of iteration "
        f"{separation_data.iteration} "
        f"for performance constraint {con_name_repr} with any of the "
        f"provided subordinate {solve_mode} optimizers. "
        f"(Termination statuses: "
        f"{[str(term_cond) for term_cond in solver_status_dict.values()]}.)"
        f"{serialization_msg}"
    )
    config.progress_logger.warning(solve_call_results.message)

    separation_obj.deactivate()

    return solve_call_results


def discrete_solve(
    separation_data, master_data, config,
    solve_globally, perf_con_to_maximize, perf_cons_to_evaluate
):
    """
    Obtain separation problem solution for each scenario
    of the uncertainty set not already added to the most
    recent master problem.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the model.
    solve_globally : bool
        Is separation problem to be solved globally.
    perf_con_to_maximize : Constraint
        Performance constraint for which to solve separation
        problem.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the each of separation problem solutions
        obtained.

    Returns
    -------
    discrete_separation_results : DiscreteSeparationSolveCallResults
        Separation solver call results on performance constraint
        of interest for every scenario considered.

    Notes
    -----
    Since we assume that models passed to PyROS are such that the DOF
    variables and uncertain parameter values uniquely define the state
    variables, this method need be only be invoked once per separation
    loop. Subject to our assumption, the choice of objective
    (``perf_con_to_maximize``) should not affect the solutions returned
    beyond subsolver tolerances. For other performance constraints, the
    optimal separation problem solution can then be evaluated by simple
    enumeration of the solutions returned by this function, since for
    discrete uncertainty sets, the number of feasible separation
    solutions is, under our assumption, merely equal to the number
    of scenarios in the uncertainty set.
    """

    uncertain_param_vars = list(
        separation_data.separation_model.uncertainty.uncertain_param_var_list
    )

    # skip scenarios already added to most recent master problem
    master_scenario_idxs = separation_data.idxs_of_master_scenarios
    scenario_idxs_to_separate = [
        idx
        for idx, _ in enumerate(config.uncertainty_set.scenarios)
        if idx not in master_scenario_idxs
    ]

    solve_call_results_dict = {}
    for scenario_idx in scenario_idxs_to_separate:
        # fix uncertain parameters to scenario value
        # hence, no need to activate uncertainty set constraints
        scenario = config.uncertainty_set.scenarios[scenario_idx]
        for param, coord_val in zip(uncertain_param_vars, scenario):
            param.fix(coord_val)

        # obtain separation problem solution
        solve_call_results = solver_call_separation(
            separation_data=separation_data,
            master_data=master_data,
            config=config,
            solve_globally=solve_globally,
            perf_con_to_maximize=perf_con_to_maximize,
            perf_cons_to_evaluate=perf_cons_to_evaluate,
        )
        solve_call_results.discrete_set_scenario_index = scenario_idx
        solve_call_results_dict[scenario_idx] = solve_call_results

        # halt at first encounter of unacceptable termination
        termination_not_ok = (
            solve_call_results.subsolver_error or solve_call_results.time_out
        )
        if termination_not_ok:
            break

    return DiscreteSeparationSolveCallResults(
        solved_globally=solve_globally,
        solver_call_results=solve_call_results_dict,
        performance_constraint=perf_con_to_maximize,
    )


class NewSeparationProblemData:
    """
    Container for objects related to the PyROS separation problem.
    """
    def __init__(self, model_data, config):
        """Initialize self (see class docstring).

        """
        self.separation_model = construct_separation_problem(model_data, config)
        self.timing = model_data.timing
        self.iteration = 0
        self.config = config
        self.points_added_to_master = {(0, 0): config.nominal_uncertain_param_vals}
        self.constraint_violations = []
        self.total_global_separation_solves = 0
        self.separation_problem_subsolver_statuses = []

        if config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS:
            self.idxs_of_master_scenarios = [
                config.uncertainty_set.scenarios.index(
                    tuple(config.nominal_uncertain_param_vals)
                )
            ]
        else:
            self.idxs_of_master_scenarios = None

    def solve_separation(self, master_data):
        """
        Solve the separation problem.
        """
        return solve_separation_problem(self, master_data, self.config)
