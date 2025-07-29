#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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

from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.core.base import Block, Constraint, maximize, Objective, value, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
    replace_expressions,
    identify_variables,
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
    call_solver,
    check_time_limit_reached,
    get_all_first_stage_eq_cons,
    write_subproblem,
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
    uncertainty_quantification = config.uncertainty_set.set_as_constraint(
        uncertain_params=indexed_param_var, block=separation_model.uncertainty
    )

    # facilitate retrieval later
    _, uncertainty_cons, param_var_list, aux_vars = uncertainty_quantification
    separation_model.uncertainty.uncertain_param_var_list = param_var_list
    separation_model.uncertainty.auxiliary_var_list = aux_vars
    separation_model.uncertainty.uncertainty_cons_list = uncertainty_cons

    config.uncertainty_set._add_bounds_on_uncertain_parameters(
        uncertain_param_vars=param_var_list, global_solver=config.global_solver
    )
    if aux_vars:
        aux_var_vals = config.uncertainty_set.compute_auxiliary_uncertain_param_vals(
            point=config.nominal_uncertain_param_vals, solver=config.global_solver
        )
        for auxvar, auxval in zip(aux_vars, aux_var_vals):
            auxvar.set_value(auxval)

    # fix the effectively certain parameters
    param_val_enum_zip = enumerate(
        zip(param_var_list, config.nominal_uncertain_param_vals)
    )
    fixed_param_var_set = ComponentSet()
    for idx, (param_var, nomval) in param_val_enum_zip:
        if idx not in separation_model.effective_uncertain_dimensions:
            param_var.fix(nomval)
            fixed_param_var_set.add(param_var)

    # deactivate constraints that depend on only the
    # effective certain parameters
    separation_model.uncertainty.certain_param_var_cons = []
    for con in uncertainty_cons:
        unfixed_param_vars_in_con = (
            ComponentSet(identify_variables(con.expr)) - fixed_param_var_set
        )
        if not unfixed_param_vars_in_con:
            con.deactivate()
            separation_model.uncertainty.certain_param_var_cons.append(con)


def construct_separation_problem(model_data):
    """
    Construct the separation problem model from the fully preprocessed
    working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.

    Returns
    -------
    separation_model : ConcreteModel
        Separation problem model.
    """
    config = model_data.config
    separation_model = model_data.working_model.clone()

    # fix/deactivate all nonadjustable components
    for var in separation_model.all_nonadjustable_variables:
        var.fix()
    for fs_eqcon in get_all_first_stage_eq_cons(separation_model):
        fs_eqcon.deactivate()
    for fs_ineqcon in separation_model.first_stage.inequality_cons.values():
        fs_ineqcon.deactivate()

    # add block for the uncertainty set quantification
    add_uncertainty_set_constraints(separation_model, config)

    # the uncertain params function as decision variables
    # in the separation problems.
    # note: expression replacement is performed only for
    #       the active constraints
    uncertain_params = separation_model.uncertain_params
    uncertain_param_vars = separation_model.uncertainty.uncertain_param_var_list
    param_id_to_var_map = {
        id(param): var for param, var in zip(uncertain_params, uncertain_param_vars)
    }
    uncertain_params_set = ComponentSet(uncertain_params)
    adjustable_cons = (
        list(separation_model.second_stage.inequality_cons.values())
        + list(separation_model.second_stage.equality_cons.values())
        + list(separation_model.second_stage.decision_rule_eqns.values())
    )
    for adjcon in adjustable_cons:
        uncertain_params_in_con = (
            ComponentSet(identify_mutable_parameters(adjcon.expr))
            & uncertain_params_set
        )
        if uncertain_params_in_con:
            adjcon.set_value(
                replace_expressions(adjcon.expr, substitution_map=param_id_to_var_map)
            )

    # second-stage inequality constraint expressions
    # become maximization objectives in the separation problems
    separation_model.second_stage_ineq_con_to_obj_map = ComponentMap()
    ss_ineq_cons = separation_model.second_stage.inequality_cons.values()
    for idx, ss_ineq_con in enumerate(ss_ineq_cons):
        ss_ineq_con.deactivate()
        separation_obj = Objective(
            expr=ss_ineq_con.body - ss_ineq_con.upper, sense=maximize
        )
        separation_model.add_component(f"separation_obj_{idx}", separation_obj)
        separation_model.second_stage_ineq_con_to_obj_map[ss_ineq_con] = separation_obj
        separation_obj.deactivate()

    return separation_model


def get_sep_objective_values(separation_data, ss_ineq_cons):
    """
    Evaluate second-stage inequality constraint functions at current
    separation solution.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    ss_ineq_cons : list of Constraint
        Second-stage inequality constraints to be evaluated.

    Returns
    -------
    violations : ComponentMap
        Mapping from second-stage inequality constraints
        to violation values.
    """
    config = separation_data.config
    con_to_obj_map = separation_data.separation_model.second_stage_ineq_con_to_obj_map
    violations = ComponentMap()

    for ss_ineq_con in ss_ineq_cons:
        obj = con_to_obj_map[ss_ineq_con]
        try:
            violations[ss_ineq_con] = value(obj.expr)
        except (ValueError, ArithmeticError):
            vars_in_expr_str = ",\n  ".join(
                f"{var.name}={var.value}" for var in identify_variables(obj.expr)
            )
            config.progress_logger.error(
                "PyROS encountered an exception evaluating "
                "expression of second-stage inequality constraint with name "
                f"{ss_ineq_con.name!r} (separation objective {obj.name!r}) "
                f"at variable values:\n  {vars_in_expr_str}\n"
                "Does the expression contain log(x) or 1/x functions "
                "or others with tricky domains?"
            )
            raise

    return violations


def get_argmax_sum_violations(solver_call_results_map, ss_ineq_cons_to_evaluate):
    """
    Get key of entry of `solver_call_results_map` which contains
    separation problem solution with maximal sum of second-stage
    inequality constraint violations over a specified sequence of
    second-stage inequality constraints.

    Parameters
    ----------
    solver_call_results : ComponentMap
        Mapping from second-stage inequality constraints to corresponding
        separation solver call results.
    ss_ineq_cons_to_evaluate : list of Constraints
        Second-stage inequality constraints to consider for evaluating
        maximal sum.

    Returns
    -------
    worst_ss_ineq_con : None or Constraint
        Second-stage inequality constraint corresponding to solver call
        results object containing solution with maximal sum
        of violations across all second-stage inequality constraints.
        If ``found_violation`` attribute of all value entries of
        `solver_call_results_map` is False, then `None` is
        returned, as this means
        none of the second-stage inequality constraints
        were found to be violated.
    """
    # get indices of second-stage ineq constraints
    # for which violation found
    idx_to_ss_ineq_con_map = {
        idx: ss_ineq_con for idx, ss_ineq_con in enumerate(solver_call_results_map)
    }
    idxs_of_violated_cons = [
        idx
        for idx, ss_ineq_con in idx_to_ss_ineq_con_map.items()
        if solver_call_results_map[ss_ineq_con].found_violation
    ]

    num_violated_cons = len(idxs_of_violated_cons)

    if num_violated_cons == 0:
        return None

    # assemble square matrix (2D array) of constraint violations.
    # matrix size: number of constraints for which violation was found
    # each row corresponds to a second-stage inequality constraint
    # each column corresponds to a separation problem solution
    violations_arr = np.zeros(shape=(num_violated_cons, num_violated_cons))
    idxs_product = product(
        enumerate(idxs_of_violated_cons), enumerate(idxs_of_violated_cons)
    )
    for (row_idx, viol_con_idx), (col_idx, viol_param_idx) in idxs_product:
        violations_arr[row_idx, col_idx] = max(
            0,
            (
                # violation of this row's second-stage inequality con
                # by this column's separation solution
                # if separation problems were solved globally,
                # then diagonal entries should be the largest in each row
                solver_call_results_map[
                    idx_to_ss_ineq_con_map[viol_param_idx]
                ].scaled_violations[idx_to_ss_ineq_con_map[viol_con_idx]]
            ),
        )

    worst_col_idx = np.argmax(np.sum(violations_arr, axis=0))

    return idx_to_ss_ineq_con_map[idxs_of_violated_cons[worst_col_idx]]


def solve_separation_problem(separation_data, master_data):
    """
    Solve PyROS separation problems.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    pyros.solve_data.SeparationResults
        Separation problem solve results.
    """
    config = separation_data.config
    run_local = not config.bypass_local_separation
    run_global = config.bypass_local_separation

    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    )

    if run_local:
        local_separation_loop_results = perform_separation_loop(
            separation_data=separation_data,
            master_data=master_data,
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
            solve_globally=True,
        )
    else:
        global_separation_loop_results = None

    return SeparationResults(
        local_separation_loop_results=local_separation_loop_results,
        global_separation_loop_results=global_separation_loop_results,
    )


def evaluate_violations_by_nominal_master(separation_data, master_data, ss_ineq_cons):
    """
    Evaluate violation of second-stage inequality constraints by
    variables in nominal block of most recent master
    problem.

    Returns
    -------
    nom_ss_ineq_con_violations : dict
        Mapping from second-stage inequality constraint names
        to floats equal to violations by nominal master
        problem variables.
    """
    nom_ss_ineq_con_violations = ComponentMap()
    for ss_ineq_con in ss_ineq_cons:
        nom_violation = value(
            master_data.master_model.scenarios[0, 0].find_component(ss_ineq_con)
        )
        nom_ss_ineq_con_violations[ss_ineq_con] = nom_violation

    return nom_ss_ineq_con_violations


def group_ss_ineq_constraints_by_priority(separation_data):
    """
    Group model second-stage inequality constraints
    by separation priority.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.

    Returns
    -------
    dict
        Mapping from an int to a list of second-stage
        inequality constraints
        (Constraint objects),
        for which the int is equal to the specified priority.
        Keys are sorted in descending order
        (i.e. highest priority first).
    """
    separation_data.config.progress_logger.debug(
        "Grouping second-stage inequality constraints by separation priority..."
    )

    ss_ineq_cons = separation_data.separation_model.second_stage.inequality_cons
    separation_priority_groups = dict()
    for name, ss_ineq_con in ss_ineq_cons.items():
        priority = separation_data.separation_priority_order[name]
        cons_with_same_priority = separation_priority_groups.setdefault(priority, [])
        cons_with_same_priority.append(ss_ineq_con)

    # sort separation priority groups
    numeric_priority_grp_items = [
        (priority, cons) for priority, cons in separation_priority_groups.items()
    ]
    sorted_priority_groups = {
        priority: ss_ineq_cons
        for priority, ss_ineq_cons in sorted(numeric_priority_grp_items, reverse=True)
    }

    num_priority_groups = len(sorted_priority_groups)
    separation_data.config.progress_logger.debug(
        f"Found {num_priority_groups} separation "
        f"priority group{'s' if num_priority_groups != 1 else ''}."
    )
    separation_data.config.progress_logger.debug(
        "Separation priority grouping statistics:"
    )
    separation_data.config.progress_logger.debug(
        f"  {'Priority':20s}{'# Ineq Cons':15s}"
    )
    for priority, cons in sorted_priority_groups.items():
        separation_data.config.progress_logger.debug(
            f"  {priority:<20d}{len(cons):<15d}"
        )

    return sorted_priority_groups


def get_worst_discrete_separation_solution(
    ss_ineq_con, config, ss_ineq_cons_to_evaluate, discrete_solve_results
):
    """
    Determine separation solution (and therefore worst-case
    uncertain parameter realization) with maximum violation
    of specified second-stage inequality constraint.

    Parameters
    ----------
    ss_ineq_con : Constraint
        Second-stage inequality constraint of interest.
    config : ConfigDict
        User-specified PyROS solver settings.
    ss_ineq_cons_to_evaluate : list of Constraint
        Second-stage inequality constraints for which to report
        violations by separation solution.
    discrete_solve_results : DiscreteSeparationSolveCallResults
        Separation problem solutions corresponding to the
        uncertain parameter scenarios listed in
        ``config.uncertainty_set.scenarios``.

    Returns
    -------
    SeparationSolveCallResult
        Solver call result for second-stage inequality constraint of interest.
    """
    # violation of specified second-stage inequality
    # constraint by separation
    # problem solutions for all scenarios
    # scenarios with subsolver errors are replaced with nan
    violations_of_ss_ineq_con = [
        (
            solve_call_res.scaled_violations[ss_ineq_con]
            if not solve_call_res.subsolver_error
            else np.nan
        )
        for solve_call_res in discrete_solve_results.solver_call_results.values()
    ]

    list_of_scenario_idxs = list(discrete_solve_results.solver_call_results.keys())

    # determine separation solution for which scaled violation of this
    # second-stage inequality constraint is the worst
    worst_case_res = discrete_solve_results.solver_call_results[
        list_of_scenario_idxs[np.nanargmax(violations_of_ss_ineq_con)]
    ]
    worst_case_violation = np.nanmax(violations_of_ss_ineq_con)
    assert worst_case_violation in worst_case_res.scaled_violations.values()

    # evaluate violations for specified second-stage inequality constraints
    eval_ss_ineq_con_scaled_violations = ComponentMap(
        (ss_ineq_con, worst_case_res.scaled_violations[ss_ineq_con])
        for ss_ineq_con in ss_ineq_cons_to_evaluate
    )

    # discrete separation solutions were obtained by optimizing
    # just one second-stage inequality constraint, as an efficiency.
    # if the constraint passed to this routine is the same as the
    # constraint used to obtain the solutions, then we bundle
    # the separation solve call results into a single list.
    # otherwise, we return an empty list, as we did not need to call
    # subsolvers for the other second-stage inequality constraints
    is_optimized_ss_ineq_con = (
        ss_ineq_con is discrete_solve_results.second_stage_ineq_con
    )
    if is_optimized_ss_ineq_con:
        results_list = [
            res
            for solve_call_results in discrete_solve_results.solver_call_results.values()
            for res in solve_call_results.results_list
        ]
    else:
        results_list = []

    # check if there were any failed scenarios for subsolver_error
    # if there are failed scenarios, subsolver error triggers for all ineq
    if any(np.isnan(violations_of_ss_ineq_con)):
        subsolver_error_flag = True
    else:
        subsolver_error_flag = False

    return SeparationSolveCallResults(
        solved_globally=worst_case_res.solved_globally,
        results_list=results_list,
        scaled_violations=eval_ss_ineq_con_scaled_violations,
        violating_param_realization=worst_case_res.violating_param_realization,
        variable_values=worst_case_res.variable_values,
        found_violation=(worst_case_violation > config.robust_feasibility_tolerance),
        time_out=False,
        subsolver_error=subsolver_error_flag,
        discrete_set_scenario_index=worst_case_res.discrete_set_scenario_index,
    )


def get_con_name_repr(separation_model, con, with_obj_name=True):
    """
    Get string representation of second-stage inequality constraint
    and the objective to which it has been mapped.

    Parameters
    ----------
    separation_model : ConcreteModel
        Separation model.
    con : ScalarConstraint or ConstraintData
        Constraint for which to get the representation.
    with_obj_name : bool, optional
        Include name of separation model objective to which
        constraint is mapped. Applicable only to second-stage inequality
        constraints of the separation problem.

    Returns
    -------
    str
        Constraint name representation.
    """
    qual_str = ""
    if with_obj_name:
        objectives_map = separation_model.second_stage_ineq_con_to_obj_map
        separation_obj = objectives_map[con]
        qual_str = f" (mapped to objective {separation_obj.name!r})"

    return f"{con.index()!r}{qual_str}"


def perform_separation_loop(separation_data, master_data, solve_globally):
    """
    Loop through, and solve, PyROS separation problems to
    desired optimality condition.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    master_data : MasterProblemData
        Master problem data.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve separation problems locally.

    Returns
    -------
    pyros.solve_data.SeparationLoopResults
        Separation problem solve results.
    """
    config = separation_data.config
    all_ss_ineq_constraints = list(
        separation_data.separation_model.second_stage.inequality_cons.values()
    )
    if not all_ss_ineq_constraints:
        # robustness certified: no separation problems to solve
        return SeparationLoopResults(
            solver_call_results=ComponentMap(),
            solved_globally=solve_globally,
            worst_case_ss_ineq_con=None,
        )

    # needed for normalizing separation solution constraint violations
    separation_data.nom_ss_ineq_con_violations = evaluate_violations_by_nominal_master(
        separation_data=separation_data,
        master_data=master_data,
        ss_ineq_cons=all_ss_ineq_constraints,
    )
    sorted_priority_groups = separation_data.separation_priority_groups
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
                worst_case_ss_ineq_con=None,
                all_discrete_scenarios_exhausted=True,
            )

        ss_ineq_con_to_maximize = sorted_priority_groups[
            max(sorted_priority_groups.keys())
        ][0]

        # efficiency: evaluate all separation problem solutions in
        # advance of entering loop
        discrete_sep_results = discrete_solve(
            separation_data=separation_data,
            master_data=master_data,
            solve_globally=solve_globally,
            ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
            ss_ineq_cons_to_evaluate=all_ss_ineq_constraints,
        )

        termination_not_ok = (
            discrete_sep_results.time_out or discrete_sep_results.subsolver_error
        )
        if termination_not_ok:
            single_solver_call_res = ComponentMap()
            results_list = [
                res
                for solve_call_results in discrete_sep_results.solver_call_results.values()
                for res in solve_call_results.results_list
            ]
            single_solver_call_res[ss_ineq_con_to_maximize] = (
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
                worst_case_ss_ineq_con=None,
            )

    all_solve_call_results = ComponentMap()
    priority_groups_enum = enumerate(sorted_priority_groups.items())
    solve_adverb = "Globally" if solve_globally else "Locally"
    for group_idx, (priority, ss_ineq_constraints) in priority_groups_enum:
        priority_group_solve_call_results = ComponentMap()

        for idx, ss_ineq_con in enumerate(ss_ineq_constraints):
            # log progress of separation loop
            config.progress_logger.debug(
                f"{solve_adverb} separating second-stage inequality constraint "
                f"{get_con_name_repr(separation_data.separation_model, ss_ineq_con)} "
                f"(priority {priority}, priority group {group_idx + 1} of "
                f"{len(sorted_priority_groups)}, "
                f"constraint {idx + 1} of {len(ss_ineq_constraints)} "
                "in priority group, "
                f"{len(all_solve_call_results) + idx + 1} of "
                f"{len(all_ss_ineq_constraints)} total)"
            )

            # solve separation problem for
            # this second-stage inequality constraint
            if uncertainty_set_is_discrete:
                solve_call_results = get_worst_discrete_separation_solution(
                    ss_ineq_con=ss_ineq_con,
                    config=config,
                    ss_ineq_cons_to_evaluate=all_ss_ineq_constraints,
                    discrete_solve_results=discrete_sep_results,
                )
            else:
                solve_call_results = solver_call_separation(
                    separation_data=separation_data,
                    master_data=master_data,
                    solve_globally=solve_globally,
                    ss_ineq_con_to_maximize=ss_ineq_con,
                    ss_ineq_cons_to_evaluate=all_ss_ineq_constraints,
                )

            priority_group_solve_call_results[ss_ineq_con] = solve_call_results

            termination_not_ok = solve_call_results.time_out
            if termination_not_ok:
                all_solve_call_results.update(priority_group_solve_call_results)
                return SeparationLoopResults(
                    solver_call_results=all_solve_call_results,
                    solved_globally=solve_globally,
                    worst_case_ss_ineq_con=None,
                )

            # provide message that PyROS will attempt to find a violation and move
            # to the next iteration even after subsolver error
            if solve_call_results.subsolver_error:
                config.progress_logger.warning(
                    "PyROS is attempting to recover and will continue to "
                    "the next iteration if a constraint violation is found."
                )

        all_solve_call_results.update(priority_group_solve_call_results)

        # there may be multiple separation problem solutions
        # found to have violated a second-stage inequality constraint.
        # we choose just one for master problem of next iteration
        worst_case_ss_ineq_con = get_argmax_sum_violations(
            solver_call_results_map=all_solve_call_results,
            ss_ineq_cons_to_evaluate=ss_ineq_constraints,
        )
        if worst_case_ss_ineq_con is not None:
            # take note of chosen separation solution
            worst_case_res = all_solve_call_results[worst_case_ss_ineq_con]
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
                f"{get_con_name_repr(separation_data.separation_model, worst_case_ss_ineq_con)} "
                "under realization "
                f"{worst_case_res.violating_param_realization}."
            )
            config.progress_logger.debug(
                f"Maximal scaled violation "
                f"{worst_case_res.scaled_violations[worst_case_ss_ineq_con]} "
                "from this constraint "
                "exceeds the robust feasibility tolerance "
                f"{config.robust_feasibility_tolerance}"
            )

            # violating separation problem solution now chosen.
            # exit loop
            break
        else:
            config.progress_logger.debug(
                "No violated second-stage inequality constraints found."
            )

    return SeparationLoopResults(
        solver_call_results=all_solve_call_results,
        solved_globally=solve_globally,
        worst_case_ss_ineq_con=worst_case_ss_ineq_con,
    )


def evaluate_ss_ineq_con_violations(
    separation_data, ss_ineq_con_to_maximize, ss_ineq_cons_to_evaluate
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
    ss_ineq_con_to_maximize : ConstraintData
        Second-stage inequality constraint
        to which the current solution is mapped.
    ss_ineq_cons_to_evaluate : list of Constraint
        Second-stage inequality constraints whose expressions are to
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
        Mapping from second-stage inequality constraints to be evaluated
        to their violations by the separation problem solution.
    constraint_violated : bool
        True if second-stage inequality constraint mapped to active
        separation model Objective is violated (beyond tolerance),
        False otherwise

    Raises
    ------
    ValueError
        If `ss_ineq_cons_to_evaluate` does not contain exactly
        1 entry which can be mapped to an active Objective
        of ``model_data.separation_model``.
    """
    config = separation_data.config

    # parameter realization for current separation problem solution
    uncertain_param_vars = (
        separation_data.separation_model.uncertainty.uncertain_param_var_list
    )
    violating_param_realization = list(
        param_var.value for param_var in uncertain_param_vars
    )

    # evaluate violations for all second-stage inequality
    # constraints provided
    violations_by_sep_solution = get_sep_objective_values(
        separation_data=separation_data, ss_ineq_cons=ss_ineq_cons_to_evaluate
    )

    # normalize constraint violation: i.e. divide by
    # absolute value of constraint expression evaluated at
    # nominal master solution (if expression value is large enough)
    scaled_violations = ComponentMap()
    for ss_ineq_con, sep_sol_violation in violations_by_sep_solution.items():
        scaled_violation = sep_sol_violation / max(
            1, abs(separation_data.nom_ss_ineq_con_violations[ss_ineq_con])
        )
        scaled_violations[ss_ineq_con] = scaled_violation
        if ss_ineq_con is ss_ineq_con_to_maximize:
            scaled_active_obj_violation = scaled_violation

    constraint_violated = (
        scaled_active_obj_violation > config.robust_feasibility_tolerance
    )

    return (violating_param_realization, scaled_violations, constraint_violated)


def initialize_separation(ss_ineq_con_to_maximize, separation_data, master_data):
    """
    Initialize separation problem variables using the solution
    to the most recent master problem.

    Parameters
    ----------
    ss_ineq_con_to_maximize : ConstraintData
        Second-stage inequality constraint
        whose violation is to be maximized
        for the separation problem of interest.
    separation_data : SeparationProblemData
        Separation problem data.
    master_data : MasterProblemData
        Master problem data.

    Note
    ----
    The point to which the separation model is initialized should,
    in general, be feasible, provided the set does not have a
    discrete geometry (as there is no master model block corresponding
    to any of the remaining discrete scenarios against which we
    separate).
    """
    config = separation_data.config
    master_model = master_data.master_model
    sep_model = separation_data.separation_model

    def eval_master_violation(scenario_idx):
        """
        Evaluate violation of `ss_ineq_con` by variables of
        specified master block.
        """
        master_con = master_model.scenarios[scenario_idx].find_component(
            ss_ineq_con_to_maximize
        )
        return value(master_con)

    # initialize from master block with max violation of the
    # second-stage ineq constraint of interest. Gives the best known
    # feasible solution (for case of non-discrete uncertainty sets).
    worst_master_block_idx = max(
        master_model.scenarios.keys(), key=eval_master_violation
    )
    worst_case_master_blk = master_model.scenarios[worst_master_block_idx]
    for sep_var in sep_model.all_variables:
        master_var = worst_case_master_blk.find_component(sep_var)
        sep_var.set_value(value(master_var, exception=False))

    # for discrete uncertainty sets, the uncertain parameters
    # have already been addressed
    if config.uncertainty_set.geometry != Geometry.DISCRETE_SCENARIOS:
        param_vars = sep_model.uncertainty.uncertain_param_var_list
        param_values = separation_data.points_added_to_master[worst_master_block_idx]
        for param_var, val in zip(param_vars, param_values):
            param_var.set_value(val)

        aux_param_vars = sep_model.uncertainty.auxiliary_var_list
        aux_param_values = separation_data.auxiliary_values_for_master_points[
            worst_master_block_idx
        ]
        for aux_param_var, aux_val in zip(aux_param_vars, aux_param_values):
            aux_param_var.set_value(aux_val)

    # confirm the initial point is feasible for cases where
    # we expect it to be (i.e. non-discrete uncertainty sets).
    # otherwise, log the violated constraints
    tol = ABS_CON_CHECK_FEAS_TOL
    ss_ineq_con_name_repr = get_con_name_repr(
        separation_model=sep_model, con=ss_ineq_con_to_maximize, with_obj_name=True
    )
    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry is Geometry.DISCRETE_SCENARIOS
    )
    for con in sep_model.component_data_objects(Constraint, active=True):
        lslack, uslack = con.lslack(), con.uslack()
        if (lslack < -tol or uslack < -tol) and not uncertainty_set_is_discrete:
            config.progress_logger.debug(
                f"Initial point for separation of second-stage ineq constraint "
                f"{ss_ineq_con_name_repr} violates the model constraint "
                f"{con.name!r} by more than {tol} ({lslack=}, {uslack=})"
            )

    for con in sep_model.uncertainty.certain_param_var_cons:
        trivially_infeasible = (
            con.lslack() < -ABS_CON_CHECK_FEAS_TOL
            or con.uslack() < -ABS_CON_CHECK_FEAS_TOL
        )
        if trivially_infeasible:
            # this should never happen in the context of a full solve,
            # since the certain parameters should be at their
            # nominal values, and the nominal point was already
            # confirmed to be a member of the set
            config.progress_logger.error(
                f"Uncertainty set "
                f"(type {type(config.uncertainty_set).__name__}) "
                f"constraint {con.name!r}, with expression {con.expr}, "
                "is trivially infeasible at the parameter realization "
                f"{config.nominal_uncertain_param_vals}. "
                f"Check implementation of "
                f"{config.uncertainty_set.set_as_constraint.__name__}."
            )
            raise ValueError(
                f"Trivial infeasibility detected in the uncertainty set "
                f"(type {type(config.uncertainty_set).__name__}) constraints."
            )


locally_acceptable = {tc.optimal, tc.locallyOptimal, tc.globallyOptimal}
globally_acceptable = {tc.optimal, tc.globallyOptimal}


def solver_call_separation(
    separation_data,
    master_data,
    solve_globally,
    ss_ineq_con_to_maximize,
    ss_ineq_cons_to_evaluate,
):
    """
    Invoke subordinate solver(s) on separation problem.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    master_data : MasterProblemData
        Master problem data.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve locally.
    ss_ineq_con_to_maximize : Constraint
        Second-stage inequality constraint
        for which to solve separation problem.
        Informs the objective (constraint violation) to maximize.
    ss_ineq_cons_to_evaluate : list of Constraint
        Second-stage inequality constraints whose expressions are to be
        evaluated at the separation problem solution
        obtained.

    Returns
    -------
    solve_call_results : pyros.solve_data.SeparationSolveCallResults
        Solve results for separation problem of interest.
    """
    config = separation_data.config
    # prepare the problem
    separation_model = separation_data.separation_model
    objectives_map = separation_data.separation_model.second_stage_ineq_con_to_obj_map
    separation_obj = objectives_map[ss_ineq_con_to_maximize]
    initialize_separation(ss_ineq_con_to_maximize, separation_data, master_data)
    separation_obj.activate()

    # get name (index) of constraint for loggers
    con_name_repr = get_con_name_repr(
        separation_model=separation_model,
        con=ss_ineq_con_to_maximize,
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
                f"separation of second-stage inequality constraint {con_name_repr} "
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
            ) = evaluate_ss_ineq_con_violations(
                separation_data=separation_data,
                ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
                ss_ineq_cons_to_evaluate=ss_ineq_cons_to_evaluate,
            )
            solve_call_results.auxiliary_param_values = [
                auxvar.value
                for auxvar in separation_model.uncertainty.auxiliary_var_list
            ]

            separation_obj.deactivate()

            return solve_call_results
        else:
            config.progress_logger.debug(
                f"Solver {opt} ({idx + 1} of {len(solvers)}) "
                f"failed for {solve_mode} separation of second-stage inequality "
                f"constraint {con_name_repr} in iteration "
                f"{separation_data.iteration}. Termination condition: "
                f"{results.solver.termination_condition!r}."
            )
            config.progress_logger.debug(f"Results:\n{results.solver}")

    # All subordinate solvers failed to optimize model to appropriate
    # termination condition. PyROS will terminate with subsolver
    # error. At this point, export model if desired
    solve_call_results.subsolver_error = True
    solve_call_results.message = (
        "Could not successfully solve separation problem of iteration "
        f"{separation_data.iteration} "
        f"for second-stage inequality constraint {con_name_repr} with any of the "
        f"provided subordinate {solve_mode} optimizers. "
        f"(Termination statuses: "
        f"{[str(term_cond) for term_cond in solver_status_dict.values()]}.)"
    )
    config.progress_logger.warning(solve_call_results.message)

    if config.keepfiles and config.subproblem_file_directory is not None:
        write_subproblem(
            model=separation_model,
            fname=(
                f"{config.uncertainty_set.type}_{separation_model.name}"
                f"_separation_{separation_data.iteration}"
                f"_obj_{separation_obj.name}"
            ),
            config=config,
        )

    separation_obj.deactivate()

    return solve_call_results


def discrete_solve(
    separation_data,
    master_data,
    solve_globally,
    ss_ineq_con_to_maximize,
    ss_ineq_cons_to_evaluate,
):
    """
    Obtain separation problem solution for each scenario
    of the uncertainty set not already added to the most
    recent master problem.

    Parameters
    ----------
    separation_data : SeparationProblemData
        Separation problem data.
    master_data : MasterProblemData
        Master problem data.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the model.
    solve_globally : bool
        Is separation problem to be solved globally.
    ss_ineq_con_to_maximize : Constraint
        Second-stage inequality constraint for which to solve separation
        problem.
    ss_ineq_cons_to_evaluate : list of Constraint
        Secnod-stage inequality constraints whose expressions are to be
        evaluated at the each of separation problem solutions
        obtained.

    Returns
    -------
    discrete_separation_results : DiscreteSeparationSolveCallResults
        Separation solver call results on second-stage inequality constraint
        of interest for every scenario considered.

    Notes
    -----
    Since we assume that models passed to PyROS are such that the DOF
    variables and uncertain parameter values uniquely define the state
    variables, this method need be only be invoked once per separation
    loop. Subject to our assumption, the choice of objective
    (``ss_ineq_con_to_maximize``) should not affect the solutions returned
    beyond subsolver tolerances.
    For other second-stage inequality constraints, the
    optimal separation problem solution can then be evaluated by simple
    enumeration of the solutions returned by this function, since for
    discrete uncertainty sets, the number of feasible separation
    solutions is, under our assumption, merely equal to the number
    of scenarios in the uncertainty set.
    """
    config = separation_data.config

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
    for idx, scenario_idx in enumerate(scenario_idxs_to_separate):
        # fix uncertain parameters to scenario value
        # hence, no need to activate uncertainty set constraints
        scenario = config.uncertainty_set.scenarios[scenario_idx]
        for param, coord_val in zip(uncertain_param_vars, scenario):
            param.fix(coord_val)

        # debug statement for solving square problem for each scenario
        config.progress_logger.debug(
            f"Attempting to solve square problem for discrete scenario {scenario}"
            f", {idx + 1} of {len(scenario_idxs_to_separate)} total"
        )

        # obtain separation problem solution
        solve_call_results = solver_call_separation(
            separation_data=separation_data,
            master_data=master_data,
            solve_globally=solve_globally,
            ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
            ss_ineq_cons_to_evaluate=ss_ineq_cons_to_evaluate,
        )
        solve_call_results.discrete_set_scenario_index = scenario_idx
        solve_call_results_dict[scenario_idx] = solve_call_results

        # halt at first encounter of unacceptable termination
        termination_not_ok = solve_call_results.time_out
        if termination_not_ok:
            break

        # report any subsolver errors, but continue
        if solve_call_results.subsolver_error:
            config.progress_logger.warning(
                f"All solvers failed to solve discrete scenario {scenario_idx}: "
                f"{scenario}"
            )

    return DiscreteSeparationSolveCallResults(
        solved_globally=solve_globally,
        solver_call_results=solve_call_results_dict,
        second_stage_ineq_con=ss_ineq_con_to_maximize,
    )


class SeparationProblemData:
    """
    Container for objects related to the PyROS separation problem.

    Parameters
    ----------
    model_data : ModelData
        PyROS model data object, equipped with the
        fully preprocessed working model.

    Attributes
    ----------
    separation_model : BlockData
        Separation problem model object.
    timing : TimingData
        Main timer for the current problem being solved.
    config : ConfigDict
        PyROS solver options.
    separation_priority_order : dict
        Standardized/preprocessed mapping from names of the
        second-stage inequality constraint objects to integers
        specifying their priorities.
    iteration : int
        Index of the current PyROS cutting set iteration.
    points_added_to_master : dict
        Maps each scenario index (2-tuple of ints) of the
        master problem model object to the corresponding
        uncertain parameter realization.
    auxiliary_values_for_master_points : dict
        Maps each scenario index (2-tuple of ints) of the
        master problem model object to the auxiliary parameter
        values corresponding to the associated uncertain parameter
        realization.
    idxs_of_master_scenarios : None or list of int
        If ``config.uncertainty_set`` is of type
        :class:`~pyomo.contrib.pyros.uncertainty_sets.DiscreteScenarioSet`,
        then this attribute is a list
        of ints, each entry of which is a list index for
        an entry in the ``scenarios`` attribute of the
        uncertainty set. Otherwise, this attribute is set to None.
    """

    def __init__(self, model_data):
        """Initialize self (see class docstring)."""
        self.separation_model = construct_separation_problem(model_data)
        self.timing = model_data.timing
        self.separation_priority_order = model_data.separation_priority_order.copy()
        self.iteration = 0

        config = model_data.config
        self.config = config
        self.points_added_to_master = {(0, 0): config.nominal_uncertain_param_vals}
        self.auxiliary_values_for_master_points = {
            (0, 0): [
                # auxiliary variable values for nominal point have already
                # been computed and loaded into separation model
                aux_var.value
                for aux_var in self.separation_model.uncertainty.auxiliary_var_list
            ]
        }

        if config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS:
            self.idxs_of_master_scenarios = [
                config.uncertainty_set.scenarios.index(
                    tuple(config.nominal_uncertain_param_vals)
                )
            ]
        else:
            self.idxs_of_master_scenarios = None

        self.separation_priority_groups = group_ss_ineq_constraints_by_priority(self)

    def solve_separation(self, master_data):
        """
        Solve the separation problem.
        """
        return solve_separation_problem(self, master_data)
