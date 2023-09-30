"""
Functions for the construction and solving of the GRCS separation problem via ROsolver
"""
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
    DiscreteSeparationSolveCallResults,
    SeparationSolveCallResults,
    SeparationLoopResults,
    SeparationResults,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
    replace_expressions,
    identify_mutable_parameters,
    identify_variables,
)
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
    TIC_TOC_SOLVE_TIME_ATTR,
    adjust_solver_time_settings,
    revert_solver_max_time_adjustment,
)
import os
from copy import deepcopy
from itertools import product


def add_uncertainty_set_constraints(model, config):
    """
    Add inequality constraint(s) representing the uncertainty set.
    """

    model.util.uncertainty_set_constraint = config.uncertainty_set.set_as_constraint(
        uncertain_params=model.util.uncertain_param_vars, model=model, config=config
    )

    config.uncertainty_set.add_bounds_on_uncertain_parameters(
        model=model, config=config
    )

    # === Pre-process out any uncertain parameters which have q_LB = q_ub via (q_ub - q_lb)/max(1,|q_UB|) <= TOL
    #     before building the uncertainty set constraint(s)
    uncertain_params = config.uncertain_params
    for i in range(len(uncertain_params)):
        if is_certain_parameter(uncertain_param_index=i, config=config):
            # This parameter is effectively certain for this set, can remove it from the uncertainty set
            # We do this by fixing it in separation to its nominal value
            model.util.uncertain_param_vars[i].fix(
                config.nominal_uncertain_param_vals[i]
            )

    return


def make_separation_objective_functions(model, config):
    """
    Inequality constraints referencing control variables, state variables, or uncertain parameters
    must be separated against in separation problem.
    """
    performance_constraints = []
    for c in model.component_data_objects(Constraint, active=True, descend_into=True):
        _vars = ComponentSet(identify_variables(expr=c.expr))
        uncertain_params_in_expr = list(
            v for v in model.util.uncertain_param_vars.values() if v in _vars
        )
        state_vars_in_expr = list(v for v in model.util.state_vars if v in _vars)
        second_stage_variables_in_expr = list(
            v for v in model.util.second_stage_variables if v in _vars
        )
        if not c.equality and (
            uncertain_params_in_expr
            or state_vars_in_expr
            or second_stage_variables_in_expr
        ):
            # This inequality constraint depends on uncertain parameters therefore it must be separated against
            performance_constraints.append(c)
        elif not c.equality and not (
            uncertain_params_in_expr
            or state_vars_in_expr
            or second_stage_variables_in_expr
        ):
            c.deactivate()  # These are x \in X constraints, not active in separation because x is fixed to x* from previous master
    model.util.performance_constraints = performance_constraints
    model.util.separation_objectives = []
    map_obj_to_constr = ComponentMap()

    for idx, c in enumerate(performance_constraints):
        # Separation objective constraints standardized to be MAXIMIZATION of <= constraints
        c.deactivate()
        if c.upper is not None:
            # This is an <= constraint, maximized in separation
            obj = Objective(expr=c.body - c.upper, sense=maximize)
            map_obj_to_constr[c] = obj
            model.add_component("separation_obj_" + str(idx), obj)
            model.util.separation_objectives.append(obj)
        elif c.lower is not None:
            # This is an >= constraint, not supported
            raise ValueError(
                "All inequality constraints in model must be in standard form (<= RHS)"
            )

    model.util.map_obj_to_constr = map_obj_to_constr
    for obj in model.util.separation_objectives:
        obj.deactivate()

    return


def make_separation_problem(model_data, config):
    """
    Swap out uncertain param Param objects for Vars
    Add uncertainty set constraints and separation objectives
    """
    separation_model = model_data.original.clone()
    separation_model.del_component("coefficient_matching_constraints")
    separation_model.del_component("coefficient_matching_constraints_index")

    uncertain_params = separation_model.util.uncertain_params
    separation_model.util.uncertain_param_vars = param_vars = Var(
        range(len(uncertain_params))
    )
    map_new_constraint_list_to_original_con = ComponentMap()

    if config.objective_focus is ObjectiveType.worst_case:
        separation_model.util.zeta = Param(initialize=0, mutable=True)
        constr = Constraint(
            expr=separation_model.first_stage_objective
            + separation_model.second_stage_objective
            - separation_model.util.zeta
            <= 0
        )
        separation_model.add_component("epigraph_constr", constr)

    substitution_map = {}
    # Separation problem initialized to nominal uncertain parameter values
    for idx, var in enumerate(list(param_vars.values())):
        param = uncertain_params[idx]
        var.set_value(param.value, skip_validation=True)
        substitution_map[id(param)] = var

    separation_model.util.new_constraints = constraints = ConstraintList()

    uncertain_param_set = ComponentSet(uncertain_params)
    for c in separation_model.component_data_objects(Constraint):
        if any(v in uncertain_param_set for v in identify_mutable_parameters(c.expr)):
            if c.equality:
                if c in separation_model.util.h_x_q_constraints:
                    # ensure that constraints subject to
                    # coefficient matching are not involved in
                    # separation problem.
                    # keeping them may induce numerical sensitivity
                    # issues, possibly leading to incorrect result
                    c.deactivate()
                else:
                    constraints.add(
                        replace_expressions(
                            expr=c.lower, substitution_map=substitution_map
                        )
                        == replace_expressions(
                            expr=c.body, substitution_map=substitution_map
                        )
                    )
            elif c.lower is not None:
                constraints.add(
                    replace_expressions(expr=c.lower, substitution_map=substitution_map)
                    <= replace_expressions(
                        expr=c.body, substitution_map=substitution_map
                    )
                )
            elif c.upper is not None:
                constraints.add(
                    replace_expressions(expr=c.upper, substitution_map=substitution_map)
                    >= replace_expressions(
                        expr=c.body, substitution_map=substitution_map
                    )
                )
            else:
                raise ValueError(
                    "Unable to parse constraint for building the separation problem."
                )
            c.deactivate()
            map_new_constraint_list_to_original_con[
                constraints[constraints.index_set().last()]
            ] = c

    separation_model.util.map_new_constraint_list_to_original_con = (
        map_new_constraint_list_to_original_con
    )

    # === Add objectives first so that the uncertainty set
    #     Constraints do not get picked up into the set
    # 	  of performance constraints which become objectives
    make_separation_objective_functions(separation_model, config)
    add_uncertainty_set_constraints(separation_model, config)

    # === Deactivate h(x,q) == 0 constraints
    for c in separation_model.util.h_x_q_constraints:
        c.deactivate()

    return separation_model


def get_sep_objective_values(model_data, config, perf_cons):
    """
    Evaluate performance constraint functions at current
    separation solution.

    Parameters
    ----------
    model_data : SeparationProblemData
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
    con_to_obj_map = model_data.separation_model.util.map_obj_to_constr
    violations = ComponentMap()

    for perf_con in perf_cons:
        obj = con_to_obj_map[perf_con]
        try:
            violations[perf_con] = value(obj.expr)
        except ValueError:
            for v in model_data.separation_model.util.first_stage_variables:
                config.progress_logger.info(v.name + " " + str(v.value))
            for v in model_data.separation_model.util.second_stage_variables:
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


def solve_separation_problem(model_data, config):
    """
    Solve PyROS separation problems.

    Parameters
    ----------
    model_data : SeparationProblemData
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
            model_data=model_data, config=config, solve_globally=False
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
            model_data=model_data, config=config, solve_globally=True
        )
    else:
        global_separation_loop_results = None

    return SeparationResults(
        local_separation_loop_results=local_separation_loop_results,
        global_separation_loop_results=global_separation_loop_results,
    )


def evaluate_violations_by_nominal_master(model_data, performance_cons):
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
    constraint_map_to_master = (
        model_data.separation_model.util.map_new_constraint_list_to_original_con
    )

    # get deterministic model constraints (include epigraph)
    set_of_deterministic_constraints = (
        model_data.separation_model.util.deterministic_constraints
    )
    if hasattr(model_data.separation_model, "epigraph_constr"):
        set_of_deterministic_constraints.add(
            model_data.separation_model.epigraph_constr
        )
    nom_perf_con_violations = {}

    for perf_con in performance_cons:
        if perf_con in set_of_deterministic_constraints:
            nom_constraint = perf_con
        else:
            nom_constraint = constraint_map_to_master[perf_con]
        nom_violation = value(
            model_data.master_nominal_scenario.find_component(nom_constraint)
        )
        nom_perf_con_violations[perf_con] = nom_violation

    return nom_perf_con_violations


def group_performance_constraints_by_priority(model_data, config):
    """
    Group model performance constraints by separation priority.

    Parameters
    ----------
    model_data : SeparationProblemData
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
    separation_priority_groups = dict()
    config_sep_priority_dict = config.separation_priority_order
    for perf_con in model_data.separation_model.util.performance_constraints:
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
    model_data,
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
    model_data : SeparationProblemData
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
            for solve_call_results in discrete_solve_results.solver_call_results.values()
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


def get_con_name_repr(separation_model, con, with_orig_name=True, with_obj_name=True):
    """
    Get string representation of performance constraint
    and any other modeling components to which it has
    been mapped.

    Parameters
    ----------
    separation_model : ConcreteModel
        Separation model.
    con : ScalarConstraint or ConstraintData
        Constraint for which to get the representation.
    with_orig_name : bool, optional
        If constraint was added during construction of the
        separation problem (i.e. if the constraint is a member of
        in `separation_model.util.new_constraints`),
        include the name of the original constraint from which
        `perf_con` was created.
    with_obj_name : bool, optional
        Include name of separation model objective to which
        constraint is mapped. Applicable only to performance
        constraints of the separation problem.

    Returns
    -------
    str
        Constraint name representation.
    """

    qual_strs = []
    if with_orig_name:
        # check performance constraint was not added
        # at construction of separation problem
        orig_con = separation_model.util.map_new_constraint_list_to_original_con.get(
            con, con
        )
        if orig_con is not con:
            qual_strs.append(f"originally {orig_con.name!r}")
    if with_obj_name:
        objectives_map = separation_model.util.map_obj_to_constr
        separation_obj = objectives_map[con]
        qual_strs.append(f"mapped to objective {separation_obj.name!r}")

    final_qual_str = f" ({', '.join(qual_strs)})" if qual_strs else ""

    return f"{con.name!r}{final_qual_str}"


def perform_separation_loop(model_data, config, solve_globally):
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
        model_data.separation_model.util.performance_constraints
    )
    if not all_performance_constraints:
        # robustness certified: no separation problems to solve
        return SeparationLoopResults(
            solver_call_results=ComponentMap(),
            solved_globally=solve_globally,
            worst_case_perf_con=None,
        )

    # needed for normalizing separation solution constraint violations
    model_data.nom_perf_con_violations = evaluate_violations_by_nominal_master(
        model_data=model_data, performance_cons=all_performance_constraints
    )
    sorted_priority_groups = group_performance_constraints_by_priority(
        model_data, config
    )
    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    )

    if uncertainty_set_is_discrete:
        all_scenarios_exhausted = len(model_data.idxs_of_master_scenarios) == len(
            config.uncertainty_set.scenarios
        )
        if all_scenarios_exhausted:
            # robustness certified: entire uncertainty set already
            # accounted for in master
            return SeparationLoopResults(
                solver_call_results=ComponentMap(),
                solved_globally=solve_globally,
                worst_case_perf_con=None,
            )

        perf_con_to_maximize = sorted_priority_groups[
            max(sorted_priority_groups.keys())
        ][0]

        # efficiency: evaluate all separation problem solutions in
        # advance of entering loop
        discrete_sep_results = discrete_solve(
            model_data=model_data,
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
                for solve_call_results in discrete_sep_results.solver_call_results.values()
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
                f"{get_con_name_repr(model_data.separation_model, perf_con)} "
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
                    model_data=model_data,
                    config=config,
                    perf_cons_to_evaluate=all_performance_constraints,
                    discrete_solve_results=discrete_sep_results,
                )
            else:
                solve_call_results = solver_call_separation(
                    model_data=model_data,
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
                model_data.idxs_of_master_scenarios.append(
                    worst_case_res.discrete_set_scenario_index
                )

            # # auxiliary log messages
            violated_con_names = "\n ".join(
                get_con_name_repr(model_data.separation_model, con)
                for con, res in all_solve_call_results.items()
                if res.found_violation
            )
            config.progress_logger.debug(
                f"Violated constraints:\n {violated_con_names} "
            )
            config.progress_logger.debug(
                "Worst-case constraint: "
                f"{get_con_name_repr(model_data.separation_model, worst_case_perf_con)} "
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
    model_data, config, perf_con_to_maximize, perf_cons_to_evaluate
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
    model_data : SeparationProblemData
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
    violating_param_realization = list(
        param.value
        for param in model_data.separation_model.util.uncertain_param_vars.values()
    )

    # evaluate violations for all performance constraints provided
    violations_by_sep_solution = get_sep_objective_values(
        model_data=model_data, config=config, perf_cons=perf_cons_to_evaluate
    )

    # normalize constraint violation: i.e. divide by
    # absolute value of constraint expression evaluated at
    # nominal master solution (if expression value is large enough)
    scaled_violations = ComponentMap()
    for perf_con, sep_sol_violation in violations_by_sep_solution.items():
        scaled_violation = sep_sol_violation / max(
            1, abs(model_data.nom_perf_con_violations[perf_con])
        )
        scaled_violations[perf_con] = scaled_violation
        if perf_con is perf_con_to_maximize:
            scaled_active_obj_violation = scaled_violation

    constraint_violated = (
        scaled_active_obj_violation > config.robust_feasibility_tolerance
    )

    return (violating_param_realization, scaled_violations, constraint_violated)


def initialize_separation(perf_con_to_maximize, model_data, config):
    """
    Initialize separation problem variables, and fix all first-stage
    variables to their corresponding values from most recent
    master problem solution.

    Parameters
    ----------
    perf_con_to_maximize : ConstraintData
        Performance constraint whose violation is to be maximized
        for the separation problem of interest.
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.

    Note
    ----
    If a static DR policy is used, then all second-stage variables
    are fixed and the decision rule equations are deactivated.

    The point to which the separation model is initialized should,
    in general, be feasible, provided the set does not have a
    discrete geometry (as there is no master model block corresponding
    to any of the remaining discrete scenarios against which we
    separate).
    """
    # initialize to values from nominal block if nominal objective.
    # else, initialize to values from latest block added to master
    if config.objective_focus == ObjectiveType.nominal:
        block_num = 0
    else:
        block_num = model_data.iteration

    master_blk = model_data.master_model.scenarios[block_num, 0]
    master_blks = list(model_data.master_model.scenarios.values())
    fsv_set = ComponentSet(master_blk.util.first_stage_variables)
    sep_model = model_data.separation_model

    def get_parent_master_blk(var):
        """
        Determine the master model scenario block of which
        a given variable is a child component (or descendant).
        """
        parent = var.parent_block()
        while parent not in master_blks:
            parent = parent.parent_block()
        return parent

    for master_var in master_blk.component_data_objects(Var, active=True):
        # parent block of the variable need not be `master_blk`
        # (e.g. for first stage and decision rule variables, it
        # may be the nominal block)
        parent_master_blk = get_parent_master_blk(master_var)
        sep_var_name = master_var.getname(
            relative_to=parent_master_blk, fully_qualified=True
        )

        # initialize separation problem var to value from master block
        sep_var = sep_model.find_component(sep_var_name)
        sep_var.set_value(value(master_var, exception=False))

        # fix first-stage variables (including decision rule vars)
        if master_var in fsv_set:
            sep_var.fix()

    # initialize uncertain parameter variables to most recent
    # point added to master
    if config.uncertainty_set.geometry != Geometry.DISCRETE_SCENARIOS:
        param_vars = sep_model.util.uncertain_param_vars
        latest_param_values = model_data.points_added_to_master[block_num]
        for param_var, val in zip(param_vars.values(), latest_param_values):
            param_var.set_value(val)

    # if static approximation, fix second-stage variables
    # and deactivate the decision rule equations
    for c in model_data.separation_model.util.second_stage_variables:
        if config.decision_rule_order != 0:
            c.unfix()
        else:
            c.fix()
    if config.decision_rule_order == 0:
        for v in model_data.separation_model.util.decision_rule_eqns:
            v.deactivate()
        for v in model_data.separation_model.util.decision_rule_vars:
            v.fix()

    if any(c.active for c in model_data.separation_model.util.h_x_q_constraints):
        raise AttributeError(
            "All h(x,q) type constraints must be deactivated in separation."
        )

    # confirm the initial point is feasible for cases where
    # we expect it to be (i.e. non-discrete uncertainty sets).
    # otherwise, log the violated constraints
    tol = ABS_CON_CHECK_FEAS_TOL
    perf_con_name_repr = get_con_name_repr(
        separation_model=model_data.separation_model,
        con=perf_con_to_maximize,
        with_orig_name=True,
        with_obj_name=True,
    )
    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry is Geometry.DISCRETE_SCENARIOS
    )
    for con in sep_model.component_data_objects(Constraint, active=True):
        lslack, uslack = con.lslack(), con.uslack()
        if (lslack < -tol or uslack < -tol) and not uncertainty_set_is_discrete:
            con_name_repr = get_con_name_repr(
                separation_model=model_data.separation_model,
                con=con,
                with_orig_name=True,
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
    model_data, config, solve_globally, perf_con_to_maximize, perf_cons_to_evaluate
):
    """
    Invoke subordinate solver(s) on separation problem.

    Parameters
    ----------
    model_data : SeparationProblemData
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
    # objective corresponding to specified performance constraint
    objectives_map = model_data.separation_model.util.map_obj_to_constr
    separation_obj = objectives_map[perf_con_to_maximize]

    if solve_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers

    # keep track of solver statuses for output logging
    solver_status_dict = {}
    nlp_model = model_data.separation_model

    # get name of constraint for loggers
    con_name_repr = get_con_name_repr(
        separation_model=nlp_model,
        con=perf_con_to_maximize,
        with_orig_name=True,
        with_obj_name=True,
    )
    solve_mode = "global" if solve_globally else "local"

    # === Initialize separation problem; fix first-stage variables
    initialize_separation(perf_con_to_maximize, model_data, config)

    separation_obj.activate()

    solve_call_results = SeparationSolveCallResults(
        solved_globally=solve_globally,
        time_out=False,
        results_list=[],
        found_violation=False,
        subsolver_error=False,
    )
    timer = TicTocTimer()
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(
                f"Invoking backup solver {opt!r} "
                f"(solver {idx + 1} of {len(solvers)}) for {solve_mode} "
                f"separation of performance constraint {con_name_repr} "
                f"in iteration {model_data.iteration}."
            )
        orig_setting, custom_setting_present = adjust_solver_time_settings(
            model_data.timing, opt, config
        )
        model_data.timing.start_timer(f"main.{solve_mode}_separation")
        timer.tic(msg=None)
        try:
            results = opt.solve(
                nlp_model,
                tee=config.tee,
                load_solutions=False,
                symbolic_solver_labels=True,
            )
        except ApplicationError:
            # account for possible external subsolver errors
            # (such as segmentation faults, function evaluation
            # errors, etc.)
            adverb = "globally" if solve_globally else "locally"
            config.progress_logger.error(
                f"Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) "
                f"encountered exception attempting "
                f"to {adverb} solve separation problem for constraint "
                f"{con_name_repr} in iteration {model_data.iteration}."
            )
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
            model_data.timing.stop_timer(f"main.{solve_mode}_separation")
        finally:
            revert_solver_max_time_adjustment(
                opt, orig_setting, custom_setting_present, config
            )

        # record termination condition for this particular solver
        solver_status_dict[str(opt)] = results.solver.termination_condition
        solve_call_results.results_list.append(results)

        # has PyROS time limit been reached?
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
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
            nlp_model.solutions.load_from(results)

            # record second-stage and state variable values
            solve_call_results.variable_values = ComponentMap()
            for var in nlp_model.util.second_stage_variables:
                solve_call_results.variable_values[var] = value(var)
            for var in nlp_model.util.state_vars:
                solve_call_results.variable_values[var] = value(var)

            # record uncertain parameter realization
            #   and constraint violations
            (
                solve_call_results.violating_param_realization,
                solve_call_results.scaled_violations,
                solve_call_results.found_violation,
            ) = evaluate_performance_constraint_violations(
                model_data, config, perf_con_to_maximize, perf_cons_to_evaluate
            )

            separation_obj.deactivate()

            return solve_call_results
        else:
            config.progress_logger.debug(
                f"Solver {opt} ({idx + 1} of {len(solvers)}) "
                f"failed for {solve_mode} separation of performance "
                f"constraint {con_name_repr} in iteration "
                f"{model_data.iteration}. Termination condition: "
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
                + nlp_model.name
                + "_separation_"
                + str(model_data.iteration)
                + "_obj_"
                + objective
                + ".bar"
            ),
        )
        nlp_model.write(
            output_problem_path, io_options={'symbolic_solver_labels': True}
        )
        serialization_msg = (
            " For debugging, problem has been serialized to the file "
            f"{output_problem_path!r}."
        )
    solve_call_results.message = (
        "Could not successfully solve separation problem of iteration "
        f"{model_data.iteration} "
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
    model_data, config, solve_globally, perf_con_to_maximize, perf_cons_to_evaluate
):
    """
    Obtain separation problem solution for each scenario
    of the uncertainty set not already added to the most
    recent master problem.

    Parameters
    ----------
    model_data : SeparationProblemData
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

    # Ensure uncertainty set constraints deactivated
    model_data.separation_model.util.uncertainty_set_constraint.deactivate()
    uncertain_param_vars = list(
        model_data.separation_model.util.uncertain_param_vars.values()
    )

    # skip scenarios already added to most recent master problem
    master_scenario_idxs = model_data.idxs_of_master_scenarios
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
            model_data=model_data,
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
