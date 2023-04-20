"""
Functions for the construction and solving of the GRCS separation problem via ROsolver
"""
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver, output_logger
from pyomo.contrib.pyros.solve_data import (
    SeparationSolveCallResults,
    SeparationLoopResults,
    SeparationResults,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.current import (
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

    if len(model.util.performance_constraints) == 0:
        raise ValueError(
            "No performance constraints identified for the postulated robust optimization problem."
        )

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
                constraints.add(
                    replace_expressions(expr=c.lower, substitution_map=substitution_map)
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


def get_index_of_max_violation(model_data, config, solve_data_list):
    """
    Get row (constraint) and column (uncertain parameter realization)
    index of PyROS separation problem results list with the
    highest average relative constraint violation.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        Mapping of PyROS solver options.
    solve_data_list : (M, N) array-like of SeparationResult
        Separation problem results for each performance
        constraint (row) and uncertain parameter scenario
        (column).

    Returns
    -------
    worst_con_idx, int
        Row (performance constraint) index.
    worst_scenario_idx, int
        Column (uncertain parameter realization) index.

    Notes
    -----
    Unless the uncertainty set (``config.uncertainty_set``)
    is a discrete-type set, `solve_data_list`
    has only one column, corresponding to the parameter realization
    for which the maximum violation of the performance constraint
    was found.
    Otherwise, each column corresponds to one of the points
    in the discrete set.
    """

    # get indices of performance cons for which violation found,
    # , and corresponding scenarios for which violation is maximized
    idxs_of_violated_cons = []
    con_to_worst_param_idx_map = {}
    for perf_con_idx, row in enumerate(solve_data_list):
        results_with_violation_found = {
            scenario_idx: res
            for scenario_idx, res in enumerate(row)
            if res.found_violation
        }
        if results_with_violation_found:
            # get index of scenario corresponding to
            # highest violation of this performance constraint
            # use insertion order of uncert param scenarios
            # for tiebreaks
            _, worst_case_scenario_idx = max(
                (perf_con_res.list_of_scaled_violations[perf_con_idx],
                 scenario_idx,)
                for scenario_idx, perf_con_res
                in results_with_violation_found.items()
            )

            # note this performance constraint was violated
            idxs_of_violated_cons.append(perf_con_idx)

            # note scenario for which violation is maximized
            con_to_worst_param_idx_map[perf_con_idx] = worst_case_scenario_idx

    num_violated_cons = len(idxs_of_violated_cons)
    if num_violated_cons == 0:
        # no violating realizations
        return None, None

    # assemble square matrix (2D array) of constraint violations.
    # matrix size: number of constraints for which violation was found
    # each row corresponds to a performance constraint
    # each column corresponds to a violating realization
    violations_arr = np.zeros(shape=(num_violated_cons, num_violated_cons))
    idxs_product = product(
        enumerate(idxs_of_violated_cons),
        enumerate(idxs_of_violated_cons),
    )
    for (row_idx, viol_con_idx), (col_idx, viol_param_idx) in idxs_product:
        # get (index of) uncert param realization which
        # maximizes violation of perf constraint indexed by
        # column
        scenario_idx = con_to_worst_param_idx_map[viol_param_idx]

        violations_arr[row_idx, col_idx] = max(
            0,
            (
                # violation of this row's performance constraint
                # by this column's separation solution
                # if separation problems were solved globally,
                # then diagonal entries should be the largest in each row
                solve_data_list[viol_param_idx][scenario_idx]
                .list_of_scaled_violations[viol_con_idx]
            ),
        )

    # determine column index for scenario with max sum of violations
    # over the violated performance constraints
    worst_con_idx = idxs_of_violated_cons[
        np.argmax(np.sum(violations_arr, axis=0))
    ]

    # get corresponding scenario index
    worst_scenario_idx = con_to_worst_param_idx_map[worst_con_idx]

    return worst_con_idx, worst_scenario_idx


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
    pyros.solve_data.SeparationResult
        Separation problem solve results.
    """
    run_local = not config.bypass_local_separation
    run_global = config.bypass_local_separation

    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry
        == Geometry.DISCRETE_SCENARIOS
    )

    # initialize separation solve times
    local_solve_time = 0
    global_solve_time = 0

    if run_local:
        separation_solve_result = perform_separation_loop(
            model_data=model_data,
            config=config,
            solve_globally=False,
        )
        run_global = not (
            separation_solve_result.found_violation
            or uncertainty_set_is_discrete
            or separation_solve_result.subsolver_error
            or separation_solve_result.time_out
            or config.bypass_global_separation
        )
        local_solve_time = separation_solve_result.solve_time

    if run_global:
        separation_solve_result = perform_separation_loop(
            model_data=model_data,
            config=config,
            solve_globally=True,
        )
        global_solve_time = separation_solve_result.solve_time

    robustness_certified = (
        not separation_solve_result.found_violation
        and (
            run_global
            or config.bypass_global_separation
            or uncertainty_set_is_discrete
        )
        and not separation_solve_result.subsolver_error
        and not separation_solve_result.time_out
    )

    return SeparationResults(
        solve_data_list=separation_solve_result.solve_data_list,
        violating_param_realization=(
            separation_solve_result.violating_param_realization
        ),
        violations=separation_solve_result.violations,
        local_solve_time=local_solve_time,
        global_solve_time=global_solve_time,
        subsolver_error=separation_solve_result.subsolver_error,
        time_out=separation_solve_result.time_out,
        solved_globally=separation_solve_result.solved_globally,
        robustness_certified=robustness_certified,
    )


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

    # List of objective functions
    objectives_map = model_data.separation_model.util.map_obj_to_constr
    constraint_map_to_master = (
        model_data
        .separation_model.util.map_new_constraint_list_to_original_con
    )

    # TODO: move this to pre-processing
    # group performance constraints by priority
    separation_priority_groups = dict()
    config_sep_priority_dict = config.separation_priority_order
    for perf_con in model_data.separation_model.util.performance_constraints:
        # by default, priority set to 0
        priority = config_sep_priority_dict.get(perf_con.name, 0)
        cons_with_same_priority = separation_priority_groups.setdefault(
            priority,
            [],
        )
        cons_with_same_priority.append(perf_con)

    # sort separation priority groups
    sorted_priority_groups = {
        priority: perf_cons
        for priority, perf_cons in
        sorted(separation_priority_groups.items(), reverse=True)
    }

    # get deterministic model constraints (include epigraph)
    set_of_deterministic_constraints = (
        model_data.separation_model.util.deterministic_constraints
    )
    if hasattr(model_data.separation_model, "epigraph_constr"):
        set_of_deterministic_constraints.add(
            model_data.separation_model.epigraph_constr
        )

    uncertainty_set_is_discrete = (
        config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    )

    acceptable_terminations = (
        globally_acceptable if solve_globally else locally_acceptable
    )

    solve_data_list = []
    solve_time = 0
    for priority, perf_constraints in sorted_priority_groups.items():
        # evaluate performance constraint functions at master
        #   solution block (0, 0) (nominal block).
        #   Needed for normalizing separation objective values
        model_data.nom_perf_con_violations = {}
        for perf_con in perf_constraints:
            if perf_con in set_of_deterministic_constraints:
                nom_constraint = perf_con
            else:
                nom_constraint = constraint_map_to_master[perf_con]
            nom_violation = value(
                model_data.master_nominal_scenario.find_component(
                    nom_constraint
                )
            )
            model_data.nom_perf_con_violations[perf_con] = nom_violation

        # now solve separation problems
        for perf_con in perf_constraints:
            # config.progress_logger.info(
            #     f"Separating constraint {perf_con.name}"
            # )

            # retrieve and activate corresponding separation objective
            separation_obj = objectives_map[perf_con]
            separation_obj.activate()

            # solve separation problem
            if uncertainty_set_is_discrete:
                solve_call_results_list = discrete_solve(
                    model_data=model_data,
                    config=config,
                    solve_globally=solve_globally,
                    perf_cons_to_evaluate=perf_constraints,
                )
            else:
                solve_call_results_list = [
                    solver_call_separation(
                        model_data=model_data,
                        config=config,
                        solve_globally=solve_globally,
                        perf_cons_to_evaluate=perf_constraints,
                    )
                ]

            solve_data_list.append(solve_call_results_list)

            # increment solve time accumulator
            solve_time_increment = sum(
                get_time_from_solver(solve_res)
                for sep_res in solve_call_results_list
                for solve_res in sep_res.results_list
            )
            solve_time += solve_time_increment

            # Terminate due to timeout or nonacceptable solve status
            time_out = any(
                scall_res.time_out
                for scall_res in solve_call_results_list
            )
            subsolver_error = any(
                not scall_res.termination_acceptable(acceptable_terminations)
                for scall_res in solve_call_results_list
            )
            if time_out or subsolver_error:
                return SeparationLoopResults(
                    solve_data_list=solve_data_list,
                    violating_param_realization=None,
                    violations=None,
                    solve_time=solve_time,
                    subsolver_error=subsolver_error,
                    time_out=time_out,
                    solved_globally=solve_globally,
                )

            separation_obj.deactivate()

    # There may be multiple separation problem solutions
    # for which a violation was found.
    # Choose just one for updating next master
    perf_con_idx, scenario_idx = get_index_of_max_violation(
        model_data=model_data,
        config=config,
        solve_data_list=solve_data_list,
    )
    if (perf_con_idx, scenario_idx) != (None, None):
        violating_param_realization = list(
            solve_data_list[perf_con_idx][scenario_idx]
            .violating_param_realization
        )
        violations = (
            solve_data_list[perf_con_idx][scenario_idx]
            .list_of_scaled_violations
        )
        # violated_con_name = list(objectives_map.keys())[perf_con_idx]
        # config.progress_logger.info(
        #     f"Violation found for constraint {violated_con_name} "
        #     f"under realization {violating_param_realization}"
        # )
    else:  # no performance constraints were violated
        violating_param_realization = None
        violations = None

    return SeparationLoopResults(
        solve_data_list=solve_data_list,
        violating_param_realization=violating_param_realization,
        violations=violations,
        solve_time=solve_time,
        subsolver_error=False,
        time_out=False,
        solved_globally=solve_globally,
    )


def update_solve_data_violations(
        model_data,
        config,
        perf_cons_to_evaluate,
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
    scaled_violations : list of float
        List of values of performance constraint functions at
        separation problem solution.
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
    objectives_map = model_data.separation_model.util.map_obj_to_constr
    active_perf_cons_and_objs = [
        (perf_con, objectives_map[perf_con])
        for perf_con in perf_cons_to_evaluate
        if objectives_map[perf_con].active
    ]
    if len(active_perf_cons_and_objs) != 1:
        raise ValueError(
            f"Exactly one performance constraint in `perf_cons_to_evalaute`"
            "should be mapped to an active objective of the separation"
            f"model (found {len(active_perf_cons_and_objs)}) "
        )
    active_perf_con, active_obj = active_perf_cons_and_objs[0]

    # parameter realization for current separation problem solution
    violating_param_realization = list(
        param.value for param in
        model_data.separation_model.util.uncertain_param_vars.values()
    )

    # evaluate violations for all performance constraints provided
    violations_by_sep_solution = get_sep_objective_values(
        model_data=model_data,
        config=config,
        perf_cons=perf_cons_to_evaluate,
    )

    # normalize constraint violation: i.e. divide by
    # absolute value of constraint expression evaluated at
    # nominal master solution (if expression value is large enough)
    scaled_violations = []
    for perf_con, sep_sol_violation in violations_by_sep_solution.items():
        scaled_violation = (
            sep_sol_violation
            / max(1, abs(model_data.nom_perf_con_violations[perf_con]))
        )
        scaled_violations.append(scaled_violation)
        if perf_con is active_perf_con:
            scaled_active_obj_violation = scaled_violation

    constraint_violated = (
        scaled_active_obj_violation > config.robust_feasibility_tolerance
    )

    return (
        violating_param_realization,
        scaled_violations,
        constraint_violated,
    )


def initialize_separation(model_data, config):
    """
    Initialize separation problem variables, and fix all first-stage
    variables to their corresponding values from most recent
    master problem solution.

    Parameters
    ----------
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

    # check: initial point feasible?
    for con in sep_model.component_data_objects(Constraint, active=True):
        lb, val, ub = value(con.lb), value(con.body), value(con.ub)
        lb_viol = val < lb - ABS_CON_CHECK_FEAS_TOL if lb is not None else False
        ub_viol = val > ub + ABS_CON_CHECK_FEAS_TOL if ub is not None else False
        if lb_viol or ub_viol:
            config.progress_logger.debug(con.name, lb, val, ub)


locally_acceptable = {tc.optimal, tc.locallyOptimal, tc.globallyOptimal}
globally_acceptable = {tc.optimal, tc.globallyOptimal}


def solver_call_separation(
        model_data,
        config,
        solve_globally,
        perf_cons_to_evaluate,
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
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the separation problem solution
        obtained.

    Returns
    -------
    solve_call_results : pyros.solve_data.SeparationSolveCallResults
        Solve results for separation problem of interest.
    """

    if solve_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers

    # keep track of solver statuses for output logging
    solver_status_dict = {}
    nlp_model = model_data.separation_model

    # === Initialize separation problem; fix first-stage variables
    initialize_separation(model_data, config)

    solve_call_results = SeparationSolveCallResults(
        solved_globally=solve_globally,
        time_out=False,
        results_list=[],
        found_violation=False,
    )
    timer = TicTocTimer()
    for opt in solvers:
        orig_setting, custom_setting_present = adjust_solver_time_settings(
            model_data.timing,
            opt,
            config,
        )
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
            config.progress_logger.error(
                f"Solver {repr(opt)} encountered exception attempting to "
                "optimize separation problem in iteration "
                f"{model_data.iteration}"
            )
            raise
        else:
            setattr(
                results.solver,
                TIC_TOC_SOLVE_TIME_ATTR,
                timer.toc(msg=None),
            )
        finally:
            revert_solver_max_time_adjustment(
                opt,
                orig_setting,
                custom_setting_present,
                config,
            )

        # record termination condition for this particular solver
        solver_status_dict[str(opt)] = results.solver.termination_condition
        solve_call_results.results_list.append(results)

        # has PyROS time limit been reached?
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                solve_call_results.time_out = True
                return solve_call_results

        # if separation problem solved to optimality, record results
        # and exit
        acceptable_conditions = (
            globally_acceptable if solve_globally else locally_acceptable
        )
        optimal_termination = solve_call_results.termination_acceptable(
            acceptable_conditions,
        )
        if optimal_termination:
            nlp_model.solutions.load_from(results)
            (
                solve_call_results.violating_param_realization,
                solve_call_results.list_of_scaled_violations,
                solve_call_results.found_violation,
            ) = update_solve_data_violations(
                model_data,
                config,
                perf_cons_to_evaluate,
            )
            return solve_call_results

    # All subordinate solvers failed to optimize model to appropriate
    # termination condition. PyROS will terminate with subsolver
    # error. At this point, export model if desired
    save_dir = config.subproblem_file_directory
    if save_dir and config.keepfiles:
        objective = str(
            list(nlp_model.component_data_objects(Objective, active=True))[0]
            .name
        )
        name = os.path.join(
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
        nlp_model.write(name, io_options={'symbolic_solver_labels': True})
        output_logger(
            config=config,
            separation_error=True,
            filename=name,
            iteration=model_data.iteration,
            objective=objective,
            status_dict=solver_status_dict,
        )

    return solve_call_results


def discrete_solve(
        model_data,
        config,
        solve_globally,
        perf_cons_to_evaluate,
        ):
    """
    Obtain separation problem solution for each scenario
    of the discrete uncertainty set.

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
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the separation problem solution
        obtained.

    Returns
    -------
    solve_data_list : list of SeparationResult
        Separation problem solve results for each scenario.
    """
    # Deactivate uncertainty set constraints
    conlist = model_data.separation_model.util.uncertainty_set_constraint
    _constraints = list(conlist.values())
    conlist.deactivate()

    # constraints are clustered by scenario into groups of
    # size equal to the dimension of the set
    chunk_size = len(model_data.separation_model.util.uncertain_param_vars)

    # ensure scenarios already in master are skipped from separation
    constraints_to_skip = ComponentSet()
    for pnt in model_data.points_added_to_master:
        _idx = config.uncertainty_set.scenarios.index(tuple(pnt))
        skip_index_list = list(
            range(chunk_size * _idx, chunk_size * _idx + chunk_size)
        )
        for _index in range(len(_constraints)):
            if _index in skip_index_list:
                constraints_to_skip.add(_constraints[_index])
    constraints = list(c for c in _constraints if c not in constraints_to_skip)

    # solve separation problem for each parameter realization
    # (i.e. ea. scenario) not already added to master
    solve_data_list = []
    for i in range(0, len(constraints), chunk_size):
        chunk = list(constraints[i:i + chunk_size])
        for idx, con in enumerate(chunk):
            con.activate()
            model_data.separation_model.util.uncertain_param_vars[idx].fix(
                con.lower,
            )
            con.deactivate()

        solve_data = solver_call_separation(
            model_data=model_data,
            config=config,
            solve_globally=solve_globally,
            perf_cons_to_evaluate=perf_cons_to_evaluate,
        )
        solve_data_list.append(solve_data)
        for con in chunk:
            con.deactivate()

    return solve_data_list
