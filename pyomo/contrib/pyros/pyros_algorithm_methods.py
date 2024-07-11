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
Methods for execution of the main PyROS cutting set algorithm.
"""

from itertools import chain

from pyomo.common.dependencies import numpy as np
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base import Constraint, Block, value, VarData
from pyomo.core.expr import MonomialTermExpression

from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
import pyomo.contrib.pyros.master_problem_methods as mp_methods
import pyomo.contrib.pyros.separation_problem_methods as sp_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
    check_time_limit_reached,
    ObjectiveType,
    get_time_from_solver,
    pyrosTerminationCondition,
    IterationLogRecord,
    generate_all_decision_rule_var_data_objects,
    get_main_elapsed_time,
    get_dr_var_to_monomial_map,
)



def update_grcs_solve_data(
    pyros_soln, term_cond, nominal_data, timing_data, separation_data, master_soln, k
):
    '''
    This function updates the results data container object to return to the user so that they have all pertinent
    information from the PyROS run.
    :param grcs_soln: PyROS solution data container object
    :param term_cond: PyROS termination condition
    :param nominal_data: Contains information on all nominal data (var values, objective)
    :param timing_data: Contains timing information on subsolver calls in PyROS
    :param separation_data: Separation model data container
    :param master_problem_subsolver_statuses: All master problem sub-solver termination conditions from the PyROS run
    :param separation_problem_subsolver_statuses: All separation problem sub-solver termination conditions from the PyROS run
    :param k: Iteration counter
    :return: None
    '''
    pyros_soln.pyros_termination_condition = term_cond
    pyros_soln.total_iters = k
    pyros_soln.nominal_data = nominal_data
    pyros_soln.timing_data = timing_data
    pyros_soln.separation_data = separation_data
    pyros_soln.master_soln = master_soln

    return


def _evaluate_shift(current, prev, initial, norm=None):
    if current.size == 0:
        return None
    else:
        normalizers = np.max(
            np.vstack((np.ones(initial.size), np.abs(initial))),
            axis=0,
        )
        return np.max(np.abs(current - prev) / normalizers)


def get_variable_value_data(working_blk, dr_var_to_monomial_map):
    """
    Get variable value data.
    """
    from collections import namedtuple

    VariableValueData = namedtuple(
        "VariableValueData",
        ("first_stage_variables", "second_stage_variables", "decision_rule_monomials"),
    )

    ep = working_blk.effective_var_partitioning

    first_stage_data = ComponentMap(
        (var, var.value) for var in ep.first_stage_variables
    )
    second_stage_data = ComponentMap(
        (var, var.value) for var in ep.second_stage_variables
    )
    dr_term_data = ComponentMap(
        (dr_var, value(monomial))
        for dr_var, monomial
        in get_dr_var_to_monomial_map(working_blk).items()
    )

    return VariableValueData(
        first_stage_variables=first_stage_data,
        second_stage_variables=second_stage_data,
        decision_rule_monomials=dr_term_data,
    )


def evaluate_variable_shifts(current_var_data, previous_var_data, initial_var_data):
    """
    Evaluate relative changes in the variable values
    across solutions to a working model block, such as the
    nominal master block.
    """
    if previous_var_data is None:
        return None, None, None
    else:
        var_shifts = []
        for attr in current_var_data._fields:
            var_shifts.append(
                _evaluate_shift(
                    current=np.array(list(getattr(current_var_data, attr).values())),
                    prev=np.array(list(getattr(previous_var_data, attr).values())),
                    initial=np.array(list(getattr(initial_var_data, attr).values())),
                )
            )

    return tuple(var_shifts)


def ROSolver_iterative_solve(model_data, config):
    """
    Solve an RO problem with the iterative GRCS algorithm.

    Parameters
    ----------
    model_data : model data object
        Model data object, equipped with the
        fully preprocessed working model.
    config : ConfigDict
        PyROS solver options

    Returns
    -------
    ...
    """
    master_data = mp_methods.MasterProblemData(model_data, config)
    separation_data = sp_methods.SeparationProblemData(model_data, config)

    # === Nominal information
    nominal_data = Block()
    nominal_data.nom_fsv_vals = []
    nominal_data.nom_ssv_vals = []
    nominal_data.nom_first_stage_cost = 0
    nominal_data.nom_second_stage_cost = 0
    nominal_data.nom_obj = 0

    # === Time information
    timing_data = Block()
    timing_data.total_master_solve_time = 0
    timing_data.total_separation_local_time = 0
    timing_data.total_separation_global_time = 0
    timing_data.total_dr_polish_time = 0

    # set up first-stage variable and DR variable sets
    nominal_master_blk = master_data.master_model.scenarios[0, 0]
    dr_var_monomial_map = get_dr_var_to_monomial_map(nominal_master_blk)

    # keep track of variable values for iteration logging
    first_iter_var_data = None
    previous_iter_var_data = None
    current_iter_var_data = None

    num_performance_cons = len(
        separation_data.separation_model.effective_performance_inequality_cons
    )
    IterationLogRecord.log_header(config.progress_logger.info)
    k = 0
    master_statuses = []
    while config.max_iter == -1 or k < config.max_iter:
        master_data.iteration = k

        # TODO: what about p-robustness?

        # === Solve Master Problem
        config.progress_logger.debug(f"PyROS working on iteration {k}...")
        master_soln = master_data.solve_master()

        # === Keep track of total time and subsolver termination conditions
        timing_data.total_master_solve_time += get_time_from_solver(master_soln.results)

        if k > 0:  # master feas problem not solved for iteration 0
            timing_data.total_master_solve_time += get_time_from_solver(
                master_soln.feasibility_problem_results
            )

        master_statuses.append(master_soln.results.solver.termination_condition)
        master_soln.master_problem_subsolver_statuses = master_statuses

        # check master solve status
        # to determine whether to terminate here
        if (
            master_soln.master_subsolver_results[1]
            is pyrosTerminationCondition.robust_infeasible
        ):
            term_cond = pyrosTerminationCondition.robust_infeasible
        elif (
            master_soln.pyros_termination_condition
            is pyrosTerminationCondition.subsolver_error
        ):
            term_cond = pyrosTerminationCondition.subsolver_error
        elif (
            master_soln.pyros_termination_condition
            is pyrosTerminationCondition.time_out
        ):
            term_cond = pyrosTerminationCondition.time_out
        else:
            term_cond = None
        if term_cond in {
            pyrosTerminationCondition.subsolver_error,
            pyrosTerminationCondition.time_out,
            pyrosTerminationCondition.robust_infeasible,
        }:
            log_record = IterationLogRecord(
                iteration=k,
                objective=None,
                first_stage_var_shift=None,
                second_stage_var_shift=None,
                dr_var_shift=None,
                num_violated_cons=None,
                max_violation=None,
                dr_polishing_success=None,
                all_sep_problems_solved=None,
                global_separation=None,
                elapsed_time=get_main_elapsed_time(model_data.timing),
            )
            log_record.log(config.progress_logger.info)
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=term_cond,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            return model_data, []

        polishing_successful = True
        polish_master_solution = (
            config.decision_rule_order != 0
            and nominal_master_blk.decision_rule_vars
            and k != 0
        )
        if polish_master_solution:
            polishing_results, polishing_successful = master_data.solve_dr_polishing()
            timing_data.total_dr_polish_time += get_time_from_solver(polishing_results)

        # track variable values
        current_iter_var_data = get_variable_value_data(
            nominal_master_blk,
            dr_var_monomial_map,
        )
        if k == 0:
            first_iter_var_data = current_iter_var_data
            previous_iter_var_data = None

        fsv_shift, ssv_shift, dr_var_shift = evaluate_variable_shifts(
            current_var_data=current_iter_var_data,
            previous_var_data=previous_iter_var_data,
            initial_var_data=first_iter_var_data,
        )

        # === Check if time limit reached after polishing
        if check_time_limit_reached(master_data.timing, config):
            iter_log_record = IterationLogRecord(
                iteration=k,
                objective=value(master_data.master_model.epigraph_obj),
                first_stage_var_shift=fsv_shift,
                second_stage_var_shift=ssv_shift,
                dr_var_shift=dr_var_shift,
                num_violated_cons=None,
                max_violation=None,
                dr_polishing_success=polishing_successful,
                all_sep_problems_solved=None,
                global_separation=None,
                elapsed_time=master_data.timing.get_main_elapsed_time(),
            )
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=pyrosTerminationCondition.time_out,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, []

        # === Solve Separation Problem
        separation_data.iteration = k
        separation_data.master_model = master_data.master_model
        separation_results = separation_data.solve_separation(master_data)
        separation_data.separation_problem_subsolver_statuses.extend(
            [
                res.solver.termination_condition
                for res in separation_results.generate_subsolver_results()
            ]
        )
        if separation_results.solved_globally:
            separation_data.total_global_separation_solves += 1

        # make updates based on separation results
        timing_data.total_separation_local_time += (
            separation_results.evaluate_local_solve_time(get_time_from_solver)
        )
        timing_data.total_separation_global_time += (
            separation_results.evaluate_global_solve_time(get_time_from_solver)
        )
        if separation_results.found_violation:
            scaled_violations = separation_results.scaled_violations
            if scaled_violations is not None:
                # can be None if time out or subsolver error
                # reported in separation
                separation_data.constraint_violations.append(scaled_violations.values())
        separation_data.points_separated = (
            separation_results.violating_param_realization
        )

        scaled_violations = [
            solve_call_res.scaled_violations[con]
            for con, solve_call_res in separation_results.main_loop_results.solver_call_results.items()
            if solve_call_res.scaled_violations is not None
        ]
        if scaled_violations:
            max_sep_con_violation = max(scaled_violations)
        else:
            max_sep_con_violation = None
        num_violated_cons = len(separation_results.violated_performance_constraints)

        all_sep_problems_solved = (
            len(scaled_violations) == num_performance_cons
            and not separation_results.subsolver_error
            and not separation_results.time_out
        ) or separation_results.all_discrete_scenarios_exhausted

        iter_log_record = IterationLogRecord(
            iteration=k,
            objective=value(master_data.master_model.epigraph_obj),
            first_stage_var_shift=fsv_shift,
            second_stage_var_shift=ssv_shift,
            dr_var_shift=dr_var_shift,
            num_violated_cons=num_violated_cons,
            max_violation=max_sep_con_violation,
            dr_polishing_success=polishing_successful,
            all_sep_problems_solved=all_sep_problems_solved,
            global_separation=separation_results.solved_globally,
            elapsed_time=get_main_elapsed_time(model_data.timing),
        )

        # terminate on time limit
        if separation_results.time_out:
            termination_condition = pyrosTerminationCondition.time_out
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # terminate on separation subsolver error
        if separation_results.subsolver_error:
            termination_condition = pyrosTerminationCondition.subsolver_error
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # === Check if we terminate due to robust optimality or feasibility,
        #     or in the event of bypassing global separation, no violations
        robustness_certified = separation_results.robustness_certified
        if robustness_certified:
            if config.bypass_global_separation:
                config.progress_logger.warning(
                    "Option to bypass global separation was chosen. "
                    "Robust feasibility and optimality of the reported "
                    "solution are not guaranteed."
                )
            robust_optimal = (
                config.solve_master_globally
                and config.objective_focus is ObjectiveType.worst_case
            )
            if robust_optimal:
                termination_condition = pyrosTerminationCondition.robust_optimal
            else:
                termination_condition = pyrosTerminationCondition.robust_feasible
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # === Add block to master at violation
        mp_methods.add_scenario_block_to_master_problem(
            master_model=master_data.master_model,
            scenario_idx=(k + 1, 0),
            param_realization=separation_results.violating_param_realization,
            from_block=nominal_master_blk,
            clone_first_stage_components=False,
        )
        separation_data.points_added_to_master[(k + 1, 0)] = (
            separation_results.violating_param_realization
        )
        separation_data.auxiliary_values_for_master_points[(k + 1, 0)] = (
            separation_results.auxiliary_param_values
        )

        config.progress_logger.debug("Points added to master:")
        config.progress_logger.debug(
            np.array([pt for pt in separation_data.points_added_to_master.values()])
        )

        # initialize second-stage and state variables
        # for new master block to separation
        # solution chosen by heuristic. consequently,
        # equality constraints should all be satisfied (up to tolerances).
        for var, val in separation_results.violating_separation_variable_values.items():
            master_var = master_data.master_model.scenarios[k + 1, 0].find_component(
                var
            )
            master_var.set_value(val)

        k += 1

        iter_log_record.log(config.progress_logger.info)
        previous_iter_var_data = current_iter_var_data

    # Iteration limit reached
    update_grcs_solve_data(
        pyros_soln=model_data,
        k=k - 1,  # remove last increment to fix iteration count
        term_cond=pyrosTerminationCondition.max_iter,
        nominal_data=nominal_data,
        timing_data=timing_data,
        separation_data=separation_data,
        master_soln=master_soln,
    )
    return model_data, separation_results
