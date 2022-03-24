'''
Methods for the execution of the grcs algorithm
'''

from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver, pyrosTerminationCondition
from pyomo.contrib.pyros.util import get_main_elapsed_time, output_logger, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet

def update_grcs_solve_data(pyros_soln, term_cond, nominal_data, timing_data, separation_data, master_soln, k):
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

def ROSolver_iterative_solve(model_data, config):
    '''
    GRCS algorithm implementation
    :model_data: ROSolveData object with deterministic model information
    :config: ConfigBlock for the instance being solved
    '''

    # === The "violation" e.g. uncertain parameter values added to the master problem are nominal in iteration 0
    #     User can supply a nominal_uncertain_param_vals if they want to set nominal to a certain point,
    #     Otherwise, the default init value for the params is used as nominal_uncertain_param_vals
    violation = list(p for p in config.nominal_uncertain_param_vals)

    # === Do coefficient matching
    constraints = [c for c in model_data.working_model.component_data_objects(Constraint) if c.equality
                   and c not in ComponentSet(model_data.working_model.util.decision_rule_eqns)]
    model_data.working_model.util.h_x_q_constraints = ComponentSet()
    for c in constraints:
        coeff_matching_success, robust_infeasible = coefficient_matching(model=model_data.working_model, constraint=c,
                                                          uncertain_params=model_data.working_model.util.uncertain_params,
                                                          config=config)
        if not coeff_matching_success and not robust_infeasible:
            raise ValueError("Equality constraint \"%s\" cannot be guaranteed to be robustly feasible, "
                             "given the current partitioning between first-stage, second-stage and state variables. "
                             "You might consider editing this constraint to reference some second-stage "
                             "and/or state variable(s)."
                             % c.name)
        elif not coeff_matching_success and robust_infeasible:
            config.progress_logger.info("PyROS has determined that the model is robust infeasible. "
                                        "One reason for this is that equality constraint \"%s\" cannot be satisfied "
                                        "against all realizations of uncertainty, "
                                        "given the current partitioning between first-stage, second-stage and state variables. "
                                         "You might consider editing this constraint to reference some (additional) second-stage "
                                         "and/or state variable(s)."
                                         % c.name)
            return None, None
        else:
            pass

    # h(x,q) == 0 becomes h'(x) == 0
    for c in model_data.working_model.util.h_x_q_constraints:
        c.deactivate()

    # === Build the master problem and master problem data container object
    master_data = master_problem_methods.initial_construct_master(model_data)

    # === If using p_robustness, add ConstraintList for additional constraints
    if config.p_robustness:
        master_data.master_model.p_robust_constraints = ConstraintList()

    # === Add scenario_0
    master_data.master_model.scenarios[0, 0].transfer_attributes_from(master_data.original.clone())
    if len(master_data.master_model.scenarios[0,0].util.uncertain_params) != len(violation):
        raise ValueError


    # === Set the nominal uncertain parameters to the violation values
    for i, v in enumerate(violation):
        master_data.master_model.scenarios[0, 0].util.uncertain_params[i].value = v

    # === Add objective function (assuming minimization of costs) with nominal second-stage costs
    if config.objective_focus is ObjectiveType.nominal:
        master_data.master_model.obj = Objective(
            expr=master_data.master_model.scenarios[0,0].first_stage_objective +
                 master_data.master_model.scenarios[0,0].second_stage_objective
        )
    elif config.objective_focus is ObjectiveType.worst_case:
        # === Worst-case cost objective
        master_data.master_model.zeta = Var(
                initialize=value(
                    master_data.master_model.scenarios[0, 0].first_stage_objective +
                    master_data.master_model.scenarios[0, 0].second_stage_objective,
                    exception=False)
        )
        master_data.master_model.obj = Objective(expr=master_data.master_model.zeta)
        master_data.master_model.scenarios[0,0].epigraph_constr = Constraint(expr=
                        master_data.master_model.scenarios[0, 0].first_stage_objective +
                        master_data.master_model.scenarios[0, 0].second_stage_objective <= master_data.master_model.zeta )
        master_data.master_model.scenarios[0,0].util.first_stage_variables.append(master_data.master_model.zeta)

    # === Add deterministic constraints to ComponentSet on original so that these become part of separation model
    master_data.original.util.deterministic_constraints = \
        ComponentSet(c for c in master_data.original.component_data_objects(Constraint, descend_into=True))

    # === Make separation problem model once before entering the solve loop
    separation_model = separation_problem_methods.make_separation_problem(model_data=master_data, config=config)

    # === Create separation problem data container object and add information to catalog during solve
    separation_data = SeparationProblemData()
    separation_data.separation_model = separation_model
    separation_data.points_separated = [] # contains last point separated in the separation problem
    separation_data.points_added_to_master = [config.nominal_uncertain_param_vals] # explicitly robust against in master
    separation_data.constraint_violations = [] # list of constraint violations for each iteration
    separation_data.total_global_separation_solves = 0 # number of times global solve is used
    separation_data.timing = master_data.timing # timing object

    # === Keep track of subsolver termination statuses from each iteration
    separation_data.separation_problem_subsolver_statuses = []

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

    dr_var_lists_original = []
    dr_var_lists_polished = []

    k = 0
    while config.max_iter == -1 or k < config.max_iter:
        master_data.iteration = k

        # === Add p-robust constraint if iteration > 0
        if k > 0 and config.p_robustness:
            master_problem_methods.add_p_robust_constraint(model_data=master_data, config=config)

        # === Solve Master Problem
        config.progress_logger.info("PyROS working on iteration %s..." % k)
        master_soln = master_problem_methods.solve_master(model_data=master_data, config=config)
        #config.progress_logger.info("Done solving Master Problem!")
        master_soln.master_problem_subsolver_statuses = []

        # === Keep track of total time and subsolver termination conditions
        timing_data.total_master_solve_time += get_time_from_solver(master_soln.results)
        timing_data.total_master_solve_time += get_time_from_solver(master_soln.feasibility_problem_results)

        master_soln.master_problem_subsolver_statuses.append(master_soln.results.solver.termination_condition)

        # === Check for robust infeasibility or error or time-out in master problem solve
        if master_soln.master_subsolver_results[1] is pyrosTerminationCondition.robust_infeasible:
            term_cond = pyrosTerminationCondition.robust_infeasible
            output_logger(config=config, robust_infeasible=True)
        elif master_soln.pyros_termination_condition is pyrosTerminationCondition.subsolver_error:
            term_cond = pyrosTerminationCondition.subsolver_error
        else:
            term_cond = None
        if term_cond == pyrosTerminationCondition.subsolver_error or \
                term_cond == pyrosTerminationCondition.robust_infeasible:
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=term_cond,
                                   nominal_data=nominal_data,
                                   timing_data=timing_data,
                                   separation_data=separation_data,
                                   master_soln=master_soln)
            return model_data, []
        # === Check if time limit reached
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                output_logger(config=config, time_out=True, elapsed=elapsed)
                update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=pyrosTerminationCondition.time_out,
                                       nominal_data=nominal_data,
                                       timing_data=timing_data,
                                       separation_data=separation_data,
                                       master_soln=master_soln)
                return model_data, []

        # === Save nominal information
        if k == 0:
            for val in master_soln.fsv_vals:
                nominal_data.nom_fsv_vals.append(val)

            for val in master_soln.ssv_vals:
                nominal_data.nom_ssv_vals.append(val)

            nominal_data.nom_first_stage_cost = master_soln.first_stage_objective
            nominal_data.nom_second_stage_cost = master_soln.second_stage_objective
            nominal_data.nom_obj = value(master_data.master_model.obj)


        if (
            # === Decision rule polishing (do not polish on first iteration if no ssv or if decision_rule_order = 0)
            (config.decision_rule_order != 0 and len(config.second_stage_variables) > 0 and k != 0)
        ):
            # === Save initial values of DR vars to file
            for varslist in master_data.master_model.scenarios[0,0].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_original.append(vals)

            polishing_results = master_problem_methods.minimize_dr_vars(model_data=master_data, config=config)
            timing_data.total_dr_polish_time += get_time_from_solver(polishing_results)

            #=== Save after polish
            for varslist in master_data.master_model.scenarios[0,0].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_polished.append(vals)

        # === Set up for the separation problem
        separation_data.opt_fsv_vals = [v.value for v in master_soln.master_model.scenarios[0,0].util.first_stage_variables]
        separation_data.opt_ssv_vals = master_soln.ssv_vals

        # === Provide master model scenarios to separation problem for initialization options
        separation_data.master_scenarios = master_data.master_model.scenarios

        if config.objective_focus is ObjectiveType.worst_case:
            separation_model.util.zeta = value(master_soln.master_model.obj)

        # === Solve Separation Problem
        separation_data.iteration = k
        separation_data.master_nominal_scenario = master_data.master_model.scenarios[0,0]

        separation_data.master_model = master_data.master_model

        separation_solns, violating_realizations, constr_violations, is_global, \
            local_sep_time, global_sep_time = \
                separation_problem_methods.solve_separation_problem(model_data=separation_data, config=config)

        for sep_soln_list in separation_solns:
            for s in sep_soln_list:
                separation_data.separation_problem_subsolver_statuses.append(s.termination_condition)

        if is_global:
            separation_data.total_global_separation_solves += 1

        timing_data.total_separation_local_time += local_sep_time
        timing_data.total_separation_global_time += global_sep_time

        separation_data.constraint_violations.append(constr_violations)


        if not any(s.found_violation for solve_data_list in separation_solns for s in solve_data_list):
            separation_data.points_separated = []
        else:
            separation_data.points_separated = violating_realizations

        # === Check if time limit reached
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                output_logger(config=config, time_out=True, elapsed=elapsed)
                termination_condition = pyrosTerminationCondition.time_out
                update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition,
                                       nominal_data=nominal_data,
                                       timing_data=timing_data,
                                       separation_data=separation_data,
                                       master_soln=master_soln)
                return model_data, separation_solns

        # === Check if we exit due to solver returning unsatisfactory statuses (not in permitted_termination_conditions)
        local_solve_term_conditions = {TerminationCondition.optimal, TerminationCondition.locallyOptimal,
                                       TerminationCondition.globallyOptimal}
        global_solve_term_conditions = {TerminationCondition.optimal, TerminationCondition.globallyOptimal}
        if (is_global and any((s.termination_condition not in global_solve_term_conditions)
                                  for sep_soln_list in separation_solns for s in sep_soln_list)) or \
            (not is_global and any((s.termination_condition not in local_solve_term_conditions)
                                  for sep_soln_list in separation_solns for s in sep_soln_list)):
            termination_condition = pyrosTerminationCondition.subsolver_error
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition,
                                   nominal_data=nominal_data,
                                   timing_data=timing_data,
                                   separation_data=separation_data,
                                   master_soln=master_soln)
            return model_data, separation_solns

        # === Check if we terminate due to robust optimality or feasibility,
        #     or in the event of bypassing global separation, no violations
        if (not any(s.found_violation for sep_soln_list in separation_solns for s in sep_soln_list)
            and (is_global or config.bypass_global_separation)):
            output_logger(
                    config=config,
                    bypass_global_separation=config.bypass_global_separation
            )
            if config.solve_master_globally and config.objective_focus is ObjectiveType.worst_case:
                output_logger(config=config, robust_optimal=True)
                termination_condition = pyrosTerminationCondition.robust_optimal
            else:
                output_logger(config=config, robust_feasible=True)
                termination_condition = pyrosTerminationCondition.robust_feasible
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition,
                                   nominal_data=nominal_data,
                                   timing_data=timing_data,
                                   separation_data=separation_data,
                                   master_soln=master_soln)
            return model_data, separation_solns

        # === Add block to master at violation
        master_problem_methods.add_scenario_to_master(master_data, violating_realizations)
        separation_data.points_added_to_master.append(violating_realizations)

        k += 1

    output_logger(config=config, max_iter=True)
    update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=pyrosTerminationCondition.max_iter,
                           nominal_data=nominal_data,
                           timing_data=timing_data,
                           separation_data=separation_data,
                           master_soln=master_soln)

    # === In this case we still return the final solution objects for the last iteration
    return model_data, separation_solns






