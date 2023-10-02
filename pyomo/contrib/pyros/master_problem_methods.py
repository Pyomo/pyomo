"""
Functions for handling the construction and solving of the GRCS master problem via ROSolver
"""
from pyomo.core.base import (
    ConcreteModel,
    Block,
    Var,
    Objective,
    Constraint,
    ConstraintList,
    SortComponents,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
    selective_clone,
    ObjectiveType,
    pyrosTerminationCondition,
    process_termination_condition_master_problem,
    adjust_solver_time_settings,
    revert_solver_max_time_adjustment,
    get_main_elapsed_time,
)
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name

from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR


def initial_construct_master(model_data):
    """
    Constructs the iteration 0 master problem
    return: a MasterProblemData object containing the master_model object
    """
    m = ConcreteModel()
    m.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)

    master_data = MasterProblemData()
    master_data.original = model_data.working_model.clone()
    master_data.master_model = m
    master_data.timing = model_data.timing

    return master_data


def get_state_vars(model, iterations):
    """
    Obtain the state variables of a two-stage model
    for a given (sequence of) iterations corresponding
    to model blocks.

    Parameters
    ----------
    model : ConcreteModel
        PyROS model.
    iterations : iterable
        Iterations to consider.

    Returns
    -------
    iter_state_var_map : dict
        Mapping from iterations to list(s) of state vars.
    """
    iter_state_var_map = dict()
    for itn in iterations:
        state_vars = [
            var for blk in model.scenarios[itn, :] for var in blk.util.state_vars
        ]
        iter_state_var_map[itn] = state_vars

    return iter_state_var_map


def construct_master_feasibility_problem(model_data, config):
    """
    Construct a slack-variable based master feasibility model.
    Initialize all model variables appropriately, and scale slack variables
    as well.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    model : ConcreteModel
        Slack variable model.
    """

    model = model_data.master_model.clone()

    # obtain mapping from master problem to master feasibility
    # problem variables
    varmap_name = unique_component_name(model_data.master_model, 'pyros_var_map')
    setattr(
        model_data.master_model,
        varmap_name,
        list(model_data.master_model.component_data_objects(Var)),
    )
    model = model_data.master_model.clone()
    model_data.feasibility_problem_varmap = list(
        zip(getattr(model_data.master_model, varmap_name), getattr(model, varmap_name))
    )
    delattr(model_data.master_model, varmap_name)
    delattr(model, varmap_name)

    for obj in model.component_data_objects(Objective):
        obj.deactivate()
    iteration = model_data.iteration

    # first stage vars are already initialized appropriately.
    # initialize second-stage DOF variables using DR equation expressions
    if model.scenarios[iteration, 0].util.second_stage_variables:
        for blk in model.scenarios[iteration, :]:
            for eq in blk.util.decision_rule_eqns:
                vars_in_dr_eq = ComponentSet(identify_variables(eq.body))
                ssv_set = ComponentSet(blk.util.second_stage_variables)

                # get second-stage var in DR eqn. should only be one var
                ssv_in_dr_eq = [var for var in vars_in_dr_eq if var in ssv_set][0]

                # update var value for initialization
                # fine since DR eqns are f(d) - z == 0 (not z - f(d) == 0)
                ssv_in_dr_eq.set_value(0)
                ssv_in_dr_eq.set_value(value(eq.body))

    # initialize state vars to previous master solution values
    if iteration != 0:
        stvar_map = get_state_vars(model, [iteration, iteration - 1])
        for current, prev in zip(stvar_map[iteration], stvar_map[iteration - 1]):
            current.set_value(value(prev))

    # constraints to which slacks should be added
    # (all the constraints for the current iteration, except the DR eqns)
    targets = []
    for blk in model.scenarios[iteration, :]:
        if blk.util.second_stage_variables:
            dr_eqs = blk.util.decision_rule_eqns
        else:
            dr_eqs = list()

        targets.extend(
            [
                con
                for con in blk.component_data_objects(
                    Constraint, active=True, descend_into=True
                )
                if con not in dr_eqs
            ]
        )

    # retain original constraint exprs (for slack initialization and scaling)
    pre_slack_con_exprs = ComponentMap((con, con.body - con.upper) for con in targets)

    # add slack variables and objective
    # inequalities g(v) <= b become g(v) -- s^-<= b
    # equalities h(v) == b become h(v) -- s^- + s^+ == b
    TransformationFactory("core.add_slack_variables").apply_to(model, targets=targets)
    slack_vars = ComponentSet(
        model._core_add_slack_variables.component_data_objects(Var, descend_into=True)
    )

    # initialize and scale slack variables
    for con in pre_slack_con_exprs:
        # obtain slack vars in updated constraints
        # and their coefficients (+/-1) in the constraint expression
        repn = generate_standard_repn(con.body)
        slack_var_coef_map = ComponentMap()
        for idx in range(len(repn.linear_vars)):
            var = repn.linear_vars[idx]
            if var in slack_vars:
                slack_var_coef_map[var] = repn.linear_coefs[idx]

        slack_substitution_map = dict()

        for slack_var in slack_var_coef_map:
            # coefficient determines whether the slack is a +ve or -ve slack
            if slack_var_coef_map[slack_var] == -1:
                con_slack = max(0, value(pre_slack_con_exprs[con]))
            else:
                con_slack = max(0, -value(pre_slack_con_exprs[con]))

            # initialize slack var, evaluate scaling coefficient
            scaling_coeff = 1
            slack_var.set_value(con_slack)

            # update expression replacement map
            slack_substitution_map[id(slack_var)] = scaling_coeff * slack_var

        # finally, scale slack(s)
        con.set_value(
            (
                replace_expressions(con.lower, slack_substitution_map),
                replace_expressions(con.body, slack_substitution_map),
                replace_expressions(con.upper, slack_substitution_map),
            )
        )

    return model


def solve_master_feasibility_problem(model_data, config):
    """
    Solve a slack variable-based feasibility model derived
    from the master problem. Initialize the master problem
    to the  solution found by the optimizer if solved successfully,
    or to the initial point provided to the solver otherwise.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Solver results.
    """
    model = construct_master_feasibility_problem(model_data, config)

    active_obj = next(model.component_data_objects(Objective, active=True))

    config.progress_logger.debug("Solving master feasibility problem")
    config.progress_logger.debug(
        f" Initial objective (total slack): {value(active_obj)}"
    )

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(
        model_data.timing, solver, config
    )
    model_data.timing.start_timer("main.master_feasibility")
    timer.tic(msg=None)
    try:
        results = solver.solve(model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        # account for possible external subsolver errors
        # (such as segmentation faults, function evaluation
        # errors, etc.)
        config.progress_logger.error(
            f"Optimizer {repr(solver)} encountered exception "
            "attempting to solve master feasibility problem in iteration "
            f"{model_data.iteration}."
        )
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer("main.master_feasibility")
    finally:
        revert_solver_max_time_adjustment(
            solver, orig_setting, custom_setting_present, config
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
            f"of iteration {model_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unoptimized point for master problem initialization."
        )

    # load master feasibility point to master model
    for master_var, feas_var in model_data.feasibility_problem_varmap:
        master_var.set_value(feas_var.value, skip_validation=True)

    return results


def minimize_dr_vars(model_data, config):
    """
    Polish the PyROS decision rule determined for the most
    recently solved master problem by minimizing the collective
    L1 norm of the vector of all decision rule variables.

    Parameters
    ----------
    model_data : MasterProblemData
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
    # config.progress_logger.info("Executing decision rule variable polishing solve.")
    model = model_data.master_model
    polishing_model = model.clone()

    first_stage_variables = polishing_model.scenarios[0, 0].util.first_stage_variables
    decision_rule_vars = polishing_model.scenarios[0, 0].util.decision_rule_vars

    polishing_model.obj.deactivate()
    index_set = decision_rule_vars[0].index_set()
    polishing_model.tau_vars = []
    # ==========
    for idx in range(len(decision_rule_vars)):
        polishing_model.scenarios[0, 0].add_component(
            "polishing_var_" + str(idx),
            Var(index_set, initialize=1e6, domain=NonNegativeReals),
        )
        polishing_model.tau_vars.append(
            getattr(polishing_model.scenarios[0, 0], "polishing_var_" + str(idx))
        )
    # ==========
    this_iter = polishing_model.scenarios[max(polishing_model.scenarios.keys())[0], 0]
    nom_block = polishing_model.scenarios[0, 0]
    if config.objective_focus == ObjectiveType.nominal:
        obj_val = value(
            this_iter.second_stage_objective + this_iter.first_stage_objective
        )
        polishing_model.scenarios[0, 0].polishing_constraint = Constraint(
            expr=obj_val
            >= nom_block.second_stage_objective + nom_block.first_stage_objective
        )
    elif config.objective_focus == ObjectiveType.worst_case:
        polishing_model.zeta.fix()  # Searching equivalent optimal solutions given optimal zeta

    # === Make absolute value constraints on polishing_vars
    polishing_model.scenarios[
        0, 0
    ].util.absolute_var_constraints = cons = ConstraintList()
    uncertain_params = nom_block.util.uncertain_params
    if config.decision_rule_order == 1:
        for i, tau in enumerate(polishing_model.tau_vars):
            for j in range(len(this_iter.util.decision_rule_vars[i])):
                if j == 0:
                    cons.add(-tau[j] <= this_iter.util.decision_rule_vars[i][j])
                    cons.add(this_iter.util.decision_rule_vars[i][j] <= tau[j])
                else:
                    cons.add(
                        -tau[j]
                        <= this_iter.util.decision_rule_vars[i][j]
                        * uncertain_params[j - 1]
                    )
                    cons.add(
                        this_iter.util.decision_rule_vars[i][j]
                        * uncertain_params[j - 1]
                        <= tau[j]
                    )
    elif config.decision_rule_order == 2:
        l = list(range(len(uncertain_params)))
        index_pairs = list(it.combinations(l, 2))
        for i, tau in enumerate(polishing_model.tau_vars):
            Z = this_iter.util.decision_rule_vars[i]
            indices = list(k for k in range(len(Z)))
            for r in indices:
                if r == 0:
                    cons.add(-tau[r] <= Z[r])
                    cons.add(Z[r] <= tau[r])
                elif r <= len(uncertain_params) and r > 0:
                    cons.add(-tau[r] <= Z[r] * uncertain_params[r - 1])
                    cons.add(Z[r] * uncertain_params[r - 1] <= tau[r])
                elif r <= len(indices) - len(uncertain_params) - 1 and r > len(
                    uncertain_params
                ):
                    cons.add(
                        -tau[r]
                        <= Z[r]
                        * uncertain_params[
                            index_pairs[r - len(uncertain_params) - 1][0]
                        ]
                        * uncertain_params[
                            index_pairs[r - len(uncertain_params) - 1][1]
                        ]
                    )
                    cons.add(
                        Z[r]
                        * uncertain_params[
                            index_pairs[r - len(uncertain_params) - 1][0]
                        ]
                        * uncertain_params[
                            index_pairs[r - len(uncertain_params) - 1][1]
                        ]
                        <= tau[r]
                    )
                elif r > len(indices) - len(uncertain_params) - 1:
                    cons.add(
                        -tau[r]
                        <= Z[r]
                        * uncertain_params[
                            r - len(index_pairs) - len(uncertain_params) - 1
                        ]
                        ** 2
                    )
                    cons.add(
                        Z[r]
                        * uncertain_params[
                            r - len(index_pairs) - len(uncertain_params) - 1
                        ]
                        ** 2
                        <= tau[r]
                    )
    else:
        raise NotImplementedError(
            "Decision rule variable polishing has not been generalized to decision_rule_order "
            + str(config.decision_rule_order)
            + "."
        )

    polishing_model.scenarios[0, 0].polishing_obj = Objective(
        expr=sum(
            sum(tau[j] for j in tau.index_set()) for tau in polishing_model.tau_vars
        )
    )

    # === Fix design
    for d in first_stage_variables:
        d.fix()

    # === Unfix DR vars
    num_dr_vars = len(
        model.scenarios[0, 0].util.decision_rule_vars[0]
    )  # there is at least one dr var
    num_uncertain_params = len(config.uncertain_params)

    if model.const_efficiency_applied:
        for d in decision_rule_vars:
            for i in range(1, num_dr_vars):
                d[i].fix(0)
                d[0].unfix()
    elif model.linear_efficiency_applied:
        for d in decision_rule_vars:
            d.unfix()
            for i in range(num_uncertain_params + 1, num_dr_vars):
                d[i].fix(0)
    else:
        for d in decision_rule_vars:
            d.unfix()

    # === Unfix all control var values
    for block in polishing_model.scenarios.values():
        for c in block.util.second_stage_variables:
            c.unfix()
        if model.const_efficiency_applied:
            for d in block.util.decision_rule_vars:
                for i in range(1, num_dr_vars):
                    d[i].fix(0)
                    d[0].unfix()
        elif model.linear_efficiency_applied:
            for d in block.util.decision_rule_vars:
                d.unfix()
                for i in range(num_uncertain_params + 1, num_dr_vars):
                    d[i].fix(0)
        else:
            for d in block.util.decision_rule_vars:
                d.unfix()

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    config.progress_logger.debug("Solving DR polishing problem")

    # NOTE: this objective evalaution may not be accurate, due
    #       to the current initialization scheme for the auxiliary
    #       variables. new initialization will be implemented in the
    #       near future.
    polishing_obj = polishing_model.scenarios[0, 0].polishing_obj
    config.progress_logger.debug(f" Initial DR norm: {value(polishing_obj)}")

    # === Solve the polishing model
    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(
        model_data.timing, solver, config
    )
    model_data.timing.start_timer("main.dr_polishing")
    timer.tic(msg=None)
    try:
        results = solver.solve(polishing_model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        config.progress_logger.error(
            f"Optimizer {repr(solver)} encountered an exception "
            "attempting to solve decision rule polishing problem "
            f"in iteration {model_data.iteration}"
        )
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer("main.dr_polishing")
    finally:
        revert_solver_max_time_adjustment(
            solver, orig_setting, custom_setting_present, config
        )

    # interested in the time and termination status for debugging
    # purposes
    config.progress_logger.debug(" Done solving DR polishing problem")
    config.progress_logger.debug(
        f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
    )
    config.progress_logger.debug(
        f"  Termination status: {results.solver.termination_condition} "
    )

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal, tc.feasible}
    if results.solver.termination_condition not in acceptable:
        # continue with "unpolished" master model solution
        config.progress_logger.warning(
            "Could not successfully solve DR polishing problem "
            f"of iteration {model_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unpolished master problem solution."
        )
        return results, False

    # update master model second-stage, state, and decision rule
    # variables to polishing model solution
    polishing_model.solutions.load_from(results)
    for idx, blk in model_data.master_model.scenarios.items():
        ssv_zip = zip(
            blk.util.second_stage_variables,
            polishing_model.scenarios[idx].util.second_stage_variables,
        )
        sv_zip = zip(
            blk.util.state_vars, polishing_model.scenarios[idx].util.state_vars
        )
        for master_ssv, polish_ssv in ssv_zip:
            master_ssv.set_value(value(polish_ssv))
        for master_sv, polish_sv in sv_zip:
            master_sv.set_value(value(polish_sv))

        # update master problem decision rule variables
        dr_var_zip = zip(
            blk.util.decision_rule_vars,
            polishing_model.scenarios[idx].util.decision_rule_vars,
        )
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)

    config.progress_logger.debug(f" Optimized DR norm: {value(polishing_obj)}")
    config.progress_logger.debug(" Polished Master objective:")

    # print master solution
    if config.objective_focus == ObjectiveType.worst_case:
        worst_blk_idx = max(
            model_data.master_model.scenarios.keys(),
            key=lambda idx: value(
                model_data.master_model.scenarios[idx].second_stage_objective
            ),
        )
    else:
        worst_blk_idx = (0, 0)

    # debugging: summarize objective breakdown
    worst_master_blk = model_data.master_model.scenarios[worst_blk_idx]
    config.progress_logger.debug(
        "  First-stage objective: " f"{value(worst_master_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        "  Second-stage objective: " f"{value(worst_master_blk.second_stage_objective)}"
    )
    polished_master_obj = value(
        worst_master_blk.first_stage_objective + worst_master_blk.second_stage_objective
    )
    config.progress_logger.debug(f"  Objective: {polished_master_obj}")

    return results, True


def add_p_robust_constraint(model_data, config):
    """
    p-robustness--adds constraints to the master problem ensuring that the
    optimal k-th iteration solution is within (1+rho) of the nominal
    objective. The parameter rho is specified by the user and should be between.
    """
    rho = config.p_robustness['rho']
    model = model_data.master_model
    block_0 = model.scenarios[0, 0]
    frac_nom_cost = (1 + rho) * (
        block_0.first_stage_objective + block_0.second_stage_objective
    )

    for block_k in model.scenarios[model_data.iteration, :]:
        model.p_robust_constraints.add(
            block_k.first_stage_objective + block_k.second_stage_objective
            <= frac_nom_cost
        )
    return


def add_scenario_to_master(model_data, violations):
    """
    Add block to master, without cloning the master_model.first_stage_variables
    """

    m = model_data.master_model
    i = max(m.scenarios.keys())[0] + 1

    # === Add a block to master for each violation
    idx = 0  # Only supporting adding single violation back to master in v1
    new_block = selective_clone(
        m.scenarios[0, 0], m.scenarios[0, 0].util.first_stage_variables
    )
    m.scenarios[i, idx].transfer_attributes_from(new_block)

    # === Set uncertain params in new block(s) to correct value(s)
    for j, p in enumerate(m.scenarios[i, idx].util.uncertain_params):
        p.set_value(violations[j])

    return


def higher_order_decision_rule_efficiency(config, model_data):
    # === Efficiencies for decision rules
    #  if iteration <= |q| then all d^n where n > 1 are fixed to 0
    #  if iteration == 0, all d^n, n > 0 are fixed to 0
    #  These efficiencies should be carried through as d* to polishing
    nlp_model = model_data.master_model
    if config.decision_rule_order != None and len(config.second_stage_variables) > 0:
        #  Ensure all are unfixed unless next conditions are met...
        for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
            dr_var.unfix()
        num_dr_vars = len(
            nlp_model.scenarios[0, 0].util.decision_rule_vars[0]
        )  # there is at least one dr var
        num_uncertain_params = len(config.uncertain_params)
        nlp_model.const_efficiency_applied = False
        nlp_model.linear_efficiency_applied = False
        if model_data.iteration == 0:
            nlp_model.const_efficiency_applied = True
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(1, num_dr_vars):
                    dr_var[i].fix(0)
        elif (
            model_data.iteration <= num_uncertain_params
            and config.decision_rule_order > 1
        ):
            # Only applied in DR order > 1 case
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(num_uncertain_params + 1, num_dr_vars):
                    nlp_model.linear_efficiency_applied = True
                    dr_var[i].fix(0)
    return


def solver_call_master(model_data, config, solver, solve_data):
    """
    Invoke subsolver(s) on PyROS master problem.

    Parameters
    ----------
    model_data : MasterProblemData
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
    nlp_model = model_data.master_model
    master_soln = solve_data
    solver_term_cond_dict = {}

    if config.solve_master_globally:
        solvers = [solver] + config.backup_global_solvers
    else:
        solvers = [solver] + config.backup_local_solvers

    higher_order_decision_rule_efficiency(config, model_data)

    solve_mode = "global" if config.solve_master_globally else "local"
    config.progress_logger.debug("Solving master problem")

    timer = TicTocTimer()
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(
                f"Invoking backup solver {opt!r} "
                f"(solver {idx + 1} of {len(solvers)}) for "
                f"master problem of iteration {model_data.iteration}."
            )
        orig_setting, custom_setting_present = adjust_solver_time_settings(
            model_data.timing, opt, config
        )
        model_data.timing.start_timer("main.master")
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
                f"Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) "
                "encountered exception attempting to "
                f"solve master problem in iteration {model_data.iteration}"
            )
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
            model_data.timing.stop_timer("main.master")
        finally:
            revert_solver_max_time_adjustment(
                solver, orig_setting, custom_setting_present, config
            )

        optimal_termination = check_optimal_termination(results)
        infeasible = results.solver.termination_condition == tc.infeasible

        if optimal_termination:
            nlp_model.solutions.load_from(results)

        # record master problem termination conditions
        # for this particular subsolver
        # pyros termination condition is determined later in the
        # algorithm
        solver_term_cond_dict[str(opt)] = str(results.solver.termination_condition)
        master_soln.termination_condition = results.solver.termination_condition
        master_soln.pyros_termination_condition = None
        (
            try_backup,
            _,
        ) = (
            master_soln.master_subsolver_results
        ) = process_termination_condition_master_problem(config=config, results=results)

        master_soln.nominal_block = nlp_model.scenarios[0, 0]
        master_soln.results = results
        master_soln.master_model = nlp_model

        # if model was solved successfully, update/record the results
        # (nominal block DOF variable and objective values)
        if not try_backup and not infeasible:
            master_soln.fsv_vals = list(
                v.value for v in nlp_model.scenarios[0, 0].util.first_stage_variables
            )
            if config.objective_focus is ObjectiveType.nominal:
                master_soln.ssv_vals = list(
                    v.value
                    for v in nlp_model.scenarios[0, 0].util.second_stage_variables
                )
                master_soln.second_stage_objective = value(
                    nlp_model.scenarios[0, 0].second_stage_objective
                )
            else:
                idx = max(nlp_model.scenarios.keys())[0]
                master_soln.ssv_vals = list(
                    v.value
                    for v in nlp_model.scenarios[idx, 0].util.second_stage_variables
                )
                master_soln.second_stage_objective = value(
                    nlp_model.scenarios[idx, 0].second_stage_objective
                )
            master_soln.first_stage_objective = value(
                nlp_model.scenarios[0, 0].first_stage_objective
            )

            # debugging: log breakdown of master objective
            config.progress_logger.debug(" Optimized master objective breakdown:")
            config.progress_logger.debug(
                f"  First-stage objective: {master_soln.first_stage_objective}"
            )
            config.progress_logger.debug(
                f"  Second-stage objective: {master_soln.second_stage_objective}"
            )
            master_obj = (
                master_soln.first_stage_objective + master_soln.second_stage_objective
            )
            config.progress_logger.debug(f"  Objective: {master_obj}")
            config.progress_logger.debug(
                f" Termination condition: {results.solver.termination_condition}"
            )
            config.progress_logger.debug(
                f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
            )

            master_soln.nominal_block = nlp_model.scenarios[0, 0]
            master_soln.results = results
            master_soln.master_model = nlp_model

        # if PyROS time limit exceeded, exit loop and return solution
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
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
                + model_data.original.name
                + "_master_"
                + str(model_data.iteration)
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

    deterministic_model_qual = (
        " (i.e., the deterministic model)" if model_data.iteration == 0 else ""
    )
    deterministic_msg = (
        (
            " Please ensure your deterministic model "
            f"is solvable by at least one of the subordinate {solve_mode} "
            "optimizers provided."
        )
        if model_data.iteration == 0
        else ""
    )
    master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
    config.progress_logger.warning(
        f"Could not successfully solve master problem of iteration "
        f"{model_data.iteration}{deterministic_model_qual} with any of the "
        f"provided subordinate {solve_mode} optimizers. "
        f"(Termination statuses: "
        f"{[term_cond for term_cond in solver_term_cond_dict.values()]}.)"
        f"{deterministic_msg}"
        f"{serialization_msg}"
    )

    return master_soln


def solve_master(model_data, config):
    """
    Solve the master problem
    """
    master_soln = MasterResult()

    # no master feas problem for iteration 0
    if model_data.iteration > 0:
        results = solve_master_feasibility_problem(model_data, config)
        master_soln.feasibility_problem_results = results

        # if pyros time limit reached, load time out status
        # to master results and return to caller
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                # load master model
                master_soln.master_model = model_data.master_model
                master_soln.nominal_block = model_data.master_model.scenarios[0, 0]

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

    solver = (
        config.global_solver if config.solve_master_globally else config.local_solver
    )

    return solver_call_master(
        model_data=model_data, config=config, solver=solver, solve_data=master_soln
    )
