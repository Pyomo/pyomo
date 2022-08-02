"""
Functions for handling the construction and solving of the GRCS master problem via ROSolver
"""
from pyomo.core.base import (ConcreteModel, Block,
                             Var,
                             Objective, Constraint,
                             ConstraintList, SortComponents)
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (selective_clone,
                                      ObjectiveType,
                                      pyrosTerminationCondition,
                                      process_termination_condition_master_problem,
                                      output_logger)
from pyomo.contrib.pyros.solve_data import (MasterProblemData,
                                            MasterResult)
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy


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
        fsv_set = ComponentSet(
                model.scenarios[itn, 0].util.first_stage_variables)
        state_vars = list()
        for blk in model.scenarios[itn, :]:
            ssv_set = ComponentSet(blk.util.second_stage_variables)
            state_vars.extend(
                    v for v in blk.component_data_objects(
                        Var,
                        active=True,
                        descend_into=True,
                        sort=SortComponents.deterministic,  # guarantee order
                    )
                    if v not in fsv_set and v not in ssv_set
            )
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
                ssv_in_dr_eq = [var for var in vars_in_dr_eq
                                if var in ssv_set][0]

                # update var value for initialization
                # fine since DR eqns are f(d) - z == 0 (not z - f(d) == 0)
                ssv_in_dr_eq.set_value(0)
                ssv_in_dr_eq.set_value(value(eq.body))

    # initialize state vars to previous master solution values
    if iteration != 0:
        stvar_map = get_state_vars(model, [iteration, iteration-1])
        for current, prev in zip(stvar_map[iteration], stvar_map[iteration-1]):
            current.set_value(value(prev))

    # constraints to which slacks should be added
    # (all the constraints for the current iteration, except the DR eqns)
    targets = []
    for blk in model.scenarios[iteration, :]:
        if blk.util.second_stage_variables:
            dr_eqs = blk.util.decision_rule_eqns
        else:
            dr_eqs = list()

        targets.extend([
            con for con in blk.component_data_objects(
                Constraint, active=True, descend_into=True)
            if con not in dr_eqs])

    # retain original constraint exprs (for slack initialization and scaling)
    pre_slack_con_exprs = ComponentMap([(con, con.body) for con in targets])

    # add slack variables and objective
    # inequalities g(v) <= b become g(v) -- s^-<= b
    # equalities h(v) == b become h(v) -- s^- + s^+ == b
    TransformationFactory("core.add_slack_variables").apply_to(model,
                                                               targets=targets)
    slack_vars = ComponentSet(
            model._core_add_slack_variables.component_data_objects(
                Var, descend_into=True)
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
            slack_substitution_map[id(slack_var)] = (scaling_coeff * slack_var)

        # finally, scale slack(s)
        con.set_value(
                (replace_expressions(con.lower, slack_substitution_map),
                 replace_expressions(con.body, slack_substitution_map),
                 replace_expressions(con.upper, slack_substitution_map),)
        )

    return model


def solve_master_feasibility_problem(model_data, config):
    """
    Solve a slack variable based feasibility model for the master problem
    """
    model = construct_master_feasibility_problem(model_data, config)

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    if not solver.available():
        raise RuntimeError("NLP solver %s is not available." %
                           config.solver)

    results = solver.solve(model, tee=config.tee, load_solutions=False)

    if check_optimal_termination(results):
        model.solutions.load_from(results)

        # load solution to master model
        for v in model.component_data_objects(Var):
            master_v = model_data.master_model.find_component(v)
            if master_v is not None:
                master_v.set_value(v.value, skip_validation=True)
    else:
        results.solver.termination_condition = tc.error
        results.solver.message = ("Cannot load a SolverResults object with "
                                  "bad status: error")
    return results


def minimize_dr_vars(model_data, config):
    """
    Decision rule polishing: For a given optimal design (x) determined in separation,
    and the optimal value for control vars (z), choose min magnitude decision_rule_var
    values.
    """
    #config.progress_logger.info("Executing decision rule variable polishing solve.")
    model = model_data.master_model
    polishing_model = model.clone()

    first_stage_variables = polishing_model.scenarios[0, 0].util.first_stage_variables
    decision_rule_vars = polishing_model.scenarios[0, 0].util.decision_rule_vars

    polishing_model.obj.deactivate()
    index_set = decision_rule_vars[0].index_set()
    polishing_model.tau_vars = []
    # ==========
    for idx in range(len(decision_rule_vars)):
        polishing_model.scenarios[0,0].add_component(
                "polishing_var_" + str(idx),
                Var(index_set, initialize=1e6, domain=NonNegativeReals))
        polishing_model.tau_vars.append(
            getattr(polishing_model.scenarios[0,0], "polishing_var_" + str(idx))
        )
    # ==========
    this_iter = polishing_model.scenarios[max(polishing_model.scenarios.keys())[0], 0]
    nom_block = polishing_model.scenarios[0, 0]
    if config.objective_focus == ObjectiveType.nominal:
        obj_val = value(this_iter.second_stage_objective + this_iter.first_stage_objective)
        polishing_model.scenarios[0,0].polishing_constraint = \
            Constraint(expr=obj_val >= nom_block.second_stage_objective + nom_block.first_stage_objective)
    elif config.objective_focus == ObjectiveType.worst_case:
        polishing_model.zeta.fix() # Searching equivalent optimal solutions given optimal zeta

    # === Make absolute value constraints on polishing_vars
    polishing_model.scenarios[0,0].util.absolute_var_constraints = cons = ConstraintList()
    uncertain_params = nom_block.util.uncertain_params
    if config.decision_rule_order == 1:
        for i, tau in enumerate(polishing_model.tau_vars):
            for j in range(len(this_iter.util.decision_rule_vars[i])):
                if j == 0:
                    cons.add(-tau[j] <= this_iter.util.decision_rule_vars[i][j])
                    cons.add(this_iter.util.decision_rule_vars[i][j] <= tau[j])
                else:
                    cons.add(-tau[j] <= this_iter.util.decision_rule_vars[i][j] * uncertain_params[j - 1])
                    cons.add(this_iter.util.decision_rule_vars[i][j] * uncertain_params[j - 1] <= tau[j])
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
                elif r <= len(indices) - len(uncertain_params) - 1 and r > len(uncertain_params):
                    cons.add(-tau[r] <= Z[r] * uncertain_params[index_pairs[r - len(uncertain_params) - 1][0]] * uncertain_params[
                        index_pairs[r - len(uncertain_params) - 1][1]])
                    cons.add(Z[r] * uncertain_params[index_pairs[r - len(uncertain_params) - 1][0]] *
                             uncertain_params[index_pairs[r - len(uncertain_params) - 1][1]] <= tau[r])
                elif r > len(indices) - len(uncertain_params) - 1:
                    cons.add(-tau[r] <= Z[r] * uncertain_params[r - len(index_pairs) - len(uncertain_params) - 1] ** 2)
                    cons.add(Z[r] * uncertain_params[r - len(index_pairs) - len(uncertain_params) - 1] ** 2 <= tau[r])
    else:
        raise NotImplementedError("Decision rule variable polishing has not been generalized to decision_rule_order "
                                  + str(config.decision_rule_order) + ".")

    polishing_model.scenarios[0,0].polishing_obj = \
        Objective(expr=sum(sum(tau[j] for j in tau.index_set()) for tau in polishing_model.tau_vars))

    # === Fix design
    for d in first_stage_variables:
        d.fix()

    # === Unfix DR vars
    num_dr_vars = len(model.scenarios[0, 0].util.decision_rule_vars[0])  # there is at least one dr var
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

    # === Solve the polishing model
    polish_soln = MasterResult()
    solver = config.global_solver

    if not solver.available():
        raise RuntimeError("NLP solver %s is not available." %
                           config.solver)
    try:
        results = solver.solve(polishing_model, tee=config.tee)
        polish_soln.termination_condition = results.solver.termination_condition
    except ValueError as err:
        polish_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
        polish_soln.termination_condition = tc.error
        raise

    polish_soln.fsv_values = list(v.value for v in polishing_model.scenarios[0, 0].util.first_stage_variables)
    polish_soln.ssv_values = list(v.value for v in polishing_model.scenarios[0, 0].util.second_stage_variables)
    polish_soln.first_stage_objective = value(nom_block.first_stage_objective)
    polish_soln.second_stage_objective = value(nom_block.second_stage_objective)

    # === Process solution by termination condition
    acceptable = [tc.optimal, tc.locallyOptimal, tc.feasible]
    if polish_soln.termination_condition not in acceptable:
        return results

    for i, d in enumerate(model_data.master_model.scenarios[0, 0].util.decision_rule_vars):
        for index in d:
            d[index].set_value(polishing_model.scenarios[0, 0].util.decision_rule_vars[i][index].value, skip_validation=True)

    return results


def add_p_robust_constraint(model_data, config):
    """
    p-robustness--adds constraints to the master problem ensuring that the
    optimal k-th iteration solution is within (1+rho) of the nominal
    objective. The parameter rho is specified by the user and should be between.
    """
    rho = config.p_robustness['rho']
    model = model_data.master_model
    block_0 = model.scenarios[0, 0]
    frac_nom_cost = (1 + rho) * (block_0.first_stage_objective +
                                        block_0.second_stage_objective)

    for block_k in model.scenarios[model_data.iteration, :]:
        model.p_robust_constraints.add(
            block_k.first_stage_objective + block_k.second_stage_objective
            <= frac_nom_cost)
    return


def add_scenario_to_master(model_data, violations):
    """
    Add block to master, without cloning the master_model.first_stage_variables
    """

    m = model_data.master_model
    i = max(m.scenarios.keys())[0] + 1

    # === Add a block to master for each violation
    idx = 0 # Only supporting adding single violation back to master in v1
    new_block = selective_clone(m.scenarios[0, 0], m.scenarios[0, 0].util.first_stage_variables)
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
        num_dr_vars = len(nlp_model.scenarios[0, 0].util.decision_rule_vars[0])  # there is at least one dr var
        num_uncertain_params = len(config.uncertain_params)
        nlp_model.const_efficiency_applied = False
        nlp_model.linear_efficiency_applied = False
        if model_data.iteration == 0:
            nlp_model.const_efficiency_applied = True
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(1, num_dr_vars):
                    dr_var[i].fix(0)
        elif model_data.iteration <= num_uncertain_params and config.decision_rule_order > 1:
            # Only applied in DR order > 1 case
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(num_uncertain_params + 1, num_dr_vars):
                    nlp_model.linear_efficiency_applied = True
                    dr_var[i].fix(0)
    return


def solver_call_master(model_data, config, solver, solve_data):
    '''
    Function interfacing with optimization solver
    :param model_data:
    :param config:
    :param solver:
    :param solve_data:
    :param is_global:
    :return:
    '''
    nlp_model = model_data.master_model
    master_soln = solve_data
    solver_term_cond_dict = {}

    if config.solve_master_globally:
        backup_solvers = deepcopy(config.backup_global_solvers)
    else:
        backup_solvers = deepcopy(config.backup_local_solvers)
    backup_solvers.insert(0, solver)

    if not solver.available():
        raise RuntimeError("NLP solver %s is not available." %
                           config.solver)

    higher_order_decision_rule_efficiency(config, model_data)

    while len(backup_solvers) > 0:
        solver = backup_solvers.pop(0)
        try:
            results = solver.solve(nlp_model, tee=config.tee)
        except ValueError as err:
            if 'Cannot load a SolverResults object with bad status: error' in str(err):
                results.solver.termination_condition = tc.error
                results.solver.message = str(err)
                master_soln.results = results
                master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
                return master_soln, ()
            else:
                raise
        solver_term_cond_dict[str(solver)] = str(results.solver.termination_condition)
        master_soln.termination_condition = results.solver.termination_condition
        master_soln.pyros_termination_condition = None  # determined later in the algorithm
        master_soln.fsv_vals = list(v.value for v in nlp_model.scenarios[0, 0].util.first_stage_variables)

        if config.objective_focus is ObjectiveType.nominal:
            master_soln.ssv_vals = list(v.value for v in nlp_model.scenarios[0, 0].util.second_stage_variables)
            master_soln.second_stage_objective = value(nlp_model.scenarios[0, 0].second_stage_objective)
        else:
            idx =  max(nlp_model.scenarios.keys())[0]
            master_soln.ssv_vals = list(v.value for v in nlp_model.scenarios[idx, 0].util.second_stage_variables)
            master_soln.second_stage_objective = value(nlp_model.scenarios[idx, 0].second_stage_objective)
        master_soln.first_stage_objective = value(nlp_model.scenarios[0, 0].first_stage_objective)

        master_soln.nominal_block = nlp_model.scenarios[0, 0]
        master_soln.results = results
        master_soln.master_model = nlp_model

        master_soln.master_subsolver_results = process_termination_condition_master_problem(config=config, results=results)

        if master_soln.master_subsolver_results[0] == False:
            return master_soln

    # === At this point, all sub-solvers have been tried and none returned an acceptable status or return code
    save_dir = config.subproblem_file_directory
    if save_dir and config.keepfiles:
        name = os.path.join(save_dir,  config.uncertainty_set.type + "_" + model_data.original.name + "_master_" + str(model_data.iteration) + ".bar")
        nlp_model.write(name, io_options={'symbolic_solver_labels':True})
        output_logger(config=config, master_error=True, status_dict=solver_term_cond_dict, filename=name, iteration=model_data.iteration)
    master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
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

    solver = config.global_solver if config.solve_master_globally else config.local_solver

    return solver_call_master(model_data=model_data, config=config, solver=solver,
                       solve_data=master_soln)
