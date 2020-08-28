"""Generates objective functions like L1, L2 and Linf distance"""

from pyomo.core import (Var, Objective, Reals,
                        RangeSet, Constraint, Block, sqrt)


def generate_L2_objective_function(model, setpoint_model, discretes_only=False):
    """Generate objective for minimum euclidean distance to setpoint_model
    L2 distance of (x,y) = \sqrt{\sum_i (x_i - y_i)^2}
    discretes_only -- only optimize on distance between the discrete variables
    """
    var_filter = (lambda v: v[1].is_binary()) if discretes_only \
        else (lambda v: True)

    model_vars, setpoint_vars = zip(*
                                    filter(
                                        var_filter,
                                        zip(model.component_data_objects(Var),
                                            setpoint_model.component_data_objects(Var))))

    assert len(model_vars) == len(
        setpoint_vars), "Trying to generate L1 objective function for models with different number of variables"

    return Objective(expr=(
        sum([(model_var - setpoint_var.value)**2
             for (model_var, setpoint_var) in
             zip(model_vars, setpoint_vars)])))


def generate_L1_objective_function(model, setpoint_model, discretes_only=False):
    """Generate objective for minimum L1 distance to setpoint model
    L1 distance of (x,y) = \sum_i |x_i - y_i|
    discretes_only -- only optimize on distance between the discrete variables
    """

    var_filter = (lambda v: v.is_binary()) if discretes_only \
        else (lambda v: True)
    model_vars = list(filter(var_filter, model.component_data_objects(Var)))
    setpoint_vars = list(
        filter(var_filter, setpoint_model.component_data_objects(Var)))
    assert len(model_vars) == len(
        setpoint_vars), "Trying to generate L1 objective function for models with different number of variables"

    obj_blk = model.L1_objective_function = Block()

    obj_blk.L1_obj_var = Var(domain=Reals, bounds=(0, None))
    obj_blk.L1_obj_ub_idx = RangeSet(len(model_vars))
    obj_blk.L1_obj_ub_constr = Constraint(
        obj_blk.L1_obj_ub_idx, rule=lambda i: obj_blk.L1_obj_var >= 0)
    obj_blk.L1_obj_lb_idx = RangeSet(len(model_vars))
    obj_blk.L1_obj_lb_constr = Constraint(
        obj_blk.L1_obj_lb_idx, rule=lambda i: obj_blk.L1_obj_var >= 0)  # 'empty' constraint (will be set later)

    for (c_lb, c_ub, v_model, v_setpoint) in zip(obj_blk.L1_obj_lb_idx,
                                                 obj_blk.L1_obj_ub_idx,
                                                 model_vars,
                                                 setpoint_vars):
        obj_blk.L1_obj_lb_constr[c_lb].set_value(
            expr=v_model - v_setpoint.value >= -obj_blk.L1_obj_var)
        obj_blk.L1_obj_ub_constr[c_ub].set_value(
            expr=v_model - v_setpoint.value <= obj_blk.L1_obj_var)

    return Objective(expr=obj_blk.L1_obj_var)


def feas_pump_converged(solve_data, config):
    """Calculates the euclidean norm between the discretes in the mip and nlp models"""
    distance = (sum((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if milp_var.is_binary()))

    return distance <= config.integer_tolerance


def feasibility_pump_loop(solve_data, config):
    """
    Main loop for MindtPy Algorithms

    This is the outermost function for the algorithms in this package; this function controls the progression of
    solving the model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    working_model = solve_data.working_model
    main_objective = next(
        working_model.component_data_objects(Objective, active=True))
    while solve_data.mip_iter < config.iteration_limit:

        config.logger.info(
            '---MindtPy Master Iteration %s---'
            % solve_data.mip_iter)

        solve_data.mip_subiter = 0
        # solve MILP master problem
        if config.strategy in {'OA', 'GOA', 'ECP'}:
            master_mip, master_mip_results = solve_OA_master(
                solve_data, config)
            if config.single_tree is False:
                if master_mip_results.solver.termination_condition is tc.optimal:
                    handle_master_mip_optimal(master_mip, solve_data, config)
                elif master_mip_results.solver.termination_condition is tc.infeasible:
                    handle_master_mip_infeasible(
                        master_mip, solve_data, config)
                    last_iter_cuts = True
                    break
                else:
                    handle_master_mip_other_conditions(master_mip, master_mip_results,
                                                       solve_data, config)
                # Call the MILP post-solve callback
                config.call_after_master_solve(master_mip, solve_data)
        else:
            raise NotImplementedError()

        if algorithm_should_terminate(solve_data, config, check_cycling=True):
            last_iter_cuts = False
            break

        if config.single_tree is False and config.strategy != 'ECP':  # if we don't use lazy callback, i.e. LP_NLP
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            fixed_nlp, fixed_nlp_result = solve_NLP_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_NLP_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition is tc.infeasible:
                handle_NLP_subproblem_infeasible(fixed_nlp, solve_data, config)
            else:
                handle_NLP_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                        solve_data, config)
            # Call the NLP post-solve callback
            config.call_after_subproblem_solve(fixed_nlp, solve_data)

        if algorithm_should_terminate(solve_data, config, check_cycling=False):
            last_iter_cuts = True
            break

        if config.strategy == 'ECP':
            add_ecp_cuts(solve_data.mip, solve_data, config)

        # if config.strategy == 'PSC':
        #     # If the hybrid algorithm is not making progress, switch to OA.
        #     progress_required = 1E-6
        #     if main_objective.sense == minimize:
        #         log = solve_data.LB_progress
        #         sign_adjust = 1
        #     else:
        #         log = solve_data.UB_progress
        #         sign_adjust = -1
        #     # Maximum number of iterations in which the lower (optimistic)
        #     # bound does not improve before switching to OA
        #     max_nonimprove_iter = 5
        #     making_progress = True
        #     # TODO-romeo Unneccesary for OA and LOA, right?
        #     for i in range(1, max_nonimprove_iter + 1):
        #         try:
        #             if (sign_adjust * log[-i]
        #                     <= (log[-i - 1] + progress_required)
        #                     * sign_adjust):
        #                 making_progress = False
        #             else:
        #                 making_progress = True
        #                 break
        #         except IndexError:
        #             # Not enough history yet, keep going.
        #             making_progress = True
        #             break
        #     if not making_progress and (
        #             config.strategy == 'hPSC' or
        #             config.strategy == 'PSC'):
        #         config.logger.info(
        #             'Not making enough progress for {} iterations. '
        #             'Switching to OA.'.format(max_nonimprove_iter))
        #         config.strategy = 'OA'

    # if add_nogood_cuts is True, the bound obtained in the last iteration is no reliable.
    # we correct it after the iteration.
    if config.add_nogood_cuts:
        bound_fix(solve_data, config, last_iter_cuts)
