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

import logging

logger = logging.getLogger(__name__)

import pyomo.environ as pyo
from pyomo.contrib.alternative_solutions import (
    aos_utils,
    shifted_lp,
    solution,
    solnpool,
)
from pyomo.contrib import appsi


def enumerate_linear_solutions(
    model,
    *,
    num_solutions=10,
    rel_opt_gap=None,
    abs_opt_gap=None,
    zero_threshold=1e-5,
    search_mode="optimal",
    solver="gurobi",
    solver_options={},
    tee=False,
    seed=None,
):
    """
    Finds alternative optimal solutions a (mixed-integer) linear program.

    This function implements the technique described here:

        S. Lee, C. Phalakornkule, M.M. Domach, and I.E. Grossmann,
        "Recursive MILP model for finding all the alternative optima in LP
        models for metabolic networks", Computers and Chemical Engineering,
        24 (2000) 711-716.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model
    num_solutions : int
        The maximum number of solutions to generate.
    rel_opt_gap : float or None
        The relative optimality gap for the original objective for which
        variable bounds will be found. None indicates that a relative gap
        constraint will not be added to the model.
    abs_opt_gap : float or None
        The absolute optimality gap for the original objective for which
        variable bounds will be found. None indicates that an absolute gap
        constraint will not be added to the model.
    zero_threshold: float
        The threshold for which a continuous variables' value is considered
        to be equal to zero.
    search_mode : 'optimal', 'random', or 'norm'
        Indicates the mode that is used to generate alternative solutions.
        The optimal mode finds the next best solution. The random mode
        finds an alternative solution in the direction of a random ray. The
        norm mode iteratively finds solution that maximize the L2 distance
        from previously discovered solutions.
    solver : string
        The solver to be used.
    solver_options : dict
        Solver option-value pairs to be passed to the solver.
    tee : boolean
        Boolean indicating that the solver output should be displayed.
    seed : int
        Optional integer seed for the numpy random number generator

    Returns
    -------
    solutions
        A list of Solution objects.
        [Solution]
    """
    logger.info("STARTING LP ENUMERATION ANALYSIS")

    assert search_mode in [
        "optimal",
        "random",
        "norm",
    ], 'search mode must be "optimal", "random", or "norm".'
    # TODO: Implement the random and norm objectives. I think it is sufficient
    # to only consider the cb.var_lower variables in the objective for these two
    # cases. The cb.var_upper variables are directly linked to these to diversity
    # in one implies diversity in the other. Diversity in the cb.basic_slack
    # variables doesn't really matter since we only really care about diversity
    # in the original problem and not in the slack space (I think)

    all_variables = aos_utils.get_model_variables(model)
    # else:
    #     binary_variables = ComponentSet()
    #     non_binary_variables = []
    #     for var in variables:
    #         if var.is_binary():
    #             binary_variables.append(var)
    #         else:
    #             non_binary_variables.append(var.name)
    #     if len(non_binary_variables) > 0:
    #         logger.warn(('Warning: The following non-binary variables were included'
    #                'in the variable list and will be ignored:'))
    #         logger.warn(", ".join(non_binary_variables))
    # all_variables = aos_utils.get_model_variables(model, None,
    #                                               include_fixed=True)

    # TODO: Relax this if possible - Should allow for the mixed-binary case
    for var in all_variables:
        assert var.is_continuous(), "Model must be an LP"

    use_appsi = False
    if "appsi" in solver:
        use_appsi = True
        opt = appsi.solvers.Gurobi()
        opt.config.load_solution = False
        opt.config.stream_solver = tee
        opt.update_config.check_for_new_or_removed_constraints = True
        opt.update_config.update_constraints = False
        opt.update_config.check_for_new_or_removed_vars = True
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.update_vars = False
        opt.update_config.update_params = False
        opt.update_config.update_named_expressions = False
        opt.update_config.treat_fixed_vars_as_params = False

        if search_mode == "norm":
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = True
        elif search_mode == "random":
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = False
        else:
            opt.update_config.check_for_new_objective = False
            opt.update_config.update_objective = False
        for parameter, value in solver_options.items():
            opt.gurobi_options[parameter] = value
    else:
        opt = pyo.SolverFactory(solver)
        opt.available()
        for parameter, value in solver_options.items():
            opt.options[parameter] = value
        if solver == "gurobi":
            # Disable gurobi heuristics, which can return
            #   solutions not at a vertex
            opt.options["Heuristics"] = 0.0

    logger.info("Performing initial solve of model.")

    if use_appsi:
        results = opt.solve(model)
        condition = results.termination_condition
        optimal_tc = appsi.base.TerminationCondition.optimal
    else:
        results = opt.solve(model, tee=tee, load_solutions=False)
        condition = results.solver.termination_condition
        optimal_tc = pyo.TerminationCondition.optimal
    if condition != optimal_tc:
        raise Exception(
            (
                "Model could not be solved. LP enumeration analysis "
                "cannot be applied, "
                "TerminationCondition = {}"
            ).format(condition.value)
        )
    if use_appsi:
        results.solution_loader.load_vars(solution_number=0)
    else:
        model.solutions.load_from(results)

    orig_objective = aos_utils.get_active_objective(model)
    orig_objective_value = pyo.value(orig_objective)
    logger.info("Found optimal solution, value = {}.".format(orig_objective_value))

    aos_block = aos_utils._add_aos_block(model, name="_lp_enum")
    aos_utils._add_objective_constraint(
        aos_block, orig_objective, orig_objective_value, rel_opt_gap, abs_opt_gap
    )
    logger.info("Added block {} to the model.".format(aos_block))

    canon_block = shifted_lp.get_shifted_linear_model(model)
    cb = canon_block

    # Set K
    cb.iteration = pyo.Set(pyo.PositiveIntegers)

    # w variables
    cb.basic_lower = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)
    cb.basic_upper = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)
    cb.basic_slack = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)

    # w upper bounds constraints (Eqn (3))
    cb.bound_lower = pyo.Constraint(pyo.Any)
    cb.bound_upper = pyo.Constraint(pyo.Any)
    cb.bound_slack = pyo.Constraint(pyo.Any)

    # non-zero basic variable no-good cut set
    cb.cut_set = pyo.Constraint(pyo.PositiveIntegers)

    # [ (continuous, binary, constraint) ]
    variable_groups = [
        (cb.var_lower, cb.basic_lower, cb.bound_lower),
        (cb.var_upper, cb.basic_upper, cb.bound_upper),
        (cb.slack_vars, cb.basic_slack, cb.bound_slack),
    ]

    solution_number = 1
    solutions = []
    while solution_number <= num_solutions:
        logger.info("Solving Iteration {}: ".format(solution_number), end="")

        if logger.isEnabledFor(logging.DEBUG):
            model.pprint()
        if use_appsi:
            results = opt.solve(model)
            condition = results.termination_condition
        else:
            results = opt.solve(cb, tee=tee, load_solutions=False)
            condition = results.solver.termination_condition

        if condition == optimal_tc:
            if use_appsi:
                results.solution_loader.load_vars(solution_number=0)
            else:
                model.solutions.load_from(results)

            for var, index in cb.var_map.items():
                var.set_value(var.lb + cb.var_lower[index].value)
            sol = solution.Solution(model, all_variables, objective=orig_objective)
            solutions.append(sol)
            orig_objective_value = sol.objective[1]

            if logger.isEnabledFor(logging.INFO):
                logger.info("Solved, objective = {}".format(orig_objective_value))
                for var, index in cb.var_map.items():
                    logger.info(
                        "{} = {}".format(var.name, var.lb + cb.var_lower[index].value)
                    )
            if logger.isEnabledFor(logging.DEBUG):
                model.display()

            if hasattr(cb, "force_out"):
                cb.del_component("force_out")
            if hasattr(cb, "link_in_out"):
                cb.del_component("link_in_out")
            if hasattr(cb, "basic_last_lower"):
                cb.del_component("basic_last_lower")
            if hasattr(cb, "basic_last_upper"):
                cb.del_component("basic_last_upper")
            if hasattr(cb, "basic_last_slack"):
                cb.del_component("basic_last_slack")

            cb.link_in_out = pyo.Constraint(pyo.Any)
            # y variables
            cb.basic_last_lower = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)
            cb.basic_last_upper = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)
            cb.basic_last_slack = pyo.Var(pyo.Any, domain=pyo.Binary, dense=False)
            basic_last_list = [
                cb.basic_last_lower,
                cb.basic_last_upper,
                cb.basic_last_slack,
            ]

            # Number of variables with non-zero values
            num_non_zero = 0
            # This expression is used to ensure that at least one of the non-zero basic
            # variables in the previous solution is selected.
            force_out_expr = -1
            # This expression is used to ensure that at most (# non-zero basic variables)-1
            # binary choice variables can be selected.
            non_zero_basic_expr = 1
            for idx in range(len(variable_groups)):
                continuous_var, binary_var, constraint = variable_groups[idx]
                for var in continuous_var:
                    if continuous_var[var].value > zero_threshold:
                        num_non_zero += 1

                        # Eqn (3): if binary choice variable is not selected, then
                        # continuous variable is zero.
                        constraint[var] = (
                            continuous_var[var]
                            <= continuous_var[var].ub * binary_var[var]
                        )
                        non_zero_basic_expr += binary_var[var]
                        basic_var = basic_last_list[idx][var]
                        force_out_expr += basic_var
                        # Eqn (4): if binary choice variable is selected, then
                        #           basic variable is zero
                        cb.link_in_out[var] = basic_var + binary_var[var] <= 1
            # Eqn (1): at least one of the non-zero basic variables in the
            #   previous solution is selected
            cb.force_out = pyo.Constraint(expr=force_out_expr >= 0)
            # Eqn (2): At most (# non-zero basic variables)-1 binary choice
            # variables can be selected
            cb.cut_set[solution_number] = non_zero_basic_expr <= num_non_zero

            solution_number += 1
        elif (
            condition == pyo.TerminationCondition.infeasibleOrUnbounded
            or condition == pyo.TerminationCondition.infeasible
        ):
            logger.info("Infeasible, all alternative solutions have been found.")
            break
        else:
            logger.info(
                (
                    "Unexpected solver condition. Stopping LP enumeration. "
                    "SolverStatus = {}, TerminationCondition = {}"
                ).format(results.solver.status.value, condition.value)
            )
            break
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("")
            logger.debug("=" * 80)
            logger.debug("")

    model.del_component("aos_block")

    logger.info("COMPLETED LP ENUMERATION ANALYSIS")

    return solutions
