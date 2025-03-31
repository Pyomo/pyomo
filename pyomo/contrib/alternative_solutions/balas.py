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
from pyomo.common.collections import ComponentSet
from pyomo.contrib.alternative_solutions import Solution
import pyomo.contrib.alternative_solutions.aos_utils as aos_utils


def enumerate_binary_solutions(
    model,
    *,
    num_solutions=10,
    variables=None,
    rel_opt_gap=None,
    abs_opt_gap=None,
    search_mode="optimal",
    solver="gurobi",
    solver_options={},
    tee=False,
    seed=None,
):
    """
    Finds alternative optimal solutions for a binary problem using no-good
    cuts.

    This function implements a no-good cuts technique inheriting from
    Balas's work on Canonical Cuts [BJ72]_.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model
    num_solutions : int
        The maximum number of solutions to generate.
    variables: None or a collection of Pyomo _GeneralVarData variables
        The variables for which bounds will be generated. None indicates
        that all variables will be included. Alternatively, a collection of
        _GenereralVarData variables can be provided.
    rel_opt_gap : float or None
        The relative optimality gap for the original objective for which
        variable bounds will be found. None indicates that a relative gap
        constraint will not be added to the model.
    abs_opt_gap : float or None
        The absolute optimality gap for the original objective for which
        variable bounds will be found. None indicates that an absolute gap
        constraint will not be added to the model.
    search_mode : 'optimal', 'random', or 'hamming'
        Indicates the mode that is used to generate alternative solutions.
        The optimal mode finds the next best solution. The random mode
        finds an alternative solution in the direction of a random ray. The
        hamming mode iteratively finds solution that maximize the hamming
        distance from previously discovered solutions.
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
    logger.info("STARTING NO-GOOD CUT ANALYSIS")

    assert search_mode in [
        "optimal",
        "random",
        "hamming",
    ], 'search mode must be "optimal", "random", or "hamming".'

    if seed is not None:
        aos_utils._set_numpy_rng(seed)

    all_variables = aos_utils.get_model_variables(model, include_fixed=True)
    if variables == None:
        binary_variables = [
            var for var in all_variables if var.is_binary() and not var.is_fixed()
        ]
        logger.debug(
            "Analysis using %d binary variables: %s"
            % (len(binary_variables), " ".join(var.name for var in binary_variables))
        )
    else:
        binary_variables = ComponentSet()
        non_binary_variables = []
        for var in variables:
            if var.is_binary():
                binary_variables.add(var)
            else:  # pragma: no cover
                non_binary_variables.append(var.name)
        if len(non_binary_variables) > 0:
            logger.warn(
                (
                    "Warning: The following non-binary variables were included"
                    "in the variable list and will be ignored:"
                )
            )
            logger.warn(", ".join(non_binary_variables))

    orig_objective = aos_utils.get_active_objective(model)

    if len(binary_variables) == 0:
        logger.warn("No binary variables found!")

    #
    # Setup solver
    #
    opt = pyo.SolverFactory(solver)
    opt.available()
    for parameter, value in solver_options.items():
        opt.options[parameter] = value
    #
    # Appsi-specific configurations
    #
    use_appsi = False
    if "appsi" in solver:
        use_appsi = True
        opt.update_config.update_constraints = False
        opt.update_config.check_for_new_or_removed_constraints = True
        opt.update_config.check_for_new_or_removed_vars = False
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.update_vars = False
        opt.update_config.update_params = False
        opt.update_config.update_named_expressions = False
        opt.update_config.treat_fixed_vars_as_params = False

        if search_mode == "hamming":
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = True
        elif search_mode == "random":
            opt.update_config.check_for_new_objective = True
            opt.update_config.update_objective = False
        else:
            opt.update_config.check_for_new_objective = False
            opt.update_config.update_objective = False

    #
    # Initial solve of the model
    #
    logger.info("Performing initial solve of model.")
    results = opt.solve(model, tee=tee, load_solutions=False)
    status = results.solver.status
    if not pyo.check_optimal_termination(results):
        condition = results.solver.termination_condition
        raise Exception(
            (
                "No-good cut analysis cannot be applied, "
                "SolverStatus = {}, "
                "TerminationCondition = {}"
            ).format(status.value, condition.value)
        )

    model.solutions.load_from(results)
    orig_objective_value = pyo.value(orig_objective)
    logger.info("Found optimal solution, value = {}.".format(orig_objective_value))
    solutions = [Solution(model, all_variables, objective=orig_objective)]
    #
    # Return just this solution if there are no binary variables
    #
    if len(binary_variables) == 0:
        return solutions

    aos_block = aos_utils._add_aos_block(model, name="_balas")
    logger.info("Added block {} to the model.".format(aos_block))
    aos_block.no_good_cuts = pyo.ConstraintList()
    aos_utils._add_objective_constraint(
        aos_block, orig_objective, orig_objective_value, rel_opt_gap, abs_opt_gap
    )

    if search_mode in ["random", "hamming"]:
        orig_objective.deactivate()

    solution_number = 2
    while solution_number <= num_solutions:

        expr = 0
        for var in binary_variables:
            if var.value > 0.5:
                expr += 1 - var
            else:
                expr += var

        aos_block.no_good_cuts.add(expr=expr >= 1)

        if search_mode == "hamming":
            if hasattr(aos_block, "hamming_objective"):
                aos_block.hamming_objective.expr += expr
                if use_appsi and opt.update_config.check_for_new_objective:
                    opt.update_config.check_for_new_objective = False
            else:
                aos_block.hamming_objective = pyo.Objective(
                    expr=expr, sense=pyo.maximize
                )

        elif search_mode == "random":
            if hasattr(aos_block, "random_objective"):
                aos_block.del_component("random_objective")
            vector = aos_utils._get_random_direction(len(binary_variables))
            idx = 0
            expr = 0
            for var in binary_variables:
                expr += vector[idx] * var
                idx += 1
            aos_block.random_objective = pyo.Objective(expr=expr, sense=pyo.maximize)

        results = opt.solve(model, tee=tee, load_solutions=False)
        status = results.solver.status
        condition = results.solver.termination_condition
        if pyo.check_optimal_termination(results):
            model.solutions.load_from(results)
            orig_obj_value = pyo.value(orig_objective)
            logger.info(
                "Iteration {}: objective = {}".format(solution_number, orig_obj_value)
            )
            solutions.append(Solution(model, all_variables, objective=orig_objective))
            solution_number += 1
        elif (
            condition == pyo.TerminationCondition.infeasibleOrUnbounded
            or condition == pyo.TerminationCondition.infeasible
        ):
            logger.info(
                "Iteration {}: Infeasible, no additional binary solutions.".format(
                    solution_number
                )
            )
            break
        else:  # pragma: no cover
            logger.info(
                (
                    "Iteration {}: Unexpected condition, SolverStatus = {}, "
                    "TerminationCondition = {}"
                ).format(solution_number, status.value, condition.value)
            )
            break

    aos_block.deactivate()
    orig_objective.activate()

    logger.info("COMPLETED NO-GOOD CUT ANALYSIS")

    return solutions
