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
from pyomo.contrib.alternative_solutions import aos_utils
from pyomo.contrib.alternative_solutions import Solution
from pyomo.contrib import appsi


def obbt_analysis(
    model,
    *,
    variables=None,
    rel_opt_gap=None,
    abs_opt_gap=None,
    refine_discrete_bounds=False,
    warmstart=True,
    solver="gurobi",
    solver_options={},
    tee=False,
):
    """
    Calculates the bounds on each variable by solving a series of min and max
    optimization problems where each variable is used as the objective function
    This can be applied to any class of problem supported by the selected
    solver.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model.
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
    refine_discrete_bounds : boolean
        Boolean indicating that new constraints should be added to the
        model at each iteration to tighten the bounds for discrete
        variables.
    warmstart : boolean
        Boolean indicating that the solver should be warmstarted from the
        best previously discovered solution.
    solver : string
        The solver to be used.
    solver_options : dict
        Solver option-value pairs to be passed to the solver.
    tee : boolean
        Boolean indicating that the solver output should be displayed.

    Returns
    -------
    variable_ranges
        A Pyomo ComponentMap containing the bounds for each variable.
        {variable: (lower_bound, upper_bound)}. An exception is raised when
        the solver encountered an issue.
    """
    bounds, solns = obbt_analysis_bounds_and_solutions(
        model,
        variables=variables,
        rel_opt_gap=rel_opt_gap,
        abs_opt_gap=abs_opt_gap,
        refine_discrete_bounds=refine_discrete_bounds,
        warmstart=warmstart,
        solver=solver,
        solver_options=solver_options,
        tee=tee,
    )
    return bounds


def obbt_analysis_bounds_and_solutions(
    model,
    *,
    variables=None,
    rel_opt_gap=None,
    abs_opt_gap=None,
    refine_discrete_bounds=False,
    warmstart=True,
    solver="gurobi",
    solver_options={},
    tee=False,
):
    """
    Calculates the bounds on each variable by solving a series of min and max
    optimization problems where each variable is used as the objective function
    This can be applied to any class of problem supported by the selected
    solver.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model.
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
    refine_discrete_bounds : boolean
        Boolean indicating that new constraints should be added to the
        model at each iteration to tighten the bounds for discrete
        variables.
    warmstart : boolean
        Boolean indicating that the solver should be warmstarted from the
        best previously discovered solution.
    solver : string
        The solver to be used.
    solver_options : dict
        Solver option-value pairs to be passed to the solver.
    tee : boolean
        Boolean indicating that the solver output should be displayed.

    Returns
    -------
    variable_ranges
        A Pyomo ComponentMap containing the bounds for each variable.
        {variable: (lower_bound, upper_bound)}. An exception is raised when
        the solver encountered an issue.
    solutions
        [Solution]
    """

    # TODO - parallelization

    logger.info("STARTING OBBT ANALYSIS")

    if warmstart:
        assert (
            variables == None
        ), "Cannot restrict variable list when warmstart is specified"
    all_variables = aos_utils.get_model_variables(model, include_fixed=False)
    if variables == None:
        variable_list = all_variables
    else:
        variable_list = list(variables)
    if warmstart:
        solutions = pyo.ComponentMap()
        for var in all_variables:
            solutions[var] = []

    num_vars = len(variable_list)
    logger.info(
        "Analyzing {} variables ({} total solves).".format(num_vars, 2 * num_vars)
    )
    orig_objective = aos_utils.get_active_objective(model)

    use_appsi = False
    if "appsi" in solver:
        opt = appsi.solvers.Gurobi()
        for parameter, value in solver_options.items():
            opt.gurobi_options[parameter] = value
        opt.config.stream_solver = tee
        opt.config.load_solution = False
        results = opt.solve(model)
        condition = results.termination_condition
        optimal_tc = appsi.base.TerminationCondition.optimal
        infeas_or_unbdd_tc = appsi.base.TerminationCondition.infeasibleOrUnbounded
        unbdd_tc = appsi.base.TerminationCondition.unbounded
        use_appsi = True
    else:
        opt = pyo.SolverFactory(solver)
        opt.available()
        for parameter, value in solver_options.items():
            opt.options[parameter] = value
        try:
            results = opt.solve(
                model, warmstart=warmstart, tee=tee, load_solutions=False
            )
        except ValueError:
            # An exception occurs if the solver does not recognize the warmstart option
            results = opt.solve(model, tee=tee, load_solutions=False)
        condition = results.solver.termination_condition
        optimal_tc = pyo.TerminationCondition.optimal
        infeas_or_unbdd_tc = pyo.TerminationCondition.infeasibleOrUnbounded
        unbdd_tc = pyo.TerminationCondition.unbounded
    logger.info("Performing initial solve of model.")

    if condition != optimal_tc:
        raise RuntimeError(
            ("OBBT cannot be applied, " "TerminationCondition = {}").format(
                condition.value
            )
        )
    if use_appsi:
        results.solution_loader.load_vars(solution_number=0)
    else:
        model.solutions.load_from(results)
    if warmstart:
        _add_solution(solutions)
    orig_objective_value = pyo.value(orig_objective)
    logger.info("Found optimal solution, value = {}.".format(orig_objective_value))
    aos_block = aos_utils._add_aos_block(model, name="_obbt")
    # placeholder for objective
    aos_block.var_objective = pyo.Objective(expr=0)
    logger.info("Added block {} to the model.".format(aos_block))
    obj_constraints = aos_utils._add_objective_constraint(
        aos_block, orig_objective, orig_objective_value, rel_opt_gap, abs_opt_gap
    )
    if refine_discrete_bounds:
        aos_block.bound_constraints = pyo.ConstraintList()
    new_constraint = False
    if len(obj_constraints) > 0:
        new_constraint = True
    orig_objective.deactivate()

    if use_appsi:
        opt.update_config.check_for_new_or_removed_constraints = new_constraint
        opt.update_config.check_for_new_or_removed_vars = False
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.check_for_new_objective = True
        opt.update_config.update_constraints = False
        opt.update_config.update_vars = False
        opt.update_config.update_params = False
        opt.update_config.update_named_expressions = False
        opt.update_config.update_objective = True
        opt.update_config.treat_fixed_vars_as_params = False

    variable_bounds = pyo.ComponentMap()
    solns = [Solution(model, all_variables, objective=orig_objective)]

    senses = [(pyo.minimize, "LB"), (pyo.maximize, "UB")]

    iteration = 1
    total_iterations = len(senses) * num_vars
    for idx in range(len(senses)):
        sense = senses[idx][0]
        bound_dir = senses[idx][1]

        for var in variable_list:
            if idx == 0:
                variable_bounds[var] = [None, None]

            aos_block.var_objective.expr = var
            aos_block.var_objective.sense = sense

            if warmstart:
                _update_values(var, bound_dir, solutions)

            if use_appsi:
                opt.update_config.check_for_new_or_removed_constraints = new_constraint
            if use_appsi:
                opt.config.stream_solver = tee
                results = opt.solve(model)
                condition = results.termination_condition
            else:
                try:
                    results = opt.solve(
                        model, warmstart=warmstart, tee=tee, load_solutions=False
                    )
                except ValueError:
                    # An exception occurs if the solver does not recognize the warmstart option
                    results = opt.solve(model, tee=tee, load_solutions=False)
                condition = results.solver.termination_condition
            new_constraint = False

            if condition == optimal_tc:
                if use_appsi:
                    results.solution_loader.load_vars(solution_number=0)
                else:
                    model.solutions.load_from(results)
                solns.append(Solution(model, all_variables, objective=orig_objective))

                if warmstart:
                    _add_solution(solutions)
                obj_val = pyo.value(var)
                variable_bounds[var][idx] = obj_val

                if refine_discrete_bounds and not var.is_continuous():
                    if sense == pyo.minimize and var.lb < obj_val:
                        aos_block.bound_constraints.add(var >= obj_val)
                        new_constraint = True

                    if sense == pyo.maximize and var.ub > obj_val:
                        aos_block.bound_constraints.add(var <= obj_val)
                        new_constraint = True

            # An infeasibleOrUnbounded status code will imply the problem is
            # unbounded since feasibility has been established previously
            elif condition == infeas_or_unbdd_tc or condition == unbdd_tc:
                if sense == pyo.minimize:
                    variable_bounds[var][idx] = float("-inf")
                else:
                    variable_bounds[var][idx] = float("inf")
            else:  # pragma: no cover
                logger.warn(
                    (
                        "Unexpected condition for the variable {} {} problem."
                        "TerminationCondition = {}"
                    ).format(var.name, bound_dir, condition.value)
                )

            var_value = variable_bounds[var][idx]
            logger.info(
                "Iteration {}/{}: {}_{} = {}".format(
                    iteration, total_iterations, var.name, bound_dir, var_value
                )
            )

            if idx == 1:
                variable_bounds[var] = tuple(variable_bounds[var])

            iteration += 1

    aos_block.deactivate()
    orig_objective.activate()

    logger.info("COMPLETED OBBT ANALYSIS")

    return variable_bounds, solns


def _add_solution(solutions):
    """Add the current variable values to the solution list."""
    for var in solutions:
        solutions[var].append(pyo.value(var))


def _update_values(var, bound_dir, solutions):
    """
    Set the values of all variables to the best solution seen previously for
    the current objective function.
    """
    if bound_dir == "LB":
        value = min(solutions[var])
    else:
        value = max(solutions[var])
    idx = solutions[var].index(value)
    for variable in solutions:
        variable.set_value(solutions[variable][idx])
