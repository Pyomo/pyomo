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

from pyomo.common.dependencies import attempt_import

gurobipy, gurobi_available = attempt_import("gurobipy")

import pyomo.environ as pyo
import pyomo.common.errors
from pyomo.contrib.alternative_solutions import aos_utils, shifted_lp, solution
from pyomo.contrib import appsi


class NoGoodCutGenerator:
    def __init__(
        self,
        model,
        variable_groups,
        zero_threshold,
        orig_model,
        all_variables,
        orig_objective,
        num_solutions,
    ):
        self.model = model
        self.zero_threshold = zero_threshold
        self.variable_groups = variable_groups
        self.variables = aos_utils.get_model_variables(model)
        self.orig_model = orig_model
        self.all_variables = all_variables
        self.orig_objective = orig_objective
        self.solutions = []
        self.num_solutions = num_solutions

    def cut_generator_callback(self, cb_m, cb_opt, cb_where):
        if cb_where == gurobipy.GRB.Callback.MIPSOL:
            cb_opt.cbGetSolution(vars=self.variables)
            logger.info("***FOUND SOLUTION***")

            for var, index in self.model.var_map.items():
                var.set_value(var.lb + self.model.var_lower[index].value)
            sol = solution.Solution(
                self.orig_model, self.all_variables, objective=self.orig_objective
            )
            self.solutions.append(sol)

            if len(self.solutions) >= self.num_solutions:
                cb_opt._solver_model.terminate()
            num_non_zero = 0
            non_zero_basic_expr = 1
            for idx in range(len(self.variable_groups)):
                continuous_var, binary_var = self.variable_groups[idx]
                for var in continuous_var:
                    if continuous_var[var].value > self.zero_threshold:
                        num_non_zero += 1
                        non_zero_basic_expr += binary_var[var]
            # TODO: JLG - If we want to add the mixed binary case, I think we
            # need to do it here. Essentially we would want to continue to
            # build up the num_non_zero as follows
            # for binary in binary_vars:
            # if binary.value > 0.5:
            # num_non_zero += 1 - binary
            # else:
            # num_non_zero += binary
            new_con = self.model.cl.add(non_zero_basic_expr <= num_non_zero)
            cb_opt.cbLazy(new_con)


def enumerate_linear_solutions_soln_pool(
    model,
    num_solutions=10,
    rel_opt_gap=None,
    abs_opt_gap=None,
    zero_threshold=1e-5,
    solver_options={},
    tee=False,
):
    """
    Finds alternative optimal solutions for a (mixed-binary) linear program
    using Gurobi's solution pool feature.

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
    zero_threshold: float
        The threshold for which a continuous variables' value is considered
        to be equal to zero.
    solver_options : dict
        Solver option-value pairs to be passed to the solver.
    tee : boolean
        Boolean indicating that the solver output should be displayed.

    Returns
    -------
    solutions
        A list of Solution objects.
        [Solution]
    """
    logger.info("STARTING LP ENUMERATION ANALYSIS USING GUROBI SOLUTION POOL")
    #
    # Setup gurobi
    #
    if not gurobi_available:
        raise pyomo.common.errors.ApplicationError(f"Solver (gurobi) not available")

    all_variables = aos_utils.get_model_variables(model)
    for var in all_variables:
        if var.is_integer():
            raise pyomo.common.errors.ApplicationError(
                f"The enumerate_linear_solutions_soln_pool() function cannot be used with models that contain discrete variables"
            )

    opt = pyo.SolverFactory("gurobi")
    if not opt.available(exception_flag=False):
        raise ValueError(solver + " is not available")
    for parameter, value in solver_options.items():
        opt.options[parameter] = value

    logger.info("Performing initial solve of model.")
    results = opt.solve(model, tee=tee)
    status = results.solver.status
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        raise Exception(
            (
                "Model could not be solve. LP enumeration analysis "
                "cannot be applied, SolverStatus = {}, "
                "TerminationCondition = {}"
            ).format(status.value, condition.value)
        )

    orig_objective = aos_utils.get_active_objective(model)
    orig_objective_value = pyo.value(orig_objective)
    logger.info("Found optimal solution, value = {}.".format(orig_objective_value))

    aos_block = aos_utils._add_aos_block(model, name="_lp_enum")
    logger.info("Added block {} to the model.".format(aos_block))
    aos_utils._add_objective_constraint(
        aos_block, orig_objective, orig_objective_value, rel_opt_gap, abs_opt_gap
    )

    canonical_block = shifted_lp.get_shifted_linear_model(model)
    cb = canonical_block
    lower_index = list(cb.var_lower.keys())
    upper_index = list(cb.var_upper.keys())

    # w variables
    cb.basic_lower = pyo.Var(lower_index, domain=pyo.Binary)
    cb.basic_upper = pyo.Var(upper_index, domain=pyo.Binary)
    cb.basic_slack = pyo.Var(cb.slack_index, domain=pyo.Binary)

    # w upper bounds constraints
    def bound_lower_rule(m, var_index):
        return (
            m.var_lower[var_index]
            <= m.var_lower[var_index].ub * m.basic_lower[var_index]
        )

    cb.bound_lower = pyo.Constraint(lower_index, rule=bound_lower_rule)

    def bound_upper_rule(m, var_index):
        return (
            m.var_upper[var_index]
            <= m.var_upper[var_index].ub * m.basic_upper[var_index]
        )

    cb.bound_upper = pyo.Constraint(upper_index, rule=bound_upper_rule)

    def bound_slack_rule(m, var_index):
        return (
            m.slack_vars[var_index]
            <= m.slack_vars[var_index].ub * m.basic_slack[var_index]
        )

    cb.bound_slack = pyo.Constraint(cb.slack_index, rule=bound_slack_rule)

    cb.cl = pyo.ConstraintList()

    # TODO: If we go the mixed binary route we also want to list the binary variables
    variable_groups = [
        (cb.var_lower, cb.basic_lower),
        (cb.var_upper, cb.basic_upper),
        (cb.slack_vars, cb.basic_slack),
    ]
    cut_generator = NoGoodCutGenerator(
        cb,
        variable_groups,
        zero_threshold,
        model,
        all_variables,
        orig_objective,
        num_solutions,
    )

    opt = appsi.solvers.Gurobi()
    for parameter, value in solver_options.items():
        opt.gurobi_options[parameter] = value
    opt.config.stream_solver = True
    opt.config.load_solution = False
    opt.gurobi_options["LazyConstraints"] = 1
    opt.set_instance(cb)
    opt.set_callback(cut_generator.cut_generator_callback)
    opt.solve(cb)

    aos_block.deactivate()
    logger.info("COMPLETED LP ENUMERATION ANALYSIS")

    return cut_generator.solutions
