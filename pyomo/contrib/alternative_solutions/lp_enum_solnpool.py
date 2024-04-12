#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pe
from pyomo.contrib.alternative_solutions import (
    aos_utils,
    shifted_lp,
    solution,
    solnpool,
)

#
# A draft enum tool using the gurobi solution pool
#


def enumerate_linear_solutions_soln_pool(
    model,
    num_solutions=10,
    variables="all",
    rel_opt_gap=None,
    abs_opt_gap=None,
    solver_options={},
    tee=False,
):
    """
    Finds alternative optimal solutions a (mixed-integer) linear program using
    Gurobi's solution pool feature.

        Parameters
        ----------
        model : ConcreteModel
            A concrete Pyomo model
        num_solutions : int
            The maximum number of solutions to generate.
        variables: 'all' or a collection of Pyomo _GeneralVarData variables
            The variables for which bounds will be generated. 'all' indicates
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
    opt = pe.SolverFactory("gurobi")
    print("STARTING LP ENUMERATION ANALYSIS USING GUROBI SOLUTION POOL")

    # For now keeping things simple
    # TODO: Relax this
    assert variables == "all"

    opt = pe.SolverFactory("gurobi")
    for parameter, value in solver_options.items():
        opt.options[parameter] = value

    print("Peforming initial solve of model.")
    results = opt.solve(model, tee=tee)
    status = results.solver.status
    condition = results.solver.termination_condition
    if condition != pe.TerminationCondition.optimal:
        raise Exception(
            (
                "Model could not be solve. LP enumeration analysis "
                "cannot be applied, SolverStatus = {}, "
                "TerminationCondition = {}"
            ).format(status.value, condition.value)
        )

    orig_objective = aos_utils.get_active_objective(model)
    orig_objective_value = pe.value(orig_objective)
    print("Found optimal solution, value = {}.".format(orig_objective_value))

    aos_block = aos_utils._add_aos_block(model, name="_lp_enum")
    print("Added block {} to the model.".format(aos_block))
    aos_utils._add_objective_constraint(
        aos_block, orig_objective, orig_objective_value, rel_opt_gap, abs_opt_gap
    )

    cannonical_block = shifted_lp.get_shifted_linear_model(model)
    cb = cannonical_block

    # w variables
    cb.basic_lower = pe.Var(cb.var_lower_index, domain=pe.Binary)
    cb.basic_upper = pe.Var(cb.var_upper_index, domain=pe.Binary)
    cb.basic_slack = pe.Var(cb.slack_index, domain=pe.Binary)

    # w upper bounds constraints
    def bound_lower_rule(m, var_index):
        return (
            m.var_lower[var_index]
            <= m.var_lower[var_index].ub * m.basic_lower[var_index]
        )

    cb.bound_lower = pe.Constraint(cb.var_lower_index, rule=bound_lower_rule)

    def bound_upper_rule(m, var_index):
        return (
            m.var_upper[var_index]
            <= m.var_upper[var_index].ub * m.basic_upper[var_index]
        )

    cb.bound_upper = pe.Constraint(cb.var_upper_index, rule=bound_upper_rule)

    def bound_slack_rule(m, var_index):
        return (
            m.slack_vars[var_index]
            <= m.slack_vars[var_index].ub * m.basic_slack[var_index]
        )

    cb.bound_slack = pe.Constraint(cb.slack_index, rule=bound_slack_rule)
    cb.pprint()
    results = solnpool.gurobi_generate_solutions(cb, num_solutions)

    #     print('Solving Iteration {}: '.format(solution_number), end='')
    #     results = opt.solve(cb, tee=tee)
    #     status = results.solver.status
    #     condition = results.solver.termination_condition
    #     if condition == pe.TerminationCondition.optimal:
    #         for var, index in cb.var_map.items():
    #             var.set_value(var.lb + cb.var_lower[index].value)
    #         sol = solution.Solution(model, all_variables,
    #                                  objective=orig_objective)
    #         solutions.append(sol)
    #         orig_objective_value = sol.objective[1]
    #         print('Solved, objective = {}'.format(orig_objective_value))
    #         for var, index in cb.var_map.items():
    #             print('{} = {}'.format(var.name, var.lb + cb.var_lower[index].value))
    #         if hasattr(cb, 'force_out'):
    #             cb.del_component('force_out')
    #         if hasattr(cb, 'link_in_out'):
    #             cb.del_component('link_in_out')

    #         if hasattr(cb, 'basic_last_lower'):
    #             cb.del_component('basic_last_lower')
    #         if hasattr(cb, 'basic_last_upper'):
    #             cb.del_component('basic_last_upper')
    #         if hasattr(cb, 'basic_last_slack'):
    #             cb.del_component('basic_last_slack')

    #         cb.link_in_out = pe.Constraint(pe.Any)
    #         cb.basic_last_lower = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    #         cb.basic_last_upper = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    #         cb.basic_last_slack = pe.Var(pe.Any, domain=pe.Binary, dense=False)
    #         basic_last_list = [cb.basic_last_lower, cb.basic_last_upper,
    #                            cb.basic_last_slack]

    #         num_non_zero = 0
    #         force_out_expr = -1
    #         non_zero_basic_expr = 1
    #         for idx in range(len(variable_groups)):
    #             continuous_var, binary_var, constraint = variable_groups[idx]
    #             for var in continuous_var:
    #                 if continuous_var[var].value > zero_threshold:
    #                     num_non_zero += 1
    #                     if var not in binary_var:
    #                         binary_var[var]
    #                         constraint[var] = continuous_var[var] <= \
    #                             continuous_var[var].ub * binary_var[var]
    #                     non_zero_basic_expr += binary_var[var]
    #                     basic_var = basic_last_list[idx][var]
    #                     force_out_expr += basic_var
    #                     cb.link_in_out[var] = basic_var + binary_var[var] <= 1

    # aos_block.deactivate()
    # print('COMPLETED LP ENUMERATION ANALYSIS')

    # return solutions
