#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:33:42 2022
@author: pmlpm
"""

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers

scip_available = bool(check_available_solvers('scip'))

import random

# ******************************************************************************
# ******************************************************************************

# carry out optimisations


def optimise(
    problem: pyo.ConcreteModel,
    solver_timelimit,
    solver_rel_mip_gap,
    solver_abs_mip_gap,
    print_solver_output: bool = False,
):
    # config

    options_dict_format = {
        'limits/time': solver_timelimit,
        'limits/gap': solver_rel_mip_gap,
        'limits/absgap': solver_abs_mip_gap,
    }

    opt = pyo.SolverFactory('scip')

    for key, value in options_dict_format.items():
        opt.options[key] = value

    # solve

    results = opt.solve(problem, tee=print_solver_output)

    # return

    return results, opt


# ******************************************************************************
# ******************************************************************************


def problem_lp_optimal():
    model = pyo.ConcreteModel('lp_optimal')

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

    return model


def problem_lp_infeasible():
    model = pyo.ConcreteModel('lp_infeasible')

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] <= -1)

    return model


def problem_lp_unbounded():
    model = pyo.ConcreteModel('lp_unbounded')

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2], sense=pyo.maximize)

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

    return model


def problem_milp_optimal():
    model = pyo.ConcreteModel('milp_optimal')

    model.x = pyo.Var([1, 2], domain=pyo.Binary)

    model.OBJ = pyo.Objective(expr=2.15 * model.x[1] + 3.8 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

    return model


def problem_milp_infeasible():
    model = pyo.ConcreteModel('milp_infeasible')

    model.x = pyo.Var([1, 2], domain=pyo.Binary)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] <= -1)

    return model


def problem_milp_unbounded():
    model = pyo.ConcreteModel('milp_unbounded')

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.y = pyo.Var(domain=pyo.Binary)

    model.OBJ = pyo.Objective(
        expr=2 * model.x[1] + 3 * model.x[2] + model.y, sense=pyo.maximize
    )

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

    return model


def problem_milp_feasible():
    model = pyo.ConcreteModel('milp_feasible')

    random.seed(6254)

    # a knapsack-type problem

    number_binary_variables = 20  # may need to be tweaked depending on specs

    model.Y = pyo.RangeSet(number_binary_variables)

    model.y = pyo.Var(model.Y, domain=pyo.Binary)

    model.OBJ = pyo.Objective(
        expr=sum(model.y[j] * random.random() for j in model.Y), sense=pyo.maximize
    )

    model.Constraint1 = pyo.Constraint(
        expr=sum(model.y[j] * random.random() for j in model.Y)
        <= round(number_binary_variables / 5)
    )

    def rule_c1(m, i):
        return (
            sum(
                model.y[j] * (random.random() - 0.5)
                for j in model.Y
                if j != i
                if random.randint(0, 1)
            )
            <= round(number_binary_variables / 5) * model.y[i]
        )

    model.constr_c1 = pyo.Constraint(model.Y, rule=rule_c1)

    return model


# ******************************************************************************
# ******************************************************************************


@unittest.skipIf(not scip_available, "SCIP solver is not available.")
def test_scip_some_more():
    # list of problems

    list_concrete_models = [
        problem_lp_unbounded(),
        problem_lp_infeasible(),
        problem_lp_optimal(),
        problem_milp_unbounded(),
        problem_milp_infeasible(),
        problem_milp_optimal(),
        problem_milp_feasible(),  # may reach optimality depending on the budget
    ]

    list_extra_data_expected = [
        (),  # problem_lp_unbounded(),
        (),  # problem_lp_infeasible(),
        ('Time', 'Gap', 'Primal bound', 'Dual bound'),  # problem_lp_optimal(),
        (),  # problem_milp_unbounded(),
        (),  # problem_milp_infeasible(),
        ('Time', 'Gap', 'Primal bound', 'Dual bound'),  # problem_milp_optimal(),
        ('Time', 'Gap', 'Primal bound', 'Dual bound'),  # problem_milp_feasible()
    ]

    # **************************************************************************
    # **************************************************************************

    # solver settings

    solver_timelimit = 1

    solver_abs_mip_gap = 0

    solver_rel_mip_gap = 1e-6

    # **************************************************************************
    # **************************************************************************

    for problem_index, problem in enumerate(list_concrete_models):
        print('******************************')
        print('******************************')

        print(problem.name)

        print('******************************')
        print('******************************')

        results, opt = optimise(
            problem,
            solver_timelimit,
            solver_rel_mip_gap,
            solver_abs_mip_gap,
            print_solver_output=True,
        )

        print(results)

        # check the version

        executable = opt._command.cmd[0]

        version = opt._known_versions[executable]

        if version < (8, 0, 0, 0):
            # if older and untested, skip tests

            continue

        # for each new attribute expected

        for log_file_attr in list_extra_data_expected[problem_index]:
            # check that it is part of the results object

            assert log_file_attr in results['Solver'][0]


# ******************************************************************************
# ******************************************************************************

# test_scip_some_more() # uncomment to run individually
