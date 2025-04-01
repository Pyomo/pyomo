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

from itertools import product
from math import ceil, floor
from collections import Counter

from pyomo.common.dependencies import numpy as np

import pyomo.environ as pyo

"""
This script has collection of test cases that can be used to enumerate solutions.
That is, simple problems where the alternative solutions can be found manually.
"""


def _is_satisfied(constraint, feasibility_tol=1e-6):
    value = pyo.value(constraint.body)
    if constraint.has_lb() and value < constraint.lb - feasibility_tol:
        return False
    if constraint.has_ub() and value > constraint.ub + feasibility_tol:
        return False
    return True


def get_2d_diamond_problem(discrete_x=False, discrete_y=False):
    """
    Simple 2d problem where the feasible is diamond-shaped.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Integers if discrete_x else pyo.Reals, bounds=(-10, 10))
    m.y = pyo.Var(within=pyo.Integers if discrete_y else pyo.Reals, bounds=(-10, 10))

    m.o = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)

    m.c1 = pyo.Constraint(expr=-4 / 5 * m.x - 4 <= m.y)
    m.c2 = pyo.Constraint(expr=5 / 9 * m.x - 5 <= m.y)
    m.c3 = pyo.Constraint(expr=2 / 9 * m.x + 2 >= m.y)
    m.c4 = pyo.Constraint(expr=-1 / 2 * m.x + 3 >= m.y)

    # Continuous exteme points and bounds
    m.extreme_points = {
        (0.737704918, -4.590163934),
        (-5.869565217, 0.695652174),
        (1.384615385, 2.307692308),
        (7.578947368, -0.789473684),
    }

    m.continuous_bounds = pyo.ComponentMap()
    m.continuous_bounds[m.x] = (-5.869565217, 7.578947368)
    m.continuous_bounds[m.y] = (-4.590163934, 2.307692308)

    # Continuous exteme points and bounds for the case where an objective
    # constraint is added within a 100% relative gap of optimality or an
    # absolute gap of 6.789473684

    m.extreme_points_cut = {
        (45 / 14, -45 / 14),
        (-18 / 11, 18 / 11),
        (1.384615385, 2.307692308),
        (7.578947368, -0.789473684),
    }

    m.continuous_bounds_cut = pyo.ComponentMap()
    m.continuous_bounds_cut[m.x] = (-18 / 11, 7.578947368)
    m.continuous_bounds_cut[m.y] = (-45 / 14, 2.307692308)

    # Discrete feasible solutions and bounds
    feasible_sols = []
    x_lower_bound = None
    x_upper_bound = None
    y_lower_bound = None
    y_upper_bound = None

    x_lower = ceil(m.continuous_bounds[m.x][0])
    x_upper = floor(m.continuous_bounds[m.x][1])
    y_lower = ceil(m.continuous_bounds[m.y][0])
    y_upper = floor(m.continuous_bounds[m.y][1])
    cons = [m.c1, m.c2, m.c3, m.c4]
    for x_value in range(x_lower, x_upper + 1):
        for y_value in range(y_lower, y_upper + 1):
            m.x.set_value(x_value)
            m.y.set_value(y_value)
            is_feasible = True
            for con in cons:
                if not _is_satisfied(con):
                    is_feasible = False
                    break
            if is_feasible:
                if x_lower_bound is None or x_value < x_lower_bound:
                    x_lower_bound = x_value
                if x_upper_bound is None or x_value > x_upper_bound:
                    x_upper_bound = x_value
                if y_lower_bound is None or y_value < y_lower_bound:
                    y_lower_bound = y_value
                if y_upper_bound is None or y_value > y_upper_bound:
                    y_upper_bound = y_value
                feasible_sols.append(((x_value, y_value), x_value + y_value))
    m.discrete_feasible = sorted(feasible_sols, key=lambda sol: sol[1], reverse=True)
    m.discrete_bounds = pyo.ComponentMap()
    m.discrete_bounds[m.x] = (x_lower_bound, x_upper_bound)
    m.discrete_bounds[m.y] = (y_lower_bound, y_upper_bound)

    return m


def get_3d_polyhedron_problem():
    """
    Simple 3d polyhedron that is expressed using all types of linear constraints
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1, 2], within=pyo.Reals)
    m.x[0].setlb(-1)
    m.x[0].setub(1)
    m.x[1].setlb(-2)
    m.x[1].setub(2)
    m.x[2].setlb(1)
    m.x[2].setub(2)

    def _constraint_switch_rule(m, i):
        if i == 0:
            return m.x[0] + m.x[1] <= 2
        elif i == 1:
            return -m.x[0] + m.x[1] <= 2
        elif i == 2:
            return m.x[0] + m.x[1] >= -2
        elif i == 3:
            return -m.x[0] + m.x[1] >= -2
        elif i == 4:
            return m.x[0] + m.x[1] + m.x[2] == 4

    m.c = pyo.Constraint([i for i in range(5)], rule=_constraint_switch_rule)

    m.o = pyo.Objective(expr=m.x[0] + m.x[2], sense=pyo.maximize)
    return m


def get_2d_unbounded_problem():
    """
    Simple 2d problem where the feasible region is unbounded, but the problem
    has an optimal solution.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Reals)
    m.y = pyo.Var(within=pyo.Reals)

    m.o = pyo.Objective(expr=m.y - m.x)

    m.c1 = pyo.Constraint(expr=m.x <= 4)
    m.c2 = pyo.Constraint(expr=m.y >= 2)

    m.extreme_points = {(4, 2)}

    m.continuous_bounds = pyo.ComponentMap()
    m.continuous_bounds[m.x] = (float("-inf"), 4)
    m.continuous_bounds[m.y] = (2, float("inf"))
    return m


def get_2d_degenerate_lp():
    """
    Simple 2d problem that includes a redundant constraint such that three
    constraints are active at optimality.
    """
    m = pyo.ConcreteModel()

    m.x = pyo.Var(within=pyo.Reals, bounds=(-1, 3))
    m.y = pyo.Var(within=pyo.Reals, bounds=(-3, 2))

    m.obj = pyo.Objective(expr=m.x + 2 * m.y, sense=pyo.maximize)

    m.con1 = pyo.Constraint(expr=m.x + m.y <= 3)
    m.con2 = pyo.Constraint(expr=m.x + 2 * m.y <= 5)
    m.con3 = pyo.Constraint(expr=m.x + m.y >= -1)

    return m


def get_triangle_ip():
    """
    Simple 2d discrete problem where the feasible region looks like a 90-45-45
    right triangle and the optimal solutions fall along the hypotenuse, where
    x + y == 5.  Alternative near-optimal have integer objective values from 0 to 4.
    """
    var_max = 5
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, var_max))
    m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, var_max))

    m.o = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
    m.c = pyo.Constraint(expr=m.x + m.y <= var_max)

    #
    # Enumerate all feasible solutions
    #
    feasible_sols = []
    for i in range(var_max + 1):
        for j in range(var_max + 1):
            if i + j <= var_max:
                feasible_sols.append(((i, j), i + j))
    feasible_sols = sorted(feasible_sols, key=lambda sol: sol[1], reverse=True)
    m.feasible_sols = feasible_sols
    #
    # Count of solutions from best to worst
    #
    m.num_ranked_solns = [6, 5, 4, 3, 2, 1]

    return m


def get_implied_bound_ip():
    """
    2d discrete problem where the bounds of z are impled by x and y. This
    facilitate testing cases where the impled bounds are tighter than the
    given bounds for the variable.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 5))
    m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 5))
    m.z = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 5))

    m.o = pyo.Objective(expr=m.x + m.z)

    m.c1 = pyo.Constraint(expr=m.x + m.y == 3)
    m.c2 = pyo.Constraint(expr=m.x + m.y + m.z <= 5)
    m.c3 = pyo.Constraint(expr=m.x + m.y + m.z >= 4)

    m.var_bounds = pyo.ComponentMap()
    m.var_bounds[m.x] = (0, 3)
    m.var_bounds[m.y] = (0, 3)
    m.var_bounds[m.z] = (1, 2)

    return m


def get_aos_test_knapsack(
    var_max, weights, values, capacity=None, capacity_fraction=1.0
):
    """
    Creates a knapsack problem, given arrays of weights and values, and
    returns all feasible solutions. The capacity represents the percent of the
    total max weight that can be selected (sum weights * var_max). The var_max
    parameter sets the upper bound on all variables, the max number of times
    they can be selected.
    """
    assert len(weights) == len(values), "weights and values must be the same length."
    assert (
        0 <= capacity_fraction and capacity_fraction <= 1
    ), "capacity_fraction must be between 0 and 1."

    num_vars = len(weights)
    if capacity is None:
        capacity = sum(weights) * var_max * capacity_fraction

    m = pyo.ConcreteModel()
    m.i = pyo.RangeSet(0, num_vars - 1)

    if var_max == 1:
        m.x = pyo.Var(m.i, within=pyo.Binary)
    else:
        m.x = pyo.Var(m.i, within=pyo.NonNegativeIntegers, bounds=(0, var_max))

    m.o = pyo.Objective(expr=sum(values[i] * m.x[i] for i in m.i), sense=pyo.maximize)

    m.c = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in m.i) <= capacity)

    var_domain = range(var_max + 1)
    all_combos = product(var_domain, repeat=num_vars)

    feasible_sols = []
    for sol in all_combos:
        if np.dot(sol, weights) <= capacity:
            feasible_sols.append((sol, np.dot(sol, values)))
    feasible_sols = sorted(feasible_sols, key=lambda sol: sol[1], reverse=True)
    m.ranked_solution_values = list(sorted([v for x, v in feasible_sols], reverse=True))
    m.num_ranked_solns = list(Counter([v for x, v in feasible_sols]).values())
    return m


def get_pentagonal_lp():
    """
    Pentagonal LP
    """
    var_max = 5
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Reals, bounds=(0, 2 * var_max))
    m.y = pyo.Var(within=pyo.Reals, bounds=(0, 2 * var_max))
    m.z = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 2 * var_max))
    m.o = pyo.Objective(expr=m.z, sense=pyo.minimize)

    base_points = np.array(
        [
            [var_max, 2 * var_max, 0],
            [2 * var_max, var_max, 0],
            [3.0 * var_max / 2.0, 0, 0],
            [var_max / 2.0, 0, 0],
            [0, var_max, 0],
        ]
    )
    apex_point = np.array([var_max, var_max, var_max])

    m.c = pyo.ConstraintList()
    for i in range(5):
        vec_1 = base_points[i] - apex_point
        vec_2 = base_points[(i + 1) % var_max] - base_points[i]
        n = np.cross(vec_1, vec_2)
        m.c.add(
            n[0] * (m.x - apex_point[0])
            + n[1] * (m.y - apex_point[1])
            + n[2] * (m.z - apex_point[2])
            >= 0
        )

    return m


def get_pentagonal_pyramid_mip():
    """
    Pentagonal pyramid with integer coordinates in the first two dimensions and
    a third continuous dimension.
    """
    var_max = 5
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Integers, bounds=(-var_max, var_max))
    m.y = pyo.Var(within=pyo.Integers, bounds=(-var_max, var_max))
    m.z = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, var_max))
    m.o = pyo.Objective(expr=m.z, sense=pyo.maximize)

    base_points = np.array(
        [
            [0, var_max, 0],
            [var_max, 0, 0],
            [var_max / 2.0, -var_max, 0],
            [-var_max / 2.0, -var_max, 0],
            [-var_max, 0, 0],
        ]
    )
    apex_point = np.array([0, 0, var_max])

    m.c = pyo.ConstraintList()
    for i in range(5):
        vec_1 = base_points[i] - apex_point
        vec_2 = base_points[(i + 1) % var_max] - base_points[i]
        n = np.cross(vec_1, vec_2)
        m.c.add(
            n[0] * (m.x - apex_point[0])
            + n[1] * (m.y - apex_point[1])
            + n[2] * (m.z - apex_point[2])
            >= 0
        )
    #
    # Count of solutions from best to worst
    #
    m.num_ranked_solns = [1, 4, 2, 8, 2, 12, 4, 16, 4, 20]
    return m


def get_indexed_pentagonal_pyramid_mip():
    """
    Pentagonal pyramid with integer coordinates in the first two dimensions and
    a third continuous dimension.
    """
    var_max = 5
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], within=pyo.Integers, bounds=(-var_max, var_max))
    m.z = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, var_max))
    m.o = pyo.Objective(expr=m.z, sense=pyo.maximize)
    base_points = np.array(
        [
            [0, var_max, 0],
            [var_max, 0, 0],
            [var_max / 2.0, -var_max, 0],
            [-var_max / 2.0, -var_max, 0],
            [-var_max, 0, 0],
        ]
    )
    apex_point = np.array([0, 0, var_max])

    def _con_rule(m, i):
        vec_1 = base_points[i] - apex_point
        vec_2 = base_points[(i + 1) % var_max] - base_points[i]
        n = np.cross(vec_1, vec_2)
        expr = (
            n[0] * (m.x[1] - apex_point[0])
            + n[1] * (m.x[2] - apex_point[1])
            + n[2] * (m.z - apex_point[2])
        )
        return expr >= 0

    m.c = pyo.Constraint([i for i in range(5)], rule=_con_rule)
    m.num_ranked_solns = [1, 4, 2, 8, 2, 12, 4, 16, 4, 20]
    return m


def get_bloated_pentagonal_pyramid_mip():
    """
    Pentagonal pyramid with integer coordinates in the first two dimensions and
    a third continuous dimension. Bounds are artificially widened for obbt testing purposes
    """
    var_max = 5
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Integers, bounds=(-2 * var_max, 2 * var_max))
    m.y = pyo.Var(within=pyo.Integers, bounds=(-2 * var_max, var_max))
    m.z = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 2 * var_max))
    m.var_bounds = pyo.ComponentMap()
    m.o = pyo.Objective(expr=m.z, sense=pyo.maximize)
    base_points = np.array(
        [
            [0, var_max, 0],
            [var_max, 0, 0],
            [var_max / 2.0, -var_max, 0],
            [-var_max / 2.0, -var_max, 0],
            [-var_max, 0, 0],
        ]
    )
    apex_point = np.array([0, 0, var_max])

    m.c = pyo.ConstraintList()
    for i in range(5):
        vec_1 = base_points[i] - apex_point
        vec_2 = base_points[(i + 1) % var_max] - base_points[i]
        n = np.cross(vec_1, vec_2)
        m.c.add(
            n[0] * (m.x - apex_point[0])
            + n[1] * (m.y - apex_point[1])
            + n[2] * (m.z - apex_point[2])
            >= 0
        )
    return m
