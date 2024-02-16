#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# wl.py # define a script to demonstrate performance profiling and improvements
# @imports:
import pyomo.environ as pyo  # import pyomo environment
import cProfile
import pstats
import io
from pyomo.common.timing import TicTocTimer, report_timing
from pyomo.opt.results import assert_optimal_termination
from pyomo.core.expr.numeric_expr import LinearExpression
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
# @:imports


# @model_func:
def create_warehouse_model(num_locations=50, num_customers=50):
    N = list(range(num_locations))  # warehouse locations
    M = list(range(num_customers))  # customers

    d = dict()  # distances from warehouse locations to customers
    for n in N:
        for m in M:
            d[n, m] = np.random.randint(low=1, high=100)
    max_num_warehouses = 2

    model = pyo.ConcreteModel(name="(WL)")
    model.P = pyo.Param(initialize=max_num_warehouses, mutable=True)

    model.x = pyo.Var(N, M, bounds=(0, 1))
    model.y = pyo.Var(N, bounds=(0, 1))

    def obj_rule(mdl):
        return sum(d[n, m] * mdl.x[n, m] for n in N for m in M)

    model.obj = pyo.Objective(rule=obj_rule)

    def demand_rule(mdl, m):
        return sum(mdl.x[n, m] for n in N) == 1

    model.demand = pyo.Constraint(M, rule=demand_rule)

    def warehouse_active_rule(mdl, n, m):
        return mdl.x[n, m] <= mdl.y[n]

    model.warehouse_active = pyo.Constraint(N, M, rule=warehouse_active_rule)

    def num_warehouses_rule(mdl):
        return sum(mdl.y[n] for n in N) <= model.P

    model.num_warehouses = pyo.Constraint(rule=num_warehouses_rule)

    return model


# @:model_func


# @model_linear_expr:
def create_warehouse_linear_expr(num_locations=50, num_customers=50):
    N = list(range(num_locations))  # warehouse locations
    M = list(range(num_customers))  # customers

    d = dict()  # distances from warehouse locations to customers
    for n in N:
        for m in M:
            d[n, m] = np.random.randint(low=1, high=100)
    max_num_warehouses = 2

    model = pyo.ConcreteModel(name="(WL)")
    model.P = pyo.Param(initialize=max_num_warehouses, mutable=True)

    model.x = pyo.Var(N, M, bounds=(0, 1))
    model.y = pyo.Var(N, bounds=(0, 1))

    def obj_rule(mdl):
        return sum(d[n, m] * mdl.x[n, m] for n in N for m in M)

    model.obj = pyo.Objective(rule=obj_rule)

    def demand_rule(mdl, m):
        return sum(mdl.x[n, m] for n in N) == 1

    model.demand = pyo.Constraint(M, rule=demand_rule)

    def warehouse_active_rule(mdl, n, m):
        expr = LinearExpression(
            constant=0, linear_coefs=[1, -1], linear_vars=[mdl.x[n, m], mdl.y[n]]
        )
        return expr <= 0

    model.warehouse_active = pyo.Constraint(N, M, rule=warehouse_active_rule)

    def num_warehouses_rule(mdl):
        return sum(mdl.y[n] for n in N) <= model.P

    model.num_warehouses = pyo.Constraint(rule=num_warehouses_rule)

    return model


# @:model_linear_expr


# @print_c_profiler:
def print_c_profiler(pr, lines_to_print=15):
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    stats.print_stats(lines_to_print)
    print(s.getvalue())
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats('tottime')
    stats.print_stats(lines_to_print)
    print(s.getvalue())


# @:print_c_profiler


# @solve_warehouse_location:
def solve_warehouse_location(m):
    opt = pyo.SolverFactory('gurobi')
    res = opt.solve(m)
    assert_optimal_termination(res)


# @:solve_warehouse_location


# @solve_parametric:
def solve_parametric():
    m = create_warehouse_model(num_locations=50, num_customers=50)
    opt = pyo.SolverFactory('gurobi')
    p_values = list(range(1, 31))
    obj_values = list()
    for p in p_values:
        m.P.value = p
        res = opt.solve(m)
        assert_optimal_termination(res)
        obj_values.append(res.problem.lower_bound)


# @:solve_parametric


# @parametric_persistent:
def solve_parametric_persistent():
    m = create_warehouse_model(num_locations=50, num_customers=50)
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    p_values = list(range(1, 31))
    obj_values = list()
    for p in p_values:
        m.P.value = p
        opt.remove_constraint(m.num_warehouses)
        opt.add_constraint(m.num_warehouses)
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        obj_values.append(res.problem.lower_bound)


# @:parametric_persistent

# @report_timing:
report_timing()
print('Building model')
print('--------------')
m = create_warehouse_model(num_locations=200, num_customers=200)
# @:report_timing

# @report_timing_with_lin_expr:
print('Building model with LinearExpression')
print('------------------------------------')
m = create_warehouse_linear_expr(num_locations=200, num_customers=200)
# @:report_timing_with_lin_expr

report_timing(False)

# @tic_toc_timer:
timer = TicTocTimer()
timer.tic('start')
m = create_warehouse_model(num_locations=200, num_customers=200)
timer.toc('Built model')
solve_warehouse_location(m)
timer.toc('Wrote LP file and solved')
# @:tic_toc_timer

# @time_parametric:
solve_parametric()
timer.toc('Finished parameter sweep')
# @:time_parametric

# @profile_parametric:
pr = cProfile.Profile()
pr.enable()
solve_parametric()
pr.disable()
print_c_profiler(pr)
# @:profile_parametric

# @time_parametric_persistent:
timer.tic()
solve_parametric_persistent()
timer.toc('Finished parameter sweep with persistent interface')
# @:time_parametric_persistent

# @profile_parametric_persistent:
# pr = cProfile.Profile()
# pr.enable()
# solve_parametric_persistent()
# pr.disable()
# print_c_profiler(pr)
# @:profile_parametric_persistent
