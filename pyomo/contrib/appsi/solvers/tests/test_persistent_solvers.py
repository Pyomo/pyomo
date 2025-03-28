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

import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest

parameterized, param_available = attempt_import('parameterized')
parameterized = parameterized.parameterized
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.appsi.solvers import Gurobi, Ipopt, Cplex, Cbc, Highs, MAiNGO
from typing import Type
from pyomo.core.expr.numeric_expr import LinearExpression
import os

numpy, numpy_available = attempt_import('numpy')
import random
from pyomo import gdp


if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

all_solvers = [
    ('gurobi', Gurobi),
    ('ipopt', Ipopt),
    ('cplex', Cplex),
    ('cbc', Cbc),
    ('highs', Highs),
    ('maingo', MAiNGO),
]
mip_solvers = [
    ('gurobi', Gurobi),
    ('cplex', Cplex),
    ('cbc', Cbc),
    ('highs', Highs),
    ('maingo', MAiNGO),
]
nlp_solvers = [('ipopt', Ipopt), ('maingo', MAiNGO)]
qcp_solvers = [
    ('gurobi', Gurobi),
    ('ipopt', Ipopt),
    ('cplex', Cplex),
    ('maingo', MAiNGO),
]
miqcqp_solvers = [('gurobi', Gurobi), ('cplex', Cplex), ('maingo', MAiNGO)]
only_child_vars_options = [True, False]


"""
The tests in this file are used to ensure basic functionality/API works with all solvers

Feature                                    Tested
-------                                    ------
config time_limit                          
config tee                                 
config load_solution True                  x
config load_solution False                 x     
results termination condition optimal      x
results termination condition infeasible   x
load_vars                                  
get_duals                                  x
get_reduced_costs                          x
range constraints                          x
MILP
Model updates - added constraints          x
Model updates - removed constraints        x
Model updates - added vars
Model updates - removed vars
Model updates - changed named expression
Model updates - mutable param modified     x
Model updates - var modified
Model updates - objective changed
Model updates - constraint modified
No objective
No constraints                             x
bounds                                     x
best feasible objective                    x
best objective bound                       x
fixed variables
"""


def _load_tests(solver_list, only_child_vars_list):
    res = list()
    for solver_name, solver in solver_list:
        for child_var_option in only_child_vars_list:
            test_name = f"{solver_name}_only_child_vars_{child_var_option}"
            res.append((test_name, solver, child_var_option))
    return res


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestSolvers(unittest.TestCase):
    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_remove_variable_and_objective(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        # this test is for issue #2888
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(2, None))
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 2)

        del m.x
        del m.obj
        m.x = pyo.Var(bounds=(2, None))
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 2)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_stale_vars(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x)
        m.x.value = 1
        m.y.value = 1
        m.z.value = 1
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        self.assertFalse(m.z.stale)

        res = opt.solve(m)
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        self.assertTrue(m.z.stale)

        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertTrue(m.x.stale)
        self.assertTrue(m.y.stale)
        self.assertTrue(m.z.stale)
        res.solution_loader.load_vars()
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        self.assertTrue(m.z.stale)

        res = opt.solve(m)
        self.assertTrue(m.x.stale)
        self.assertTrue(m.y.stale)
        self.assertTrue(m.z.stale)
        res.solution_loader.load_vars([m.y])
        self.assertTrue(m.x.stale)
        self.assertFalse(m.y.stale)
        self.assertTrue(m.z.stale)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_range_constraint(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=(-1, m.x, 1))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c], 1)
        m.obj.sense = pyo.maximize
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c], 1)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_reduced_costs(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var(bounds=(-2, 2))
        m.obj = pyo.Objective(expr=3 * m.x + 4 * m.y)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.y.value, -2)
        if opt_class != MAiNGO:
            rc = opt.get_reduced_costs()
            self.assertAlmostEqual(rc[m.x], 3)
            self.assertAlmostEqual(rc[m.y], 4)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_reduced_costs2(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        if opt_class != MAiNGO:
            rc = opt.get_reduced_costs()
            self.assertAlmostEqual(rc[m.x], 1)
        m.obj.sense = pyo.maximize
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        if opt_class != MAiNGO:
            rc = opt.get_reduced_costs()
            self.assertAlmostEqual(rc[m.x], 1)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_param_changes(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(mutable=True)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pyo.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
            self.assertTrue(res.best_objective_bound <= m.y.value)
            if opt_class != MAiNGO:
                duals = opt.get_duals()
                self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
                self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_immutable_param(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        """
        This test is important because component_data_objects returns immutable params as floats.
        We want to make sure we process these correctly.
        """
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(initialize=-1)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pyo.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))

        params_to_test = [(1, 2, 1), (1, 2, 1), (1, 3, 1)]
        for a1, b1, b2 in params_to_test:
            a2 = m.a2.value
            m.a1.value = a1
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
            self.assertTrue(res.best_objective_bound <= m.y.value)
            if opt_class != MAiNGO:
                duals = opt.get_duals()
                self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
                self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_equality(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(mutable=True)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y == m.a1 * m.x + m.b1)
        m.c2 = pyo.Constraint(expr=m.y == m.a2 * m.x + m.b2)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
            self.assertTrue(res.best_objective_bound <= m.y.value)
            if opt_class != MAiNGO:
                duals = opt.get_duals()
                self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
                self.assertAlmostEqual(duals[m.c2], -a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_linear_expression(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(mutable=True)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        e = LinearExpression(
            constant=m.b1, linear_coefs=[-1, m.a1], linear_vars=[m.y, m.x]
        )
        m.c1 = pyo.Constraint(expr=e == 0)
        e = LinearExpression(
            constant=m.b2, linear_coefs=[-1, m.a2], linear_vars=[m.y, m.x]
        )
        m.c2 = pyo.Constraint(expr=e == 0)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
            self.assertTrue(res.best_objective_bound <= m.y.value)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_no_objective(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(mutable=True)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.c1 = pyo.Constraint(expr=m.y == m.a1 * m.x + m.b1)
        m.c2 = pyo.Constraint(expr=m.y == m.a2 * m.x + m.b2)
        opt.config.stream_solver = True

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertEqual(res.best_feasible_objective, None)
            self.assertEqual(res.best_objective_bound, None)
            if opt_class != MAiNGO:
                duals = opt.get_duals()
                self.assertAlmostEqual(duals[m.c1], 0)
                self.assertAlmostEqual(duals[m.c2], 0)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_add_remove_cons(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        a1 = -1
        a2 = 1
        b1 = 1
        b2 = 2
        a3 = 1
        b3 = 3
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= a1 * m.x + b1)
        m.c2 = pyo.Constraint(expr=m.y >= a2 * m.x + b2)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
        self.assertTrue(res.best_objective_bound <= m.y.value)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
            self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

        m.c3 = pyo.Constraint(expr=m.y >= a3 * m.x + b3)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, (b3 - b1) / (a1 - a3))
        self.assertAlmostEqual(m.y.value, a1 * (b3 - b1) / (a1 - a3) + b1)
        self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
        self.assertTrue(res.best_objective_bound <= m.y.value)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a3 - a1)))
            self.assertAlmostEqual(duals[m.c2], 0)
            self.assertAlmostEqual(duals[m.c3], a1 / (a3 - a1))

        del m.c3
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(res.best_feasible_objective, m.y.value)
        self.assertTrue(res.best_objective_bound <= m.y.value)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
            self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_results_infeasible(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x)
        m.c2 = pyo.Constraint(expr=m.y <= m.x - 1)
        with self.assertRaises(Exception):
            res = opt.solve(m)
        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertNotEqual(res.termination_condition, TerminationCondition.optimal)
        if opt_class is Ipopt:
            acceptable_termination_conditions = {
                TerminationCondition.infeasible,
                TerminationCondition.unbounded,
            }
        else:
            acceptable_termination_conditions = {
                TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded,
            }
        self.assertIn(res.termination_condition, acceptable_termination_conditions)
        self.assertAlmostEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, None)
        self.assertTrue(res.best_feasible_objective is None)

        with self.assertRaisesRegex(
            RuntimeError, '.*does not currently have a valid solution.*'
        ):
            res.solution_loader.load_vars()
        if opt_class != MAiNGO:
            with self.assertRaisesRegex(
                RuntimeError, '.*does not currently have valid duals.*'
            ):
                res.solution_loader.get_duals()
            with self.assertRaisesRegex(
                RuntimeError, '.*does not currently have valid reduced costs.*'
            ):
                res.solution_loader.get_reduced_costs()

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_duals(self, name: str, opt_class: Type[PersistentSolver], only_child_vars):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y - m.x >= 0)
        m.c2 = pyo.Constraint(expr=m.y + m.x - 2 >= 0)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        if opt_class != MAiNGO:
            duals = opt.get_duals()
            self.assertAlmostEqual(duals[m.c1], 0.5)
            self.assertAlmostEqual(duals[m.c2], 0.5)

            duals = opt.get_duals(cons_to_load=[m.c1])
            self.assertAlmostEqual(duals[m.c1], 0.5)
            self.assertNotIn(m.c2, duals)

    @parameterized.expand(input=_load_tests(qcp_solvers, only_child_vars_options))
    def test_mutable_quadratic_coefficient(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=-1, mutable=True)
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c = pyo.Constraint(expr=m.y >= (m.a * m.x + m.b) ** 2)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.41024548525899274, 4)
        self.assertAlmostEqual(m.y.value, 0.34781038127030117, 4)
        m.a.value = 2
        m.b.value = -0.5
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.10256137418973625, 4)
        self.assertAlmostEqual(m.y.value, 0.0869525991355825, 4)

    @parameterized.expand(input=_load_tests(qcp_solvers, only_child_vars_options))
    def test_mutable_quadratic_objective(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=-1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.d = pyo.Param(initialize=1, mutable=True)
        m.obj = pyo.Objective(expr=m.x**2 + m.c * m.y**2 + m.d * m.x)
        m.ccon = pyo.Constraint(expr=m.y >= (m.a * m.x + m.b) ** 2)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.2719178742733325, 4)
        self.assertAlmostEqual(m.y.value, 0.5301035741688002, 4)
        m.c.value = 3.5
        m.d.value = -1
        res = opt.solve(m)

        self.assertAlmostEqual(m.x.value, 0.6962249634573562, 4)
        self.assertAlmostEqual(m.y.value, 0.09227926676152151, 4)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_fixed_vars(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        for treat_fixed_vars_as_params in [True, False]:
            opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
            opt.update_config.treat_fixed_vars_as_params = treat_fixed_vars_as_params
            if not opt.available():
                raise unittest.SkipTest
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.x.fix(0)
            m.y = pyo.Var()
            a1 = 1
            a2 = -1
            b1 = 1
            b2 = 2
            m.obj = pyo.Objective(expr=m.y)
            m.c1 = pyo.Constraint(expr=m.y >= a1 * m.x + b1)
            m.c2 = pyo.Constraint(expr=m.y >= a2 * m.x + b2)
            res = opt.solve(m)
            self.assertAlmostEqual(m.x.value, 0)
            self.assertAlmostEqual(m.y.value, 2)
            m.x.unfix()
            res = opt.solve(m)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            m.x.fix(0)
            res = opt.solve(m)
            self.assertAlmostEqual(m.x.value, 0)
            self.assertAlmostEqual(m.y.value, 2)
            m.x.value = 2
            res = opt.solve(m)
            self.assertAlmostEqual(m.x.value, 2)
            self.assertAlmostEqual(m.y.value, 3)
            m.x.value = 0
            res = opt.solve(m)
            self.assertAlmostEqual(m.x.value, 0)
            self.assertAlmostEqual(m.y.value, 2)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_fixed_vars_2(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        opt.update_config.treat_fixed_vars_as_params = True
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.x.fix(0)
        m.y = pyo.Var()
        a1 = 1
        a2 = -1
        b1 = 1
        b2 = 2
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= a1 * m.x + b1)
        m.c2 = pyo.Constraint(expr=m.y >= a2 * m.x + b2)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)
        m.x.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)
        m.x.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 3)
        m.x.value = 0
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_fixed_vars_3(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        opt.update_config.treat_fixed_vars_as_params = True
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x + m.y)
        m.c1 = pyo.Constraint(expr=m.x == 2 / m.y)
        m.y.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)

    @parameterized.expand(input=_load_tests(nlp_solvers, only_child_vars_options))
    def test_fixed_vars_4(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        opt.update_config.treat_fixed_vars_as_params = True
        if not opt.available() or opt_class == MAiNGO:
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.x == 2 / m.y)
        m.y.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        m.y.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2**0.5)
        self.assertAlmostEqual(m.y.value, 2**0.5)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_mutable_param_with_range(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        try:
            import numpy as np
        except:
            raise unittest.SkipTest('numpy is not available')
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(initialize=0, mutable=True)
        m.a2 = pyo.Param(initialize=0, mutable=True)
        m.b1 = pyo.Param(initialize=0, mutable=True)
        m.b2 = pyo.Param(initialize=0, mutable=True)
        m.c1 = pyo.Param(initialize=0, mutable=True)
        m.c2 = pyo.Param(initialize=0, mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        m.con1 = pyo.Constraint(expr=(m.b1, m.y - m.a1 * m.x, m.c1))
        m.con2 = pyo.Constraint(expr=(m.b2, m.y - m.a2 * m.x, m.c2))

        np.random.seed(0)
        params_to_test = [
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pyo.minimize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pyo.maximize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pyo.minimize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pyo.maximize,
            ),
        ]
        for a1, a2, b1, b2, c1, c2, sense in params_to_test:
            m.a1.value = float(a1)
            m.a2.value = float(a2)
            m.b1.value = float(b1)
            m.b2.value = float(b2)
            m.c1.value = float(c1)
            m.c2.value = float(c2)
            m.obj.sense = sense
            res: Results = opt.solve(m)
            self.assertEqual(res.termination_condition, TerminationCondition.optimal)
            if sense is pyo.minimize:
                self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2), 6)
                self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1, 6)
                self.assertAlmostEqual(res.best_feasible_objective, m.y.value, 6)
                self.assertTrue(res.best_objective_bound <= m.y.value + 1e-12)
                if opt_class != MAiNGO:
                    duals = opt.get_duals()
                    self.assertAlmostEqual(duals[m.con1], (1 + a1 / (a2 - a1)), 6)
                    self.assertAlmostEqual(duals[m.con2], -a1 / (a2 - a1), 6)
            else:
                self.assertAlmostEqual(m.x.value, (c2 - c1) / (a1 - a2), 6)
                self.assertAlmostEqual(m.y.value, a1 * (c2 - c1) / (a1 - a2) + c1, 6)
                self.assertAlmostEqual(res.best_feasible_objective, m.y.value, 6)
                self.assertTrue(res.best_objective_bound >= m.y.value - 1e-12)
                if opt_class != MAiNGO:
                    duals = opt.get_duals()
                    self.assertAlmostEqual(duals[m.con1], (1 + a1 / (a2 - a1)), 6)
                    self.assertAlmostEqual(duals[m.con2], -a1 / (a2 - a1), 6)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_add_and_remove_vars(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.y = pyo.Var(bounds=(-1, None))
        m.obj = pyo.Objective(expr=m.y)
        opt.update_config.update_params = False
        opt.update_config.update_vars = False
        opt.update_config.update_constraints = False
        opt.update_config.update_named_expressions = False
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.check_for_new_or_removed_constraints = False
        opt.update_config.check_for_new_or_removed_vars = False
        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        opt.load_vars()
        self.assertAlmostEqual(m.y.value, -1)
        m.x = pyo.Var()
        a1 = 1
        a2 = -1
        b1 = 2
        b2 = 1
        m.c1 = pyo.Constraint(expr=(0, m.y - a1 * m.x - b1, None))
        m.c2 = pyo.Constraint(expr=(None, -m.y + a2 * m.x + b2, 0))
        if only_child_vars:
            opt.add_variables([m.x])
        opt.add_constraints([m.c1, m.c2])
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        opt.remove_constraints([m.c1, m.c2])
        if only_child_vars:
            opt.remove_variables([m.x])
        m.x.value = None
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        opt.load_vars()
        self.assertEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, -1)
        with self.assertRaises(Exception):
            opt.load_vars([m.x])

    @parameterized.expand(input=_load_tests(nlp_solvers, only_child_vars_options))
    def test_exp(self, name: str, opt_class: Type[PersistentSolver], only_child_vars):
        opt = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.42630274815985264, 6)
        self.assertAlmostEqual(m.y.value, 0.6529186341994245, 6)

    @parameterized.expand(input=_load_tests(nlp_solvers, only_child_vars_options))
    def test_log(self, name: str, opt_class: Type[PersistentSolver], only_child_vars):
        opt = opt_class(only_child_vars=only_child_vars)
        if not opt.available() or opt_class == MAiNGO:
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1)
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y <= pyo.log(m.x))
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.6529186341994245)
        self.assertAlmostEqual(m.y.value, -0.42630274815985264)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_with_numpy(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        a1 = 1
        b1 = 3
        a2 = -2
        b2 = 1
        m.c1 = pyo.Constraint(
            expr=(numpy.float64(0), m.y - numpy.int64(1) * m.x - numpy.float32(3), None)
        )
        m.c2 = pyo.Constraint(
            expr=(
                None,
                -m.y + numpy.int32(-2) * m.x + numpy.float64(1),
                numpy.float16(0),
            )
        )
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_bounds_with_params(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True)
        m.y.setlb(m.p)
        m.p.value = 1
        m.obj = pyo.Objective(expr=m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 1)
        m.p.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, -1)
        m.y.setlb(None)
        m.y.setub(m.p)
        m.obj.sense = pyo.maximize
        m.p.value = 5
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 5)
        m.p.value = 4
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 4)
        m.y.setub(None)
        m.y.setlb(m.p)
        m.obj.sense = pyo.minimize
        m.p.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 3)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_bounds_with_immutable_params(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        # this test is for issue #2574
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=False, initialize=1)
        m.q = pyo.Param([1, 2], mutable=False, initialize=10)
        m.y = pyo.Var()
        m.y.setlb(m.p)
        m.y.setub(m.q[1])
        m.obj = pyo.Objective(expr=m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 1)
        m.y.setlb(m.q[2])
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 10)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_solution_loader(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, None))
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=(0, m.y - m.x, None))
        m.c2 = pyo.Constraint(expr=(0, m.y - m.x + 1, None))
        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.y.value)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        m.x.value = None
        m.y.value = None
        res.solution_loader.load_vars([m.y])
        self.assertIsNone(m.x.value)
        self.assertAlmostEqual(m.y.value, 1)
        primals = res.solution_loader.get_primals()
        self.assertIn(m.x, primals)
        self.assertIn(m.y, primals)
        self.assertAlmostEqual(primals[m.x], 1)
        self.assertAlmostEqual(primals[m.y], 1)
        primals = res.solution_loader.get_primals([m.y])
        self.assertNotIn(m.x, primals)
        self.assertIn(m.y, primals)
        self.assertAlmostEqual(primals[m.y], 1)
        if opt_class != MAiNGO:
            reduced_costs = res.solution_loader.get_reduced_costs()
            self.assertIn(m.x, reduced_costs)
            self.assertIn(m.y, reduced_costs)
            self.assertAlmostEqual(reduced_costs[m.x], 1)
            self.assertAlmostEqual(reduced_costs[m.y], 0)
            reduced_costs = res.solution_loader.get_reduced_costs([m.y])
            self.assertNotIn(m.x, reduced_costs)
            self.assertIn(m.y, reduced_costs)
            self.assertAlmostEqual(reduced_costs[m.y], 0)
            duals = res.solution_loader.get_duals()
            self.assertIn(m.c1, duals)
            self.assertIn(m.c2, duals)
            self.assertAlmostEqual(duals[m.c1], 1)
            self.assertAlmostEqual(duals[m.c2], 0)
            duals = res.solution_loader.get_duals([m.c1])
            self.assertNotIn(m.c2, duals)
            self.assertIn(m.c1, duals)
            self.assertAlmostEqual(duals[m.c1], 1)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_time_limit(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available() or opt_class == MAiNGO:
            raise unittest.SkipTest
        from sys import platform

        if platform == 'win32':
            raise unittest.SkipTest

        N = 30
        m = pyo.ConcreteModel()
        m.jobs = pyo.Set(initialize=list(range(N)))
        m.tasks = pyo.Set(initialize=list(range(N)))
        m.x = pyo.Var(m.jobs, m.tasks, bounds=(0, 1))

        random.seed(0)
        coefs = list()
        lin_vars = list()
        for j in m.jobs:
            for t in m.tasks:
                coefs.append(random.uniform(0, 10))
                lin_vars.append(m.x[j, t])
        obj_expr = LinearExpression(
            linear_coefs=coefs, linear_vars=lin_vars, constant=0
        )
        m.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)

        m.c1 = pyo.Constraint(m.jobs)
        m.c2 = pyo.Constraint(m.tasks)
        for j in m.jobs:
            expr = LinearExpression(
                linear_coefs=[1] * N,
                linear_vars=[m.x[j, t] for t in m.tasks],
                constant=0,
            )
            m.c1[j] = expr == 1
        for t in m.tasks:
            expr = LinearExpression(
                linear_coefs=[1] * N,
                linear_vars=[m.x[j, t] for j in m.jobs],
                constant=0,
            )
            m.c2[t] = expr == 1
        if type(opt) is Ipopt:
            opt.config.time_limit = 1e-6
        else:
            opt.config.time_limit = 0
        opt.config.load_solution = False
        res = opt.solve(m)
        if type(opt) is Cbc:  # I can't figure out why CBC is reporting max iter...
            self.assertIn(
                res.termination_condition,
                {TerminationCondition.maxIterations, TerminationCondition.maxTimeLimit},
            )
        else:
            self.assertEqual(
                res.termination_condition, TerminationCondition.maxTimeLimit
            )

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_objective_changes(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        m.obj = pyo.Objective(expr=m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        m.obj = pyo.Objective(expr=2 * m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 2)
        m.obj.expr = 3 * m.y
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 3)
        m.obj.sense = pyo.maximize
        opt.config.load_solution = False
        res = opt.solve(m)
        if opt_class != MAiNGO:
            self.assertIn(
                res.termination_condition,
                {
                    TerminationCondition.unbounded,
                    TerminationCondition.infeasibleOrUnbounded,
                },
            )
        m.obj.sense = pyo.minimize
        opt.config.load_solution = True
        m.obj = pyo.Objective(expr=m.x * m.y)
        m.x.fix(2)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 6, 6)
        m.x.fix(3)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 12, 6)
        m.x.unfix()
        m.y.fix(2)
        m.x.setlb(-3)
        m.x.setub(5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -2, 6)
        m.y.unfix()
        m.x.setlb(None)
        m.x.setub(None)
        m.e = pyo.Expression(expr=2)
        m.obj = pyo.Objective(expr=m.e * m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 2)
        m.e.expr = 3
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 3)
        opt.update_config.check_for_new_objective = False
        m.e.expr = 4
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 4)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_domain(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, None), domain=pyo.NonNegativeReals)
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        m.x.setlb(-1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0)
        m.x.setlb(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        m.x.setlb(-1)
        m.x.domain = pyo.Reals
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -1)
        m.x.domain = pyo.NonNegativeReals
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0)

    @parameterized.expand(input=_load_tests(mip_solvers, only_child_vars_options))
    def test_domain_with_integers(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, None), domain=pyo.NonNegativeIntegers)
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0)
        m.x.setlb(0.5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        m.x.setlb(-5.5)
        m.x.domain = pyo.Integers
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -5)
        m.x.domain = pyo.Binary
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0)
        m.x.setlb(0.5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_fixed_binaries(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c = pyo.Constraint(expr=m.y >= m.x)
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0, 5)
        m.x.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1, 5)

        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        opt.update_config.treat_fixed_vars_as_params = False
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 0, 5)
        m.x.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1, 5)

    @parameterized.expand(input=_load_tests(mip_solvers, only_child_vars_options))
    def test_with_gdp(
        self, name: str, opt_class: Type[PersistentSolver], only_child_vars
    ):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var(bounds=(-10, 10))
        m.obj = pyo.Objective(expr=m.y)
        m.d1 = gdp.Disjunct()
        m.d1.c1 = pyo.Constraint(expr=m.y >= m.x + 2)
        m.d1.c2 = pyo.Constraint(expr=m.y >= -m.x + 2)
        m.d2 = gdp.Disjunct()
        m.d2.c1 = pyo.Constraint(expr=m.y >= m.x + 1)
        m.d2.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        m.disjunction = gdp.Disjunction(expr=[m.d2, m.d1])
        pyo.TransformationFactory("gdp.bigm").apply_to(m)

        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1, 6)
        self.assertAlmostEqual(m.x.value, 0, 6)
        self.assertAlmostEqual(m.y.value, 1, 6)

        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        opt.use_extensions = True
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1, 6)
        self.assertAlmostEqual(m.x.value, 0, 6)
        self.assertAlmostEqual(m.y.value, 1, 6)

    @parameterized.expand(input=all_solvers)
    def test_variables_elsewhere(self, name: str, opt_class: Type[PersistentSolver]):
        opt: PersistentSolver = opt_class(only_child_vars=False)
        if not opt.available():
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.b = pyo.Block()
        m.b.obj = pyo.Objective(expr=m.y)
        m.b.c1 = pyo.Constraint(expr=m.y >= m.x + 2)
        m.b.c2 = pyo.Constraint(expr=m.y >= -m.x)

        res = opt.solve(m.b)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.y.value, 1)

        m.x.setlb(0)
        res = opt.solve(m.b)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 2)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)

    @parameterized.expand(input=all_solvers)
    def test_variables_elsewhere2(self, name: str, opt_class: Type[PersistentSolver]):
        opt: PersistentSolver = opt_class(only_child_vars=False)
        if not opt.available():
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x)
        m.c3 = pyo.Constraint(expr=m.y >= m.z + 1)
        m.c4 = pyo.Constraint(expr=m.y >= -m.z + 1)

        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        sol = res.solution_loader.get_primals()
        self.assertIn(m.x, sol)
        self.assertIn(m.y, sol)
        self.assertIn(m.z, sol)

        del m.c3
        del m.c4
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 0)
        sol = res.solution_loader.get_primals()
        self.assertIn(m.x, sol)
        self.assertIn(m.y, sol)
        self.assertNotIn(m.z, sol)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_bug_1(self, name: str, opt_class: Type[PersistentSolver], only_child_vars):
        opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
        if not opt.available():
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(3, 7))
        m.y = pyo.Var(bounds=(-10, 10))
        m.p = pyo.Param(mutable=True, initialize=0)

        m.obj = pyo.Objective(expr=m.y)
        m.c = pyo.Constraint(expr=m.y >= m.p * m.x)

        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 0)

        m.p.value = 1
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 3)

    @parameterized.expand(input=_load_tests(all_solvers, only_child_vars_options))
    def test_bug_2(self, name: str, opt_class: Type[PersistentSolver], only_child_vars):
        """
        This test is for a bug where an objective containing a fixed variable does
        not get updated properly when the variable is unfixed.
        """
        for fixed_var_option in [True, False]:
            opt: PersistentSolver = opt_class(only_child_vars=only_child_vars)
            if not opt.available():
                raise unittest.SkipTest
            opt.update_config.treat_fixed_vars_as_params = fixed_var_option

            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(-10, 10))
            m.y = pyo.Var()
            m.obj = pyo.Objective(expr=3 * m.y - m.x)
            m.c = pyo.Constraint(expr=m.y >= m.x)

            m.x.fix(1)
            res = opt.solve(m)
            self.assertAlmostEqual(res.best_feasible_objective, 2, 5)

            m.x.unfix()
            m.x.setlb(-9)
            m.x.setub(9)
            res = opt.solve(m)
            self.assertAlmostEqual(res.best_feasible_objective, -18, 5)


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestLegacySolverInterface(unittest.TestCase):
    @parameterized.expand(input=all_solvers)
    def test_param_updates(self, name: str, opt_class: Type[PersistentSolver]):
        opt = pyo.SolverFactory('appsi_' + name)
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.a1 = pyo.Param(mutable=True)
        m.a2 = pyo.Param(mutable=True)
        m.b1 = pyo.Param(mutable=True)
        m.b2 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pyo.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))
        if opt_class != MAiNGO:
            m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res = opt.solve(m)
            pyo.assert_optimal_termination(res)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            if opt_class != MAiNGO:
                self.assertAlmostEqual(m.dual[m.c1], (1 + a1 / (a2 - a1)))
                self.assertAlmostEqual(m.dual[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=all_solvers)
    def test_load_solutions(self, name: str, opt_class: Type[PersistentSolver]):
        opt = pyo.SolverFactory('appsi_' + name)
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=(-1, m.x, 1))
        if opt_class != MAiNGO:
            m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        res = opt.solve(m, load_solutions=False)
        pyo.assert_optimal_termination(res)
        self.assertIsNone(m.x.value)
        if opt_class != MAiNGO:
            self.assertNotIn(m.c, m.dual)
        m.solutions.load_from(res)
        self.assertAlmostEqual(m.x.value, -1)
        if opt_class != MAiNGO:
            self.assertAlmostEqual(m.dual[m.c], 1)
