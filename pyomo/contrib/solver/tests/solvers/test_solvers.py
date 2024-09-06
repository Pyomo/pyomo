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

import random
import math
from typing import Type

import pyomo.environ as pe
from pyomo import gdp
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus, Results
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.ipopt import Ipopt
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.gurobi_direct import GurobiDirect
from pyomo.core.expr.numeric_expr import LinearExpression


np, numpy_available = attempt_import('numpy')
parameterized, param_available = attempt_import('parameterized')
parameterized = parameterized.parameterized


if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

all_solvers = [('gurobi', Gurobi), ('gurobi_direct', GurobiDirect), ('ipopt', Ipopt)]
mip_solvers = [('gurobi', Gurobi), ('gurobi_direct', GurobiDirect)]
nlp_solvers = [('ipopt', Ipopt)]
qcp_solvers = [('gurobi', Gurobi), ('ipopt', Ipopt)]
miqcqp_solvers = [('gurobi', Gurobi)]
nl_solvers = [('ipopt', Ipopt)]
nl_solvers_set = {i[0] for i in nl_solvers}


def _load_tests(solver_list):
    res = list()
    for solver_name, solver in solver_list:
        if solver_name in nl_solvers_set:
            test_name = f"{solver_name}_presolve"
            res.append((test_name, solver, True))
            test_name = f"{solver_name}"
            res.append((test_name, solver, False))
        else:
            test_name = f"{solver_name}"
            res.append((test_name, solver, None))
    return res


@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestSolvers(unittest.TestCase):
    @parameterized.expand(input=all_solvers)
    def test_config_overwrite(self, name: str, opt_class: Type[SolverBase]):
        self.assertIsNot(SolverBase.CONFIG, opt_class.CONFIG)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_remove_variable_and_objective(
        self, name: str, opt_class: Type[SolverBase], use_presolve
    ):
        # this test is for issue #2888
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(2, None))
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 2)

        del m.x
        del m.obj
        m.x = pe.Var(bounds=(2, None))
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 2)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_stale_vars(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y >= -m.x)
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

        opt.config.load_solutions = False
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
        self.assertFalse(m.y.stale)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_range_constraint(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.obj = pe.Objective(expr=m.x)
        m.c = pe.Constraint(expr=(-1, m.x, 1))
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c], 1)
        m.obj.sense = pe.maximize
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c], 1)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_reduced_costs(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var(bounds=(-2, 2))
        m.obj = pe.Objective(expr=3 * m.x + 4 * m.y)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.y.value, -2)
        rc = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rc[m.x], 3)
        self.assertAlmostEqual(rc[m.y], 4)
        m.obj.expr *= -1
        res = opt.solve(m)
        rc = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rc[m.x], -3)
        self.assertAlmostEqual(rc[m.y], -4)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_reduced_costs2(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, -1)
        rc = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rc[m.x], 1)
        m.obj.sense = pe.maximize
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        rc = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rc[m.x], 1)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_param_changes(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(mutable=True)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pe.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.incumbent_objective, m.y.value)
            if res.objective_bound is None:
                bound = -math.inf
            else:
                bound = res.objective_bound
            self.assertTrue(bound <= m.y.value)
            duals = res.solution_loader.get_duals()
            self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
            self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_immutable_param(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        """
        This test is important because component_data_objects returns immutable params as floats.
        We want to make sure we process these correctly.
        """
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(initialize=-1)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pe.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))

        params_to_test = [(1, 2, 1), (1, 2, 1), (1, 3, 1)]
        for a1, b1, b2 in params_to_test:
            a2 = m.a2.value
            m.a1.value = a1
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.incumbent_objective, m.y.value)
            if res.objective_bound is None:
                bound = -math.inf
            else:
                bound = res.objective_bound
            self.assertTrue(bound <= m.y.value)
            duals = res.solution_loader.get_duals()
            self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
            self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_equality(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        check_duals = True
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
                check_duals = False
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(mutable=True)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y == m.a1 * m.x + m.b1)
        m.c2 = pe.Constraint(expr=m.y == m.a2 * m.x + m.b2)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.incumbent_objective, m.y.value)
            if res.objective_bound is None:
                bound = -math.inf
            else:
                bound = res.objective_bound
            self.assertTrue(bound <= m.y.value)
            if check_duals:
                duals = res.solution_loader.get_duals()
                self.assertAlmostEqual(duals[m.c1], (1 + a1 / (a2 - a1)))
                self.assertAlmostEqual(duals[m.c2], -a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_linear_expression(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(mutable=True)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.y)
        e = LinearExpression(
            constant=m.b1, linear_coefs=[-1, m.a1], linear_vars=[m.y, m.x]
        )
        m.c1 = pe.Constraint(expr=e == 0)
        e = LinearExpression(
            constant=m.b2, linear_coefs=[-1, m.a2], linear_vars=[m.y, m.x]
        )
        m.c2 = pe.Constraint(expr=e == 0)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(res.incumbent_objective, m.y.value)
            if res.objective_bound is None:
                bound = -math.inf
            else:
                bound = res.objective_bound
            self.assertTrue(bound <= m.y.value)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_no_objective(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        check_duals = True
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                check_duals = False
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(mutable=True)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.c1 = pe.Constraint(expr=m.y == m.a1 * m.x + m.b1)
        m.c2 = pe.Constraint(expr=m.y == m.a2 * m.x + m.b2)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res: Results = opt.solve(m)
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertEqual(res.incumbent_objective, None)
            self.assertEqual(res.objective_bound, None)
            if check_duals:
                duals = res.solution_loader.get_duals()
                self.assertAlmostEqual(duals[m.c1], 0)
                self.assertAlmostEqual(duals[m.c2], 0)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_add_remove_cons(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        a1 = -1
        a2 = 1
        b1 = 1
        b2 = 2
        a3 = 1
        b3 = 3
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= a1 * m.x + b1)
        m.c2 = pe.Constraint(expr=m.y >= a2 * m.x + b2)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(res.incumbent_objective, m.y.value)
        if res.objective_bound is None:
            bound = -math.inf
        else:
            bound = res.objective_bound
        self.assertTrue(bound <= m.y.value)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
        self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

        m.c3 = pe.Constraint(expr=m.y >= a3 * m.x + b3)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, (b3 - b1) / (a1 - a3))
        self.assertAlmostEqual(m.y.value, a1 * (b3 - b1) / (a1 - a3) + b1)
        self.assertAlmostEqual(res.incumbent_objective, m.y.value)
        self.assertTrue(res.objective_bound is None or res.objective_bound <= m.y.value)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a3 - a1)))
        self.assertAlmostEqual(duals[m.c2], 0)
        self.assertAlmostEqual(duals[m.c3], a1 / (a3 - a1))

        del m.c3
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(res.incumbent_objective, m.y.value)
        self.assertTrue(res.objective_bound is None or res.objective_bound <= m.y.value)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
        self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_results_infeasible(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y <= m.x - 1)
        with self.assertRaises(Exception):
            res = opt.solve(m)
        opt.config.load_solutions = False
        opt.config.raise_exception_on_nonoptimal_result = False
        res = opt.solve(m)
        self.assertNotEqual(res.solution_status, SolutionStatus.optimal)
        if isinstance(opt, Ipopt):
            acceptable_termination_conditions = {
                TerminationCondition.locallyInfeasible,
                TerminationCondition.unbounded,
            }
        else:
            acceptable_termination_conditions = {
                TerminationCondition.provenInfeasible,
                TerminationCondition.infeasibleOrUnbounded,
            }
        self.assertIn(res.termination_condition, acceptable_termination_conditions)
        self.assertAlmostEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, None)
        self.assertTrue(res.incumbent_objective is None)

        if not isinstance(opt, Ipopt):
            # ipopt can return the values of the variables/duals at the last iterate
            # even if it did not converge; raise_exception_on_nonoptimal_result
            # is set to False, so we are free to load infeasible solutions
            with self.assertRaisesRegex(
                RuntimeError, '.*does not currently have a valid solution.*'
            ):
                res.solution_loader.load_vars()
            with self.assertRaisesRegex(
                RuntimeError, '.*does not currently have valid duals.*'
            ):
                res.solution_loader.get_duals()
            with self.assertRaisesRegex(
                RuntimeError, '.*does not currently have valid reduced costs.*'
            ):
                res.solution_loader.get_reduced_costs()

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_duals(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y - m.x >= 0)
        m.c2 = pe.Constraint(expr=m.y + m.x - 2 >= 0)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], 0.5)
        self.assertAlmostEqual(duals[m.c2], 0.5)

        duals = res.solution_loader.get_duals(cons_to_load=[m.c1])
        self.assertAlmostEqual(duals[m.c1], 0.5)
        self.assertNotIn(m.c2, duals)

    @parameterized.expand(input=_load_tests(qcp_solvers))
    def test_mutable_quadratic_coefficient(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a = pe.Param(initialize=1, mutable=True)
        m.b = pe.Param(initialize=-1, mutable=True)
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.y >= (m.a * m.x + m.b) ** 2)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.41024548525899274, 4)
        self.assertAlmostEqual(m.y.value, 0.34781038127030117, 4)
        m.a.value = 2
        m.b.value = -0.5
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.10256137418973625, 4)
        self.assertAlmostEqual(m.y.value, 0.0869525991355825, 4)

    @parameterized.expand(input=_load_tests(qcp_solvers))
    def test_mutable_quadratic_objective(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a = pe.Param(initialize=1, mutable=True)
        m.b = pe.Param(initialize=-1, mutable=True)
        m.c = pe.Param(initialize=1, mutable=True)
        m.d = pe.Param(initialize=1, mutable=True)
        m.obj = pe.Objective(expr=m.x**2 + m.c * m.y**2 + m.d * m.x)
        m.ccon = pe.Constraint(expr=m.y >= (m.a * m.x + m.b) ** 2)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.2719178742733325, 4)
        self.assertAlmostEqual(m.y.value, 0.5301035741688002, 4)
        m.c.value = 3.5
        m.d.value = -1
        res = opt.solve(m)

        self.assertAlmostEqual(m.x.value, 0.6962249634573562, 4)
        self.assertAlmostEqual(m.y.value, 0.09227926676152151, 4)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_fixed_vars(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        for treat_fixed_vars_as_params in [True, False]:
            opt: SolverBase = opt_class()
            if opt.is_persistent():
                opt.config.auto_updates.treat_fixed_vars_as_params = (
                    treat_fixed_vars_as_params
                )
            if not opt.available():
                raise unittest.SkipTest(f'Solver {opt.name} not available.')
            if any(name.startswith(i) for i in nl_solvers_set):
                if use_presolve:
                    opt.config.writer_config.linear_presolve = True
                else:
                    opt.config.writer_config.linear_presolve = False
            m = pe.ConcreteModel()
            m.x = pe.Var()
            m.x.fix(0)
            m.y = pe.Var()
            a1 = 1
            a2 = -1
            b1 = 1
            b2 = 2
            m.obj = pe.Objective(expr=m.y)
            m.c1 = pe.Constraint(expr=m.y >= a1 * m.x + b1)
            m.c2 = pe.Constraint(expr=m.y >= a2 * m.x + b2)
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

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_fixed_vars_2(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = True
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.x.fix(0)
        m.y = pe.Var()
        a1 = 1
        a2 = -1
        b1 = 1
        b2 = 2
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= a1 * m.x + b1)
        m.c2 = pe.Constraint(expr=m.y >= a2 * m.x + b2)
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

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_fixed_vars_3(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = True
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c1 = pe.Constraint(expr=m.x == 2 / m.y)
        m.y.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 3)
        self.assertAlmostEqual(m.x.value, 2)

    @parameterized.expand(input=_load_tests(nlp_solvers))
    def test_fixed_vars_4(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = True
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.x == 2 / m.y)
        m.y.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        m.y.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2**0.5)
        self.assertAlmostEqual(m.y.value, 2**0.5)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_mutable_param_with_range(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(initialize=0, mutable=True)
        m.a2 = pe.Param(initialize=0, mutable=True)
        m.b1 = pe.Param(initialize=0, mutable=True)
        m.b2 = pe.Param(initialize=0, mutable=True)
        m.c1 = pe.Param(initialize=0, mutable=True)
        m.c2 = pe.Param(initialize=0, mutable=True)
        m.obj = pe.Objective(expr=m.y)
        m.con1 = pe.Constraint(expr=(m.b1, m.y - m.a1 * m.x, m.c1))
        m.con2 = pe.Constraint(expr=(m.b2, m.y - m.a2 * m.x, m.c2))

        np.random.seed(0)
        params_to_test = [
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pe.minimize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pe.maximize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pe.minimize,
            ),
            (
                np.random.uniform(0, 10),
                np.random.uniform(-10, 0),
                np.random.uniform(-5, 2.5),
                np.random.uniform(-5, 2.5),
                np.random.uniform(2.5, 10),
                np.random.uniform(2.5, 10),
                pe.maximize,
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
            self.assertEqual(res.solution_status, SolutionStatus.optimal)
            if sense is pe.minimize:
                self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2), 6)
                self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1, 6)
                self.assertAlmostEqual(res.incumbent_objective, m.y.value, 6)
                self.assertTrue(
                    res.objective_bound is None
                    or res.objective_bound <= m.y.value + 1e-12
                )
                duals = res.solution_loader.get_duals()
                self.assertAlmostEqual(duals[m.con1], (1 + a1 / (a2 - a1)), 6)
                self.assertAlmostEqual(duals[m.con2], -a1 / (a2 - a1), 6)
            else:
                self.assertAlmostEqual(m.x.value, (c2 - c1) / (a1 - a2), 6)
                self.assertAlmostEqual(m.y.value, a1 * (c2 - c1) / (a1 - a2) + c1, 6)
                self.assertAlmostEqual(res.incumbent_objective, m.y.value, 6)
                self.assertTrue(
                    res.objective_bound is None
                    or res.objective_bound >= m.y.value - 1e-12
                )
                duals = res.solution_loader.get_duals()
                self.assertAlmostEqual(duals[m.con1], (1 + a1 / (a2 - a1)), 6)
                self.assertAlmostEqual(duals[m.con2], -a1 / (a2 - a1), 6)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_add_and_remove_vars(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.y = pe.Var(bounds=(-1, None))
        m.obj = pe.Objective(expr=m.y)
        if opt.is_persistent():
            opt.config.auto_updates.update_parameters = False
            opt.config.auto_updates.update_vars = False
            opt.config.auto_updates.update_constraints = False
            opt.config.auto_updates.update_named_expressions = False
            opt.config.auto_updates.check_for_new_or_removed_params = False
            opt.config.auto_updates.check_for_new_or_removed_constraints = False
            opt.config.auto_updates.check_for_new_or_removed_vars = False
        opt.config.load_solutions = False
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.y.value, -1)
        m.x = pe.Var()
        a1 = 1
        a2 = -1
        b1 = 2
        b2 = 1
        m.c1 = pe.Constraint(expr=(0, m.y - a1 * m.x - b1, None))
        m.c2 = pe.Constraint(expr=(None, -m.y + a2 * m.x + b2, 0))
        if opt.is_persistent():
            opt.add_constraints([m.c1, m.c2])
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        m.c1.deactivate()
        m.c2.deactivate()
        if opt.is_persistent():
            opt.remove_constraints([m.c1, m.c2])
        m.x.value = None
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        res.solution_loader.load_vars()
        self.assertEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, -1)

    @parameterized.expand(input=_load_tests(nlp_solvers))
    def test_exp(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.42630274815985264)
        self.assertAlmostEqual(m.y.value, 0.6529186341994245)

    @parameterized.expand(input=_load_tests(nlp_solvers))
    def test_log(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=1)
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y <= pe.log(m.x))
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.6529186341994245)
        self.assertAlmostEqual(m.y.value, -0.42630274815985264)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_with_numpy(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        a1 = 1
        b1 = 3
        a2 = -2
        b2 = 1
        m.c1 = pe.Constraint(
            expr=(np.float64(0), m.y - np.int64(1) * m.x - np.float32(3), None)
        )
        m.c2 = pe.Constraint(
            expr=(None, -m.y + np.int32(-2) * m.x + np.float64(1), np.float16(0))
        )
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_bounds_with_params(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.y = pe.Var()
        m.p = pe.Param(mutable=True)
        m.y.setlb(m.p)
        m.p.value = 1
        m.obj = pe.Objective(expr=m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 1)
        m.p.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, -1)
        m.y.setlb(None)
        m.y.setub(m.p)
        m.obj.sense = pe.maximize
        m.p.value = 5
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 5)
        m.p.value = 4
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 4)
        m.y.setub(None)
        m.y.setlb(m.p)
        m.obj.sense = pe.minimize
        m.p.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 3)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_solution_loader(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, None))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=(0, m.y - m.x, None))
        m.c2 = pe.Constraint(expr=(0, m.y - m.x + 1, None))
        opt.config.load_solutions = False
        res = opt.solve(m)
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.y.value)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        m.x.value = None
        m.y.value = None
        res.solution_loader.load_vars([m.y])
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

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_time_limit(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        from sys import platform

        if platform == 'win32':
            raise unittest.SkipTest

        N = 30
        m = pe.ConcreteModel()
        m.jobs = pe.Set(initialize=list(range(N)))
        m.tasks = pe.Set(initialize=list(range(N)))
        m.x = pe.Var(m.jobs, m.tasks, bounds=(0, 1))

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
        m.obj = pe.Objective(expr=obj_expr, sense=pe.maximize)

        m.c1 = pe.Constraint(m.jobs)
        m.c2 = pe.Constraint(m.tasks)
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
        if isinstance(opt, Ipopt):
            opt.config.time_limit = 1e-6
        else:
            opt.config.time_limit = 0
        opt.config.load_solutions = False
        opt.config.raise_exception_on_nonoptimal_result = False
        res = opt.solve(m)
        self.assertIn(
            res.termination_condition,
            {TerminationCondition.maxTimeLimit, TerminationCondition.iterationLimit},
        )

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_objective_changes(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        m.obj = pe.Objective(expr=m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        del m.obj
        m.obj = pe.Objective(expr=2 * m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 2)
        m.obj.expr = 3 * m.y
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 3)
        m.obj.sense = pe.maximize
        opt.config.raise_exception_on_nonoptimal_result = False
        opt.config.load_solutions = False
        res = opt.solve(m)
        self.assertIn(
            res.termination_condition,
            {
                TerminationCondition.unbounded,
                TerminationCondition.infeasibleOrUnbounded,
            },
        )
        m.obj.sense = pe.minimize
        opt.config.load_solutions = True
        del m.obj
        m.obj = pe.Objective(expr=m.x * m.y)
        m.x.fix(2)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 6, 6)
        m.x.fix(3)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 12, 6)
        m.x.unfix()
        m.y.fix(2)
        m.x.setlb(-3)
        m.x.setub(5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -2, 6)
        m.y.unfix()
        m.x.setlb(None)
        m.x.setub(None)
        m.e = pe.Expression(expr=2)
        del m.obj
        m.obj = pe.Objective(expr=m.e * m.y)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 2)
        m.e.expr = 3
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 3)
        if opt.is_persistent():
            opt.config.auto_updates.check_for_new_objective = False
            m.e.expr = 4
            res = opt.solve(m)
            self.assertAlmostEqual(res.incumbent_objective, 4)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_domain(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, None), domain=pe.NonNegativeReals)
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        m.x.setlb(-1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        m.x.setlb(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        m.x.setlb(-1)
        m.x.domain = pe.Reals
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -1)
        m.x.domain = pe.NonNegativeReals
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)

    @parameterized.expand(input=_load_tests(mip_solvers))
    def test_domain_with_integers(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, None), domain=pe.NonNegativeIntegers)
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        m.x.setlb(0.5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        m.x.setlb(-5.5)
        m.x.domain = pe.Integers
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -5)
        m.x.domain = pe.Binary
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        m.x.setlb(0.5)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_fixed_binaries(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.Binary)
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c = pe.Constraint(expr=m.y >= m.x)
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        m.x.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)

        opt: SolverBase = opt_class()
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = False
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        m.x.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)

    @parameterized.expand(input=_load_tests(mip_solvers))
    def test_with_gdp(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var(bounds=(-10, 10))
        m.obj = pe.Objective(expr=m.y)
        m.d1 = gdp.Disjunct()
        m.d1.c1 = pe.Constraint(expr=m.y >= m.x + 2)
        m.d1.c2 = pe.Constraint(expr=m.y >= -m.x + 2)
        m.d2 = gdp.Disjunct()
        m.d2.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.d2.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        m.disjunction = gdp.Disjunction(expr=[m.d2, m.d1])
        pe.TransformationFactory("gdp.bigm").apply_to(m)

        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt: SolverBase = opt_class()
        opt.use_extensions = True
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_variables_elsewhere(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False

        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.b = pe.Block()
        m.b.obj = pe.Objective(expr=m.y)
        m.b.c1 = pe.Constraint(expr=m.y >= m.x + 2)
        m.b.c2 = pe.Constraint(expr=m.y >= -m.x)

        res = opt.solve(m.b)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.y.value, 1)

        m.x.setlb(0)
        res = opt.solve(m.b)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 2)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_variables_elsewhere2(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False

        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()

        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y >= -m.x)
        m.c3 = pe.Constraint(expr=m.y >= m.z + 1)
        m.c4 = pe.Constraint(expr=m.y >= -m.z + 1)

        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        sol = res.solution_loader.get_primals()
        self.assertIn(m.x, sol)
        self.assertIn(m.y, sol)
        self.assertIn(m.z, sol)

        del m.c3
        del m.c4
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        sol = res.solution_loader.get_primals()
        self.assertIn(m.x, sol)
        self.assertIn(m.y, sol)
        self.assertNotIn(m.z, sol)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_bug_1(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(3, 7))
        m.y = pe.Var(bounds=(-10, 10))
        m.p = pe.Param(mutable=True, initialize=0)

        m.obj = pe.Objective(expr=m.y)
        m.c = pe.Constraint(expr=m.y >= m.p * m.x)

        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 0)

        m.p.value = 1
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(res.incumbent_objective, 3)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_bug_2(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        """
        This test is for a bug where an objective containing a fixed variable does
        not get updated properly when the variable is unfixed.
        """
        for fixed_var_option in [True, False]:
            opt: SolverBase = opt_class()
            if not opt.available():
                raise unittest.SkipTest(f'Solver {opt.name} not available.')
            if any(name.startswith(i) for i in nl_solvers_set):
                if use_presolve:
                    opt.config.writer_config.linear_presolve = True
                else:
                    opt.config.writer_config.linear_presolve = False
            if opt.is_persistent():
                opt.config.auto_updates.treat_fixed_vars_as_params = fixed_var_option

            m = pe.ConcreteModel()
            m.x = pe.Var(bounds=(-10, 10))
            m.y = pe.Var()
            m.obj = pe.Objective(expr=3 * m.y - m.x)
            m.c = pe.Constraint(expr=m.y >= m.x)

            m.x.fix(1)
            res = opt.solve(m)
            self.assertAlmostEqual(res.incumbent_objective, 2, 5)

            m.x.unfix()
            m.x.setlb(-9)
            m.x.setub(9)
            res = opt.solve(m)
            self.assertAlmostEqual(res.incumbent_objective, -18, 5)

    @parameterized.expand(input=_load_tests(nl_solvers))
    def test_presolve_with_zero_coef(
        self, name: str, opt_class: Type[SolverBase], use_presolve: bool
    ):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False

        """
        when c2 gets presolved out, c1 becomes 
        x - y + y = 0 which becomes
        x - 0*y == 0 which is the zero we are testing for
        """
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2 + m.z**2)
        m.c1 = pe.Constraint(expr=m.x == m.y + m.z + 1.5)
        m.c2 = pe.Constraint(expr=m.z == -m.y)

        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 2.25)
        self.assertAlmostEqual(m.x.value, 1.5)
        self.assertAlmostEqual(m.y.value, 0)
        self.assertAlmostEqual(m.z.value, 0)

        m.x.setlb(2)
        res = opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        if use_presolve:
            exp = TerminationCondition.provenInfeasible
        else:
            exp = TerminationCondition.locallyInfeasible
        self.assertEqual(res.termination_condition, exp)

        m = pe.ConcreteModel()
        m.w = pe.Var()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2 + m.z**2 + m.w**2)
        m.c1 = pe.Constraint(expr=m.x + m.w == m.y + m.z)
        m.c2 = pe.Constraint(expr=m.z == -m.y)
        m.c3 = pe.Constraint(expr=m.x == -m.w)

        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 0)
        self.assertAlmostEqual(m.w.value, 0)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0)
        self.assertAlmostEqual(m.z.value, 0)

        del m.c1
        m.c1 = pe.Constraint(expr=m.x + m.w == m.y + m.z + 1.5)
        res = opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        self.assertEqual(res.termination_condition, exp)

    @parameterized.expand(input=_load_tests(all_solvers))
    def test_scaling(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        check_duals = True
        if any(name.startswith(i) for i in nl_solvers_set):
            if use_presolve:
                check_duals = False
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False

        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= (m.x - 1) + 1)
        m.c2 = pe.Constraint(expr=m.y >= -(m.x - 1) + 1)
        m.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
        m.scaling_factor[m.x] = 0.5
        m.scaling_factor[m.y] = 2
        m.scaling_factor[m.c1] = 0.5
        m.scaling_factor[m.c2] = 2
        m.scaling_factor[m.obj] = 2

        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        primals = res.solution_loader.get_primals()
        self.assertAlmostEqual(primals[m.x], 1)
        self.assertAlmostEqual(primals[m.y], 1)
        if check_duals:
            duals = res.solution_loader.get_duals()
            self.assertAlmostEqual(duals[m.c1], -0.5)
            self.assertAlmostEqual(duals[m.c2], -0.5)
            rc = res.solution_loader.get_reduced_costs()
            self.assertAlmostEqual(rc[m.x], 0)
            self.assertAlmostEqual(rc[m.y], 0)

        m.x.setlb(2)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 2)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 2)
        primals = res.solution_loader.get_primals()
        self.assertAlmostEqual(primals[m.x], 2)
        self.assertAlmostEqual(primals[m.y], 2)
        if check_duals:
            duals = res.solution_loader.get_duals()
            self.assertAlmostEqual(duals[m.c1], -1)
            self.assertAlmostEqual(duals[m.c2], 0)
            rc = res.solution_loader.get_reduced_costs()
            self.assertAlmostEqual(rc[m.x], 1)
            self.assertAlmostEqual(rc[m.y], 0)


class TestLegacySolverInterface(unittest.TestCase):
    @parameterized.expand(input=all_solvers)
    def test_param_updates(self, name: str, opt_class: Type[SolverBase]):
        opt = pe.SolverFactory(name + '_v2')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.a1 = pe.Param(mutable=True)
        m.a2 = pe.Param(mutable=True)
        m.b1 = pe.Param(mutable=True)
        m.b2 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=(0, m.y - m.a1 * m.x - m.b1, None))
        m.c2 = pe.Constraint(expr=(None, -m.y + m.a2 * m.x + m.b2, 0))
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
        for a1, a2, b1, b2 in params_to_test:
            m.a1.value = a1
            m.a2.value = a2
            m.b1.value = b1
            m.b2.value = b2
            res = opt.solve(m)
            pe.assert_optimal_termination(res)
            self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
            self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
            self.assertAlmostEqual(m.dual[m.c1], (1 + a1 / (a2 - a1)))
            self.assertAlmostEqual(m.dual[m.c2], a1 / (a2 - a1))

    @parameterized.expand(input=all_solvers)
    def test_load_solutions(self, name: str, opt_class: Type[SolverBase]):
        opt = pe.SolverFactory(name + '_v2')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.obj = pe.Objective(expr=m.x)
        m.c = pe.Constraint(expr=(-1, m.x, 1))
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        res = opt.solve(m, load_solutions=False)
        pe.assert_optimal_termination(res)
        self.assertIsNone(m.x.value)
        self.assertNotIn(m.c, m.dual)
        m.solutions.load_from(res)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.dual[m.c], 1)
