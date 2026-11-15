# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver

from pyomo.environ import (
    SolverFactory,
    Objective,
    maximize,
    TerminationCondition,
    value,
    Var,
    Integers,
    Constraint,
    ConcreteModel,
)
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models


class _ExpiredEnumerationSolver(GDP_Enumeration_Solver):
    """Enumeration solver that enters its GDP solve after the time limit."""

    def _solve_gdp(self, original_model, config):
        """Expire the active timer before running the real GDP solve."""
        self.timing.main_timer_start_time -= config.time_limit
        return super()._solve_gdp(original_model, config)

    def _discrete_solution_iterator(self, *args):
        """Fail instead of requesting a discrete solution."""
        raise AssertionError('discrete solutions were enumerated')


class TestGDPoptEnumerateUnit(unittest.TestCase):
    def test_large_space_not_enumerated_after_time_limit(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1))
        m.i = Var(domain=Integers, bounds=(0.5, 10**20))
        m.obj = Objective(expr=m.x + m.i)

        def disjunction_rule(m, _):
            return [[m.x <= 0], [m.x >= 0]]

        m.disjunctions = Disjunction(range(32), rule=disjunction_rule)
        solver = _ExpiredEnumerationSolver()
        results = solver.solve(m, force_subproblem_nlp=True, time_limit=1)

        self.assertEqual(solver.num_discrete_solns, 2**32 * 10**20)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxTimeLimit
        )

    def test_noninteger_integer_bounds(self):
        m = ConcreteModel()
        m.i = Var(domain=Integers, bounds=(0.5, 3.5))
        solver = GDP_Enumeration_Solver(force_subproblem_nlp=True)

        solutions = solver._discrete_solution_iterator([], [], [m.i], solver.config)

        self.assertEqual([solution[2] for solution in solutions], [(1,), (2,), (3,)])

    def test_unbounded_integer_variable(self):
        m = ConcreteModel()
        m.i = Var(domain=Integers)
        m.disjunction = Disjunction(expr=[[m.i >= 0], [m.i <= 0]])
        m.obj = Objective(expr=m.i)

        with self.assertRaisesRegex(ValueError, 'finite bounds'):
            GDP_Enumeration_Solver().solve(m, force_subproblem_nlp=True)

    def test_completion(self):
        m = ConcreteModel()
        m.disjunction = Disjunction(
            expr=[[Constraint.Infeasible], [Constraint.Infeasible]]
        )
        m.obj = Objective(expr=0)

        results = GDP_Enumeration_Solver().solve(m, iterlim=2)

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )


@unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi not available')
@unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'Gurobi not licensed')
class TestGDPoptEnumerate(unittest.TestCase):
    def test_solve_two_term_disjunction(self):
        m = models.makeTwoTermDisj()
        m.obj = Objective(expr=m.x, sense=maximize)

        results = SolverFactory('gdpopt.enumerate').solve(m)

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 9)
        self.assertEqual(results.problem.upper_bound, 9)

        self.assertEqual(value(m.x), 9)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def modify_two_term_disjunction(self, m):
        # Make first disjunct feasible
        m.a.setlb(0)
        # Discrete variable
        m.y = Var(domain=Integers, bounds=(2, 4))
        m.d[1].c3 = Constraint(expr=m.x <= 6)
        m.d[0].c2 = Constraint(expr=m.y + m.a - 5 <= 2)

        m.obj = Objective(expr=-m.x - m.y)

    def test_solve_GDP_iterate_over_discrete_variables(self):
        m = models.makeTwoTermDisj()
        self.modify_two_term_disjunction(m)

        results = SolverFactory('gdpopt.enumerate').solve(m, force_subproblem_nlp=True)

        self.assertEqual(results.solver.iterations, 6)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, -11)
        self.assertEqual(results.problem.upper_bound, -11)

        self.assertEqual(value(m.x), 9)
        self.assertEqual(value(m.y), 2)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def test_solve_GDP_do_not_iterate_over_discrete_variables(self):
        m = models.makeTwoTermDisj()
        self.modify_two_term_disjunction(m)

        results = SolverFactory('gdpopt.enumerate').solve(m)

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, -11)
        self.assertEqual(results.problem.upper_bound, -11)

        self.assertEqual(value(m.x), 9)
        self.assertEqual(value(m.y), 2)
        self.assertTrue(value(m.d[0].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))

    def test_solve_GDP_iterate_over_Boolean_variables(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(m, force_subproblem_nlp=True)

        self.assertEqual(results.solver.iterations, 16)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 8)
        self.assertEqual(results.problem.upper_bound, 8)

        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))
        self.assertEqual(value(m.x), 8)
        # We don't know what values they take, but they have to be different
        self.assertNotEqual(value(m.Y[1]), value(m.Y[2]))

    def test_solve_GDP_do_not_iterate_over_Boolean_variables(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(m)

        self.assertEqual(results.solver.iterations, 4)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(results.problem.lower_bound, 8)
        self.assertEqual(results.problem.upper_bound, 8)

        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))
        self.assertEqual(value(m.x), 8)
        # We don't know what values they take, but they have to be different
        self.assertNotEqual(value(m.Y[1]), value(m.Y[2]))

    def test_stop_at_iteration_limit(self):
        m = models.makeLogicalConstraintsOnDisjuncts()

        results = SolverFactory('gdpopt.enumerate').solve(
            m, iterlim=4, force_subproblem_nlp=True
        )

        self.assertEqual(results.solver.iterations, 4)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxIterations
        )

    def test_unbounded_GDP(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.z)

        results = SolverFactory('gdpopt.enumerate').solve(m)

        self.assertEqual(results.solver.iterations, 1)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.unbounded
        )
        self.assertEqual(results.problem.lower_bound, -float('inf'))
        self.assertEqual(results.problem.upper_bound, -float('inf'))


@unittest.skipUnless(SolverFactory('ipopt').available(), 'Ipopt not available')
class TestGDPoptEnumerate_ipopt_tests(unittest.TestCase):
    def test_infeasible_GDP(self):
        m = models.make_infeasible_gdp_model()

        results = SolverFactory('gdpopt.enumerate').solve(m)

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(results.problem.lower_bound, float('inf'))

    def test_algorithm_specified_to_solve(self):
        m = models.twoDisj_twoCircles_easy()

        results = SolverFactory('gdpopt').solve(m, algorithm='enumerate', tee=True)

        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertAlmostEqual(results.problem.lower_bound, 9)
        self.assertAlmostEqual(results.problem.upper_bound, 9)

        self.assertAlmostEqual(value(m.x), 2)
        self.assertAlmostEqual(value(m.y), 7)
        self.assertTrue(value(m.upper_circle.indicator_var))
        self.assertFalse(value(m.lower_circle.indicator_var))
