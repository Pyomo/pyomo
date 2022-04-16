#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for the GDPopt solver plugin."""

from io import StringIO
import logging
from math import fabs
from os.path import join, normpath

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.loa import GDP_LOA_Solver
from pyomo.contrib.gdpopt.mip_solve import solve_MILP_master_problem
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.environ import ( ConcreteModel, Objective, SolverFactory, Var, value,
                            Integers, Block, Constraint, maximize,
                            LogicalConstraint, sqrt, RangeSet)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition

exdir = normpath(join(PYOMO_ROOT_DIR, 'examples', 'gdp'))

mip_solver = 'glpk'
nlp_solver = 'ipopt'
global_nlp_solver = 'baron'
global_nlp_solver_args = dict()
minlp_solver = 'baron'
LOA_solvers = (mip_solver, nlp_solver)
GLOA_solvers = (mip_solver, global_nlp_solver, minlp_solver)
LOA_solvers_available = all(SolverFactory(s).available() for s in LOA_solvers)
GLOA_solvers_available = all(SolverFactory(s).available() for s in GLOA_solvers)
license_available = SolverFactory(global_nlp_solver).license_is_valid() if \
                    GLOA_solvers_available else False


class TestGDPoptUnit(unittest.TestCase):
    """Real unit tests for GDPopt"""

    @unittest.skipUnless(SolverFactory(mip_solver).available(),
                         "MIP solver not available")
    def test_solve_master_problem_unbounded(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        # Include a disjunction so that we don't default to just a MIP solver
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.z)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0],
                                        m.d._autodisjuncts[1]]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            timing = Bunch()
            with time_code(timing, 'main', is_main_timer=True):
                tc = solve_MILP_master_problem(
                    m.GDPopt_utils,
                    GDP_LOA_Solver.CONFIG(dict(mip_solver=mip_solver)),
                    timing)
            self.assertIn("Master problem was unbounded. Re-solving with "
                          "arbitrary bound values", output.getvalue().strip())
        self.assertIs(tc, TerminationCondition.unbounded)

    @unittest.skipUnless(SolverFactory(mip_solver).available(),
                         "MIP solver not available")
    def test_solve_lp(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver=mip_solver)
            self.assertIn("Your model is an LP (linear program).",
                          output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)

            self.assertEqual(results.problem.number_of_binary_variables, 0)
            self.assertEqual(results.problem.number_of_integer_variables, 0)
            self.assertEqual(results.problem.number_of_disjunctions, 0)
            self.assertAlmostEqual(results.problem.lower_bound, 1)
            self.assertAlmostEqual(results.problem.upper_bound, 1)

    @unittest.skipUnless(SolverFactory('gurobi').available(),
                         'Gurobi not available')
    def test_solve_nlp(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=m.x ** 2)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                m, nlp_solver='gurobi')
            self.assertIn("Your model is an NLP (nonlinear program).",
                          output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)

            self.assertEqual(results.problem.number_of_binary_variables, 0)
            self.assertEqual(results.problem.number_of_integer_variables, 0)
            self.assertEqual(results.problem.number_of_disjunctions, 0)
            self.assertAlmostEqual(results.problem.lower_bound, 1)
            self.assertAlmostEqual(results.problem.upper_bound, 1)

    @unittest.skipUnless(SolverFactory(mip_solver).available(),
                         "MIP solver not available")
    def test_solve_constant_obj(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=1)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver=mip_solver)
            self.assertIn("Your model is an LP (linear program).",
                          output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)

    @unittest.skipUnless(SolverFactory(nlp_solver).available(),
                         'NLP solver not available')
    def test_no_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x ** 2 >= 1)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            SolverFactory('gdpopt', algorithm='LOA').solve(
                m, nlp_solver=nlp_solver)
            self.assertIn("Model has no active objectives. Adding dummy "
                          "objective.", output.getvalue().strip())

    def test_multiple_objectives(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective(expr=m.x)
        m.o2 = Objective(expr=m.x + 1)
        with self.assertRaisesRegex(ValueError, "Model has multiple active "
                                    "objectives"):
            SolverFactory('gdpopt', algorithm='LOA').solve(m)

    def test_is_feasible_function(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 3), initialize=2)
        m.c = Constraint(expr=m.x == 2)
        self.assertTrue(
            is_feasible(m, GDP_LOA_Solver.CONFIG()))

        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(
            is_feasible(m, GDP_LOA_Solver.CONFIG()))

        m = ConcreteModel()
        m.x = Var(bounds=(0, 3), initialize=2)
        m.c = Constraint(expr=m.x >= 5)
        self.assertFalse(
            is_feasible(m, GDP_LOA_Solver.CONFIG()))

        m = ConcreteModel()
        m.x = Var(bounds=(3, 3), initialize=2)
        self.assertFalse(
            is_feasible(m, GDP_LOA_Solver.CONFIG()))

        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=2)
        self.assertFalse(
            is_feasible(m, GDP_LOA_Solver.CONFIG()))

        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=2)
        m.d = Disjunct()
        with self.assertRaisesRegex(NotImplementedError,
                                    "Found active disjunct"):
            is_feasible(m, GDP_LOA_Solver.CONFIG())

    def test_invalid_solver_name(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError,
                                    ".*Expected a valid name for a solver. "
                                    "Received 'grubi'"):
            SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver='grubi')

@unittest.skipIf(not LOA_solvers_available,
                 "Required subsolvers %s are not available"
                 % (LOA_solvers,))
class TestGDPopt(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn("Set covering problem was infeasible.",
                          output.getvalue().strip())

        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.infeasible)

        # Test maximization problem infeasibility also
        m.o.sense = maximize
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver=mip_solver, nlp_solver=nlp_solver,
                init_strategy='no_init')
            self.assertIn("GDPopt exiting--problem is infeasible.",
                          output.getvalue().strip())

        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.infeasible)

    @unittest.skipUnless(SolverFactory(mip_solver).available(),
                         "MIP solver not available")
    def test_unbounded_gdp_minimization(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.z)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0],
                                        m.d._autodisjuncts[1]]
        results = SolverFactory('gdpopt', algorithm='LOA').solve(
            m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.unbounded)

    @unittest.skipUnless(SolverFactory(mip_solver).available(),
                         "MIP solver not available")
    def test_unbounded_gdp_maximization(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[
            [m.x + m.y <= 5], [m.x - m.y >= 3]
        ])
        m.o = Objective(expr=m.z, sense=maximize)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0],
                                        m.d._autodisjuncts[1]]
        results = SolverFactory('gdpopt', algorithm='LOA').solve(
            m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.unbounded)

    def test_GDP_nonlinear_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.x ** 2)
        SolverFactory('gdpopt', algorithm='LOA').solve( m,
                                                        mip_solver=mip_solver,
                                                        nlp_solver=nlp_solver )
        self.assertAlmostEqual(value(m.o), 0)

        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=-m.x ** 2, sense=maximize)
        SolverFactory('gdpopt', algorithm='LOA').solve( m,
                                                        mip_solver=mip_solver,
                                                        nlp_solver=nlp_solver )
        self.assertAlmostEqual(value(m.o), 0)

    def test_nested_disjunctions_set_covering(self):
        # This test triggers the InfeasibleConstraintException in
        # deactivate_trivial_constraints in one of the subproblem solves during
        # initialization. This makes sure we get the correct answer anyway, as
        # there is a feasible solution.
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            m, mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            init_strategy='set_covering')
        self.assertAlmostEqual(value(m.x), sqrt(2)/2)
        self.assertAlmostEqual(value(m.y), sqrt(2)/2)
        self.assertTrue(value(m.disj.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertTrue(value(m.d1.indicator_var))
        self.assertFalse(value(m.d2.indicator_var))

    def test_equality_propagation_infeasibility_in_subproblems(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 10))
        m.y = Var(bounds=(-10, 10))
        m.disj = Disjunction(expr=[[m.x == m.y, m.y == 2],
                                   [m.y == 8],
                                   [m.x + m.y >= 4, m.y == m.x + 1]])
        m.cons = Constraint(expr=m.x == 3)
        m.obj = Objective(expr=m.x + m.y)
        SolverFactory('gdpopt', algorithm='RIC').solve(
            m, mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            init_strategy='set_covering')
        self.assertAlmostEqual(value(m.x), 3)
        self.assertAlmostEqual(value(m.y), 4)
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertFalse(value(m.disj.disjuncts[1].indicator_var))
        self.assertTrue(value(m.disj.disjuncts[2].indicator_var))

    def test_bound_infeasibility_in_subproblems(self):
        m = ConcreteModel()
        m.x = Var(bounds=(2,4))
        m.y = Var(bounds=(5,10))
        m.disj = Disjunction(expr=[[m.x == m.y, m.x + m.y >= 8],
                                   [m.x == 4]])
        m.obj = Objective(expr=m.x + m.y)
        logger = logging.getLogger('pyomo.contrib.gdpopt')
        logger.setLevel(logging.DEBUG)
        SolverFactory('gdpopt', algorithm='RIC').solve(
            m, mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            init_strategy='set_covering',
            tee=True, logger=logger)
        self.assertAlmostEqual(value(m.x), 4)
        self.assertAlmostEqual(value(m.y), 5)
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertTrue(value(m.disj.disjuncts[1].indicator_var))

    def test_subproblem_preprocessing_encounters_trivial_constraints(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.z = Var(bounds=(-10, 10))
        m.disjunction = Disjunction(expr=[[m.x == 0, m.z >= 4],
                                          [m.x + m.z <= 0]])
        m.cons = Constraint(expr=m.x*m.z <= 0)
        m.obj = Objective(expr=-m.z)
        m.disjunction.disjuncts[0].indicator_var.fix(True)
        m.disjunction.disjuncts[1].indicator_var.fix(False)
        SolverFactory('gdpopt',
                      algorithm='RIC').solve(m, mip_solver=mip_solver,
                                             nlp_solver=nlp_solver,
                                             init_strategy='fix_disjuncts')
        # The real test is that this doesn't throw an error when we preprocess
        # to solve the first subproblem (in the initialization). The nonlinear
        # constraint becomes trivial, which we need to make sure is handled
        # correctly.
        self.assertEqual(value(m.x), 0)
        self.assertEqual(value(m.z), 10)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[1].indicator_var))

    def make_convex_circle_and_circle_slice_disjunction(self):
        m = ConcreteModel()
        # x will only appear in nonlinear expressions
        m.x = Var(bounds=(-10, 18))
        m.y = Var(bounds=(0,7))
        m.obj = Objective(expr=m.x**2 + m.y)
        m.disjunction = Disjunction(expr=[[m.x**2 + m.y**2 <= 3, m.y >= 1],
                                          (m.x - 3)**2 + (m.y - 2)**2 <= 1])

        return m

    def test_some_vars_only_in_subproblem(self):
        # Even variables that only appear in nonlinear constraints will be
        # introduced to the master problem via the cuts, so we check that that
        # works out.
        m = self.make_convex_circle_and_circle_slice_disjunction()

        results = SolverFactory('gdpopt',
                                algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, 
                         TerminationCondition.optimal)
        self.assertAlmostEqual(results.problem.upper_bound, 1)
        self.assertAlmostEqual(value(m.x), 0)
        self.assertAlmostEqual(value(m.y), 1)

    def test_fixed_vars_honored(self):
        # Make sure that the algorithm doesn't stomp on user-fixed variables
        m = self.make_convex_circle_and_circle_slice_disjunction()

        # first, force ourselves into the suboptimal disjunct
        m.disjunction.disjuncts[0].indicator_var.fix(False)
        logger = logging.getLogger('pyomo.contrib.gdpopt')
        logger.setLevel(logging.DEBUG)
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver,
                                                       logger=logger)
        self.assertTrue(value(m.disjunction.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertTrue(m.disjunction.disjuncts[0].indicator_var.fixed)
        self.assertAlmostEqual(value(m.x), 2.029, places=3)
        self.assertAlmostEqual(value(m.y), 1.761, places=3)
        self.assertAlmostEqual(value(m.obj), 5.878, places=3)
        
        # Now, do it by fixing a continuous variable
        m.disjunction.disjuncts[0].indicator_var.fixed = False
        m.x.fix(3)
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertTrue(value(m.disjunction.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertEqual(value(m.x), 3)
        self.assertTrue(m.x.fixed)
        self.assertAlmostEqual(value(m.y), 1)
        self.assertAlmostEqual(value(m.obj), 10)

    def test_ignore_set_for_oa_cuts(self):
        m = self.make_convex_circle_and_circle_slice_disjunction()
        
        m.disjunction.disjuncts[1].GDPopt_ignore_OA = [
            m.disjunction.disjuncts[1].constraint[1]]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.DEBUG):
            SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn('OA cut addition for '
                          'disjunction_disjuncts[1].constraint[1] skipped '
                          'because it is in the ignore set.', 
                          output.getvalue().strip())
        # and the solution is optimal
        self.assertAlmostEqual(value(m.x), 0)
        self.assertAlmostEqual(value(m.y), 1)

    def test_reverse_numeric_differentiation_in_LOA(self):
        # Having more than 1000 variables in the same constraint expression will
        # trigger reverse numeric differentiation. We make sure that works with
        # a disjunction between a pair of hyperspheres in 1300-dimensional
        # space.
        m = ConcreteModel()
        m.s = RangeSet(1300)
        m.x = Var(m.s, bounds=(-10, 10))
        m.d1 = Disjunct()
        m.d1.hypersphere = Constraint(expr=sum(m.x[i]**2 for i in m.s) <= 1)
        m.d2 = Disjunct()
        m.d2.translated_hyper_sphere = Constraint(expr=sum((m.x[i] - i)**2 for i
                                                           in m.s) <= 1)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        # we'll go in the sphere centered at (0,0,...,0)
        m.obj = Objective(expr=sum(m.x[i] for i in m.s))

        results = SolverFactory('gdpopt', algorithm='LOA').solve(
            m, mip_solver=mip_solver, nlp_solver=nlp_solver)

        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertTrue(value(m.d1.indicator_var))
        self.assertFalse(value(m.d2.indicator_var))
        x_val = -sqrt(1300)/1300
        for x in m.x.values():
            self.assertAlmostEqual(value(x), x_val)
        self.assertAlmostEqual(results.problem.upper_bound, 1300*x_val,
                               places=6)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_logical_constraints_on_disjuncts_nonlinear_convex(self):
        m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver,
                                                       tee=True)
        self.assertAlmostEqual(value(m.x), 4)

    def test_nested_disjunctions_no_init(self):
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver,
                                                       init_strategy='no_init')
        self.assertAlmostEqual(value(m.x), sqrt(2)/2)
        self.assertAlmostEqual(value(m.y), sqrt(2)/2)

    def test_nested_disjunctions_max_binary(self):
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            m, mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            init_strategy='max_binary')
        self.assertAlmostEqual(value(m.x), sqrt(2)/2)
        self.assertAlmostEqual(value(m.y), sqrt(2)/2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt', algorithm='LOA').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_LOA_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_iteration_limit(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                eight_process,
                mip_solver=mip_solver,
                nlp_solver=nlp_solver,
                iterlim=2)
            self.assertIn("GDPopt unable to converge bounds within iteration "
                          "limit of 2 iterations.", output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.maxIterations)

    def test_time_limit(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                eight_process,
                mip_solver=mip_solver,
                nlp_solver=nlp_solver,
                time_limit=1)
            self.assertIn("GDPopt exiting--Did not converge bounds before "
                          "time limit of 1 seconds.", output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.maxTimeLimit)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_LOA_8PP_logical_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(SolverFactory('gams').available(exception_flag=False),
                         'GAMS solver not available')
    def test_LOA_8PP_gams_solver(self):
        # Make sure that the duals are still correct
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver='gams',
            max_slack=0,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_8PP_force_NLP(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            force_subproblem_nlp=True,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            strip_pack,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_LOA_strip_pack_logical_constraints(self):
        """Test logic-based outer approximation with variation of strip
        packing with some logical constraints."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        # add logical constraints
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(
            expr=strip_pack.no_overlap[1,3].disjuncts[2].indicator_var.lor(
                strip_pack.no_overlap[1,3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(
            expr=strip_pack.no_overlap[2,3].disjuncts[0].indicator_var.lor(
                strip_pack.no_overlap[2,3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt', algorithm='LOA').solve(
            strip_pack,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 13) <= 1E-2)

    @unittest.pytest.mark.expensive
    def test_LOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            cons_layout,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            iterlim=120,
            max_slack=5,  # problem is convex, so can decrease slack
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_LOA_8PP_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_LOA_8PP_logical_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_maxBinary(self):
        """Test LOA with strip packing using max_binary initialization."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt', algorithm='LOA').solve(
            strip_pack, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_LOA_strip_pack_maxBinary_logical_constraints(self):
        """Test LOA with strip packing using max_binary initialization and
        logical constraints."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        # add logical constraints
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(
            expr=strip_pack.no_overlap[1,3].disjuncts[2].indicator_var.lor(
                strip_pack.no_overlap[1,3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(
            expr=strip_pack.no_overlap[2,3].disjuncts[0].indicator_var.lor(
                strip_pack.no_overlap[2,3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt', algorithm='LOA').solve(
            strip_pack, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 13) <= 1E-2)

    def test_LOA_8PP_fixed_disjuncts(self):
        """Test LOA with 8PP using fixed disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            eight_process.use_unit_1or2.disjuncts[0],
            eight_process.use_unit_3ornot.disjuncts[1],
            eight_process.use_unit_4or5ornot.disjuncts[0],
            eight_process.use_unit_6or7ornot.disjuncts[1],
            eight_process.use_unit_8ornot.disjuncts[0]
        ]
        for disj in eight_process.component_data_objects(Disjunct):
            if disj in initialize:
                disj.binary_indicator_var.set_value(1)
            else:
                disj.binary_indicator_var.set_value(0)
        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process, init_strategy='fix_disjuncts',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_custom_disjuncts_with_silly_components_in_list(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()

        eight_process.goofy = Disjunct()
        eight_process.goofy2 = Disjunct()
        eight_process.goofy.deactivate()
        eight_process.goofy2.deactivate()

        initialize = [
            # Use units 1, 4, 7, 8
            [eight_process.use_unit_1or2.disjuncts[0],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[1],
             eight_process.use_unit_8ornot.disjuncts[0],
             eight_process.goofy, eight_process.goofy2], # not a Disjunct
            # Use units 2, 4, 6, 8
            [eight_process.use_unit_1or2.disjuncts[1],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[0],
             eight_process.use_unit_8ornot.disjuncts[0]]
        ]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            results = SolverFactory('gdpopt', algorithm='LOA').solve(
                eight_process, init_strategy='custom_disjuncts',
                custom_init_disjuncts=initialize, mip_solver=mip_solver,
                nlp_solver=nlp_solver)

            self.assertIn("The following disjuncts from the custom disjunct "
                          "initialization set number 0 were unused: goofy2, "
                          "goofy", output.getvalue().strip())
        # and the solution is optimal to boot
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            [eight_process.use_unit_1or2.disjuncts[0],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[1],
             eight_process.use_unit_8ornot.disjuncts[0]],
            # Use units 2, 4, 6, 8
            [eight_process.use_unit_1or2.disjuncts[1],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[0],
             eight_process.use_unit_8ornot.disjuncts[0]]
        ]

        def assert_correct_disjuncts_active(subprob_util_block,
                                            master_util_block):
            # I can get the iteration based on the number of no-good
            # cuts in this case...
            iteration = len(master_util_block.no_good_cuts)
            master = master_util_block.model()
            subprob = subprob_util_block.model()
            if iteration >= 2:
                return  # only checking initialization
            disjs_should_be_active = initialize[iteration]
            seen = set()
            for orig_disj in disjs_should_be_active:
                parent_nm = orig_disj.parent_component().name
                idx = orig_disj.index()
                # Find the corresponding components on the master and subproblem
                master_parent = master.component(parent_nm)
                subprob_parent = subprob.component(parent_nm)
                self.assertIsInstance(master_parent, Disjunct)
                self.assertIsInstance(subprob_parent, Block)
                master_disj = master_parent[idx]
                subprob_disj = subprob_parent[idx]
                self.assertTrue(value(master_disj.indicator_var))
                self.assertTrue(subprob_disj.active)
                seen.add(subprob_disj)
            # check that all the others are inactive (False)
            for disj in subprob_util_block.disjunct_list:
                if disj not in seen:
                    self.assertFalse(disj.active)

        SolverFactory('gdpopt', algorithm='LOA').solve(
            eight_process, init_strategy='custom_disjuncts',
            custom_init_disjuncts=initialize,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            subproblem_initialization_method=assert_correct_disjuncts_active)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(Gurobi().available(),
                         "APPSI Gurobi solver is not available")
    def test_non_auto_persistent_solver_error_checking(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        m = exfile.build_eight_process_flowsheet()
        with self.assertRaisesRegex(
                ValueError,
                ".*GDPopt does not currently support the 'gurobi_persistent' "
                "solver. The only supported persistent solvers are those in "
                "the APPSI package."):
            SolverFactory('gdpopt', algorithm='LOA').solve(
                m, mip_solver='gurobi_persistent')

        # [ESJ 4/8/22] TODO: This will pass when #2355 is resolved
        # But APPSI is okay:
        # SolverFactory('gdpopt', algorithm='LOA').solve(
        #         m, mip_solver='appsi_gurobi')

        # self.assertTrue(fabs(value(m.profit.expr) - 68) <= 1E-2)

@unittest.skipIf(not LOA_solvers_available,
                 "Required subsolvers %s are not available"
                 % (LOA_solvers,))
class TestGDPoptRIC(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            SolverFactory('gdpopt', algorithm='RIC').solve(
                m,
                mip_solver=mip_solver,
                nlp_solver=nlp_solver)
            self.assertIn("Set covering problem was infeasible.",
                          output.getvalue().strip())

    def test_GDP_nonlinear_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.x ** 2)
        SolverFactory('gdpopt', algorithm='RIC').solve(
            m,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver
        )
        self.assertAlmostEqual(value(m.o), 0)

        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=-m.x ** 2, sense=maximize)
        SolverFactory('gdpopt', algorithm='RIC').solve( m,
                                                        mip_solver=mip_solver,
                                                        nlp_solver=nlp_solver )
        self.assertAlmostEqual(value(m.o), 0)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt', algorithm='RIC').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt', algorithm='RIC').solve(m, mip_solver=mip_solver,
                                                       nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_RIC_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_RIC_8PP_logical_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(SolverFactory('gams').available(exception_flag=False),
                         'GAMS solver not available')
    def test_RIC_8PP_gams_solver(self):
        # Make sure that the duals are still correct
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver='gams',
            max_slack=0,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_RIC_8PP_force_NLP(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            force_subproblem_nlp=True,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_RIC_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            strip_pack,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_RIC_strip_pack_default_init_logical_constraints(self):
        """Test logic-based outer approximation with strip packing with
        logical constraints."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        # add logical constraints
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(
            expr=strip_pack.no_overlap[1,3].disjuncts[2].indicator_var.lor(
                strip_pack.no_overlap[1,3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(
            expr=strip_pack.no_overlap[2,3].disjuncts[0].indicator_var.lor(
                strip_pack.no_overlap[2,3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt', algorithm='RIC').solve(
            strip_pack,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 13) <= 1E-2)

    @unittest.pytest.mark.expensive
    def test_RIC_constrained_layout_default_init(self):
        """Test RIC with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            cons_layout,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            iterlim=120,
            max_slack=5,  # problem is convex, so can decrease slack
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_RIC_8PP_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_RIC_strip_pack_maxBinary(self):
        """Test RIC with strip packing using max_binary initialization."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt', algorithm='RIC').solve(
            strip_pack, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_RIC_strip_pack_maxBinary_logical_constraints(self):
        """Test RIC with strip packing using max_binary initialization and
        including logical constraints."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        # add logical constraints
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(
            expr=strip_pack.no_overlap[1,3].disjuncts[2].indicator_var.lor(
                strip_pack.no_overlap[1,3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(
            expr=strip_pack.no_overlap[2,3].disjuncts[0].indicator_var.lor(
                strip_pack.no_overlap[2,3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt', algorithm='RIC').solve(
            strip_pack, init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 13) <= 1E-2)

    def test_RIC_8PP_fixed_disjuncts(self):
        """Test RIC with 8PP using fixed disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            eight_process.use_unit_1or2.disjuncts[0],
            eight_process.use_unit_3ornot.disjuncts[1],
            eight_process.use_unit_4or5ornot.disjuncts[0],
            eight_process.use_unit_6or7ornot.disjuncts[1],
            eight_process.use_unit_8ornot.disjuncts[0]
        ]
        for disj in eight_process.component_data_objects(Disjunct):
            if disj in initialize:
                disj.binary_indicator_var.set_value(1)
            else:
                disj.binary_indicator_var.set_value(0)
        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process, init_strategy='fix_disjuncts',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_RIC_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            [eight_process.use_unit_1or2.disjuncts[0],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[1],
             eight_process.use_unit_8ornot.disjuncts[0]],
            # Use units 2, 4, 6, 8
            [eight_process.use_unit_1or2.disjuncts[1],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[0],
             eight_process.use_unit_8ornot.disjuncts[0]]
        ]

        def assert_correct_disjuncts_active(subprob_util_block,
                                            master_util_block):
            # I can get the iteration based on the number of no-good
            # cuts in this case...
            iteration = len(master_util_block.no_good_cuts)
            master = master_util_block.model()
            subprob = subprob_util_block.model()
            if iteration >= 2:
                return  # only checking initialization
            disjs_should_be_active = initialize[iteration]
            seen = set()
            for orig_disj in disjs_should_be_active:
                parent_nm = orig_disj.parent_component().name
                idx = orig_disj.index()
                # Find the corresponding components on the master and subproblem
                master_parent = master.component(parent_nm)
                subprob_parent = subprob.component(parent_nm)
                self.assertIsInstance(master_parent, Disjunct)
                self.assertIsInstance(subprob_parent, Block)
                master_disj = master_parent[idx]
                subprob_disj = subprob_parent[idx]
                self.assertTrue(value(master_disj.indicator_var))
                self.assertTrue(subprob_disj.active)
                seen.add(subprob_disj)
            # check that all the others are inactive (False)
            for disj in subprob_util_block.disjunct_list:
                if disj not in seen:
                    self.assertFalse(disj.active)

        SolverFactory('gdpopt', algorithm='RIC').solve(
            eight_process, init_strategy='custom_disjuncts',
            custom_init_disjuncts=initialize,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            subproblem_initialization_method=assert_correct_disjuncts_active)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_force_nlp_subproblem_with_general_integer_variables(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(0, 10))
        m.y = Var(bounds=(0, 10))
        m.disjunction = Disjunction(expr=[[m.x**2 <= 4, m.y**2 <= 1],
                                          [(m.x - 1)**2 + (m.y - 1)**2 <= 4,
                                           m.y <= 4]])
        m.obj = Objective(expr=-m.y - m.x)
        results = SolverFactory('gdpopt', algorithm='RIC').solve(
            m, init_strategy='no_init', mip_solver=mip_solver,
            nlp_solver=nlp_solver, force_subproblem_nlp=True)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.x), 2)
        self.assertAlmostEqual(value(m.y), 1 + sqrt(3))

@unittest.skipIf(not GLOA_solvers_available,
                 "Required subsolvers %s are not available"
                 % (GLOA_solvers,))
@unittest.skipIf(not mcpp_available(), "MC++ is not available")
class TestGLOA(unittest.TestCase):
    """Tests for global logic-based outer approximation."""

    def test_GDP_integer_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[
            m.x >= m.y, m.y >= 3.5
        ],
        [
            m.x >= m.y, m.y >= 2.5
        ]])
        m.o = Objective(expr=m.x)
        SolverFactory('gdpopt').solve(
            m, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            minlp_solver=minlp_solver
        )
        self.assertAlmostEqual(value(m.o.expr), 3)

    def make_nonlinear_gdp_with_int_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[
            m.x**2 >= m.y, m.y >= 3.5
        ],
        [
            m.x**2 >= m.y, m.y >= 2.5
        ]])
        m.o = Objective(expr=m.x)
        return m

    def test_nonlinear_GDP_integer_vars(self):
        m = self.make_nonlinear_gdp_with_int_vars()
        SolverFactory('gdpopt').solve(
            m, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            minlp_solver=minlp_solver
        )
        self.assertAlmostEqual(value(m.o.expr), sqrt(3))
        self.assertAlmostEqual(value(m.y), 3)

    def test_nonlinear_GDP_integer_vars_force_nlp_subproblem(self):
        m = self.make_nonlinear_gdp_with_int_vars()
        SolverFactory('gdpopt').solve(
            m, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            minlp_solver=minlp_solver,
            force_subproblem_nlp=True
        )
        self.assertAlmostEqual(value(m.o.expr), sqrt(3))
        self.assertAlmostEqual(value(m.y), 3)

    def test_GDP_integer_vars_infeasible(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.d = Disjunction(expr=[[
            m.x >= m.y, m.y >= 3.5
        ],
        [
            m.x >= m.y, m.y >= 2.5
        ]])
        m.o = Objective(expr=m.x)
        res = SolverFactory('gdpopt').solve(
            m, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            minlp_solver=minlp_solver
        )
        self.assertEqual(res.solver.termination_condition,
                         TerminationCondition.infeasible)

    @unittest.skipUnless(license_available and sympy_available,
                         "Global NLP solver license not available or sympy "
                         "not available.")
    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt').solve(m, strategy='GLOA', mip_solver=mip_solver,
                                      nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(license_available and sympy_available,
                         "Global NLP solver license not available or sympy "
                         "not available.")
    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt').solve(m, strategy='GLOA', mip_solver=mip_solver,
                                      nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    def test_GLOA_8PP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='GLOA', tee=False,
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(license_available and sympy_available,
                         "Global NLP solver license not available or sympy "
                         "not available.")
    def test_GLOA_8PP_logical(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='GLOA', tee=False,
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    def test_GLOA_8PP_force_NLP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='GLOA', tee=False,
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            force_subproblem_nlp=True
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    def test_GLOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(license_available and sympy_available,
                         "Global NLP solver license not available or sympy "
                         "not available.")
    def test_GLOA_strip_pack_default_init_logical_constraints(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        # add logical constraints
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(
            expr=strip_pack.no_overlap[1,3].disjuncts[2].indicator_var.lor(
                strip_pack.no_overlap[1,3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(
            expr=strip_pack.no_overlap[2,3].disjuncts[0].indicator_var.lor(
                strip_pack.no_overlap[2,3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 13) <= 1E-2)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    @unittest.pytest.mark.expensive
    def test_GLOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, strategy='GLOA',
            mip_solver=mip_solver,
            iterlim=36,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            tee=False)
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_GLOA_ex_633_trespalacios(self):
        """Test LOA with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt').solve(
            model, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            tee=False)
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    def test_GLOA_nonconvex_HENS(self):
        exfile = import_file(join(exdir, 'small_lit', 'nonconvex_HEN.py'))
        model = exfile.build_gdp_model()
        SolverFactory('gdpopt').solve(
            model, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            tee=False)
        objective_value = value(model.objective.expr)
        self.assertAlmostEqual(objective_value * 1E-5, 1.14385, 2)

    @unittest.skipUnless(license_available,
                         "Global NLP solver license not available.")
    def test_GLOA_disjunctive_bounds(self):
        exfile = import_file(join(exdir, 'small_lit', 'nonconvex_HEN.py'))
        model = exfile.build_gdp_model()
        SolverFactory('gdpopt').solve(
            model, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            calc_disjunctive_bounds=True,
            tee=False)
        objective_value = value(model.objective.expr)
        self.assertAlmostEqual(objective_value * 1E-5, 1.14385, 2)


if __name__ == '__main__':
    unittest.main()
