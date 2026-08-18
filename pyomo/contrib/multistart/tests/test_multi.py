# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
from itertools import product

from io import StringIO

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import, numpy_available, scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Var,
    maximize,
    sin,
    value,
)
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.solvers.ipopt import Ipopt
from pyomo.contrib.solver.common.util import NoOptimalSolutionError
from pyomo.contrib.solver.common.results import TerminationCondition


@unittest.skipIf(not numpy_available, "Numpy not available")
@unittest.skipIf(not SolverFactory('ipopt').available(), "IPOPT not available")
class MultistartTests(unittest.TestCase):
    """
    Due to stochastic nature of the random restarts, these tests just
    demonstrate, that for a small sample, the test will not do worse than the
    standard solver. this is non-exhaustive due to the randomness. Hence all
    asserts are inequalities.
    """

    def test_as_good_as_standard(self):
        standard_model = build_model()
        SolverFactory('ipopt').solve(standard_model)
        standard_objective_value = value(
            next(standard_model.component_data_objects(Objective, active=True))
        )

        fresh_model = build_model()
        multistart_iterations = 10
        test_trials = 10
        for strategy, _ in product(strategies.keys(), range(test_trials)):
            m2 = fresh_model.clone()
            SolverFactory('multistart').solve(
                m2, iterations=multistart_iterations, strategy=strategy
            )
            clone_objective_value = value(
                next(m2.component_data_objects(Objective, active=True))
            )
            self.assertGreaterEqual(
                clone_objective_value, standard_objective_value
            )  # assumes maximization

    def test_as_good_with_HCS_rule(self):
        """test that the high confidence stopping rule with very lenient
        parameters does no worse.
        """
        # initialize model with data
        m = build_model()

        # create ipopt solver
        SolverFactory('ipopt').solve(m)
        for i in range(5):
            m2 = build_model()
            SolverFactory('multistart').solve(
                m2, iterations=-1, stopping_mass=0.99, stopping_delta=0.99
            )
            m_objectives = m.component_data_objects(Objective, active=True)
            m_obj = next(m_objectives, None)
            m2_objectives = m2.component_data_objects(Objective, active=True)
            m2_obj = next(m2_objectives, None)
            # Assert that multistart solver does no worse than standard solver
            self.assertTrue((value(m2_obj.expr)) >= (value(m_obj.expr) - 0.001))
            del m2

    def test_missing_bounds(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            SolverFactory('multistart').solve(m)
            self.assertIn(
                "Skipping reinitialization of unbounded "
                "variable x with bounds (0, None).",
                output.getvalue().strip(),
            )
        with self.assertRaises(ValueError):
            SolverFactory('multistart').solve(m, strategy="rand_vector")

    def test_var_value_None(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)
        SolverFactory('multistart').solve(m)

    def test_no_obj(self):
        m = ConcreteModel()
        m.x = Var()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            try:
                SolverFactory('multistart').solve(m)
            except:
                pass
            self.assertIn(
                "No objective found in the provided model. The solver will "
                "stop if it finds a feasible solution before completing "
                "the total number of iterations.",
                output.getvalue().strip(),
            )

    def test_model_infeasible(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.c = Constraint(expr=m.x >= 2)
        m.o = Objective(expr=m.x)

        with self.assertRaises(NoOptimalSolutionError):
            SolverFactory('multistart').solve(m, iterations=2)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            SolverFactory('multistart').solve(
                m,
                iterations=-1,
                HCS_max_iterations=3,
                raise_exception_on_nonoptimal_result=False,
            )
            self.assertIn(
                "High confidence stopping rule was unable to "
                "complete after 3 iterations.",
                output.getvalue().strip(),
            )

    def test_should_stop(self):
        soln = [0] * 149
        self.assertFalse(should_stop(soln, 0.5, 0.5, 0.001))
        soln += [0.001]
        self.assertTrue(should_stop(soln, 0.5, 0.5, 0.001))
        soln = [0] * 149 + [0.01]
        self.assertFalse(should_stop(soln, 0.5, 0.5, 0.001))
        soln = [0] * 149 + [-0.001]
        self.assertTrue(should_stop(soln, 0.5, 0.5, 0.001))

    def test_multiple_obj(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective(expr=m.x)
        m.o2 = Objective(expr=m.x)
        with self.assertRaisesRegex(RuntimeError, "multiple active objectives"):
            SolverFactory('multistart').solve(m)

    def test_unsupported_sampling_method(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)
        with self.assertRaises(ValueError):
            SolverFactory('multistart').solve(m, sampling_method="dummy")

    def test_solver_object_matches_solver_string(self):
        nlp_solver = Ipopt()
        solver_str = "ipopt"
        seed = 145

        fresh_model = build_model()

        m1 = fresh_model.clone()
        results_obj_obj = SolverFactory('multistart').solve(
            m1, subsolver=nlp_solver, seed=seed
        )
        m2 = fresh_model.clone()
        results_obj_str = SolverFactory('multistart').solve(
            m2, subsolver=solver_str, seed=seed
        )
        self.assertAlmostEqual(
            results_obj_obj.incumbent_objective, results_obj_str.incumbent_objective
        )

    @unittest.skipIf(not scipy_available, "Scipy not available")
    def test_sampling_methods(self):
        sampling_methods = ["uniform", "latin_hypercube", "sobol"]
        strategies = ["rand", "rand_vector"]
        seed = 145
        # For the simple model, check all sampling methods converge
        simple_model = build_model()
        for method, strategy in product(sampling_methods, strategies):
            m = simple_model.clone()
            res = SolverFactory("multistart").solve(
                m, strategy=strategy, sampling_method=method
            )
            self.assertEqual(
                res.termination_condition,
                TerminationCondition.convergenceCriteriaSatisfied,
            )

    def test_max_time_limit(self):
        simple_model = build_model()
        output = StringIO()

        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            res = SolverFactory('multistart').solve(
                simple_model,
                raise_exception_on_nonoptimal_result=False,
                time_limit=1e-6,
            )
            self.assertIn(
                "Time limit reached after 1 iterations.", output.getvalue().strip()
            )
            self.assertEqual(
                res.termination_condition, TerminationCondition.maxTimeLimit
            )


def build_model():
    """Simple non-convex model with many local minima"""
    model = ConcreteModel()
    model.x1 = Var(initialize=1, bounds=(0, 100))
    model.x2 = Var(initialize=5, bounds=(5, 6))
    model.x2.fix(5)
    model.objtv = Objective(expr=model.x1 * sin(model.x1), sense=maximize)
    return model


if __name__ == '__main__':
    unittest.main()
