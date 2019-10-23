import logging
from itertools import product

from six import StringIO
from six.moves import range

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
    ConcreteModel, Constraint, NonNegativeReals, Objective, SolverFactory, Var,
    maximize, sin, value
)


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
        standard_objective_value = value(next(standard_model.component_data_objects(Objective, active=True)))

        fresh_model = build_model()
        multistart_iterations = 10
        test_trials = 10
        for strategy, _ in product(strategies.keys(), range(test_trials)):
            m2 = fresh_model.clone()
            SolverFactory('multistart').solve(m2, iterations=multistart_iterations, strategy=strategy)
            clone_objective_value = value(next(m2.component_data_objects(Objective, active=True)))
            self.assertGreaterEqual(clone_objective_value, standard_objective_value)  # assumes maximization

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
                m2, iterations=-1, stopping_mass=0.99, stopping_delta=0.99)
            m_objectives = m.component_data_objects(Objective, active=True)
            m_obj = next(m_objectives, None)
            m2_objectives = m2.component_data_objects(Objective, active=True)
            m2_obj = next(m2_objectives,None)
            # Assert that multistart solver does no worse than standard solver
            self.assertTrue((value(m2_obj.expr)) >= (value(m_obj.expr) - .001))
            del m2

    def test_missing_bounds(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            SolverFactory('multistart').solve(m)
            self.assertIn("Skipping reinitialization of unbounded "
                          "variable x with bounds (0, None).",
                          output.getvalue().strip())

    def test_var_value_None(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)
        SolverFactory('multistart').solve(m)

    def test_model_infeasible(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.c = Constraint(expr=m.x >= 2)
        m.o = Objective(expr=m.x)
        SolverFactory('multistart').solve(m, iterations=2)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
            SolverFactory('multistart').solve(
                m, iterations=-1, HCS_max_iterations=3)
            self.assertIn("High confidence stopping rule was unable to "
                          "complete after 3 iterations.",
                          output.getvalue().strip())

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
        with self.assertRaisesRegexp(RuntimeError, "multiple active objectives"):
            SolverFactory('multistart').solve(m)

    def test_no_obj(self):
        m = ConcreteModel()
        m.x = Var()
        with self.assertRaisesRegexp(RuntimeError, "no active objective"):
            SolverFactory('multistart').solve(m)

    def test_const_obj(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective(expr = 5)
        with self.assertRaisesRegexp(RuntimeError, "constant objective"):
            SolverFactory('multistart').solve(m)


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
