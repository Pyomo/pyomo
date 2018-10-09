from six.moves import range

import pyutilib.th as unittest
from pyomo.environ import (
    ConcreteModel, Objective, SolverFactory, Var, maximize, sin, value
)


class MultistartTests(unittest.TestCase):
    """
    Due to stochastic nature of the random restarts, these tests just
    demonstrate, that for a small sample, the test will not do worse than the
    standard solver. this is non-exhaustive due to the randomness.
    """

    # test standard random restarts
    def test_as_good_with_iteration_rand(self):
        # initialize model with data
        m = build_model()

        # create ipopt solver
        optsolver = SolverFactory('ipopt')
        optsolver.solve(m)
        for i in range(10):

            m2 = build_model()
            SolverFactory('multistart').solve(m2, iterations=10)
            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            del m2

    def test_as_good_with_iteration_other_strategies(self):
        """Test that other strategies do no worse"""
        # initialize model with data
        m = build_model()

        # create ipopt solver
        SolverFactory('ipopt').solve(m)
        for i in range(10):

            m2 = build_model()
            SolverFactory('multistart').solve(
                m2, iterations=10, strategy='rand_distributed')

            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            del m2
        for i in range(10):
            m2 = build_model()
            SolverFactory('multistart').solve(
                m2, iterations=10, strategy='midpoint_guess_and_bound')

            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            del m2
        for i in range(10):
            m2 = build_model()
            SolverFactory('multistart').solve(
                m2, iterations=10, strategy='rand_guess_and_bound')

            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            del m2

    def test_as_good_with_HCS_rule(self):
        """test that the high confidence stopping rule with very lenient
        parameters does no worse.
        """
        # initialize model with data
        m = build_model()

        # create ipopt solver
        optsolver = SolverFactory('ipopt')
        optsolver.solve(m)
        for i in range(5):

            m2 = build_model()
            SolverFactory('multistart').solve(
                m2, iterations=-1, stopping_mass=0.99, stopping_delta=0.99)

            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            del m2


def build_model():
    """Simple non-convex model with many local minima"""
    model = ConcreteModel()
    model.x1 = Var(initialize=1, bounds=(0, 100))

    def obj_rule(amodel):
        return model.x1 * sin(model.x1)

    model.obj = Objective(rule=obj_rule, sense=maximize)
    return model


if __name__ == '__main__':
    unittest.main()
