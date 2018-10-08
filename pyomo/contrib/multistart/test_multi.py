import pyutilib.th as unittest
from pyomo.environ import *
from six.moves import range
from math import fabs


# due to stochastic nature of the random restarts, these tests just demonstrate,
# that for a small sample, the test will not do worse than the standard solver.
# this is non-exhaustive due to the randomness
class MultistartTests(unittest.TestCase):

    # test standard random restarts
    def test_as_good_with_iteration_rand(self):
        # initialize model with data
        m = build_model()

        # create ipopt solver
        optsolver = SolverFactory('ipopt')
        optsolver.solve(m)
        for i in range(1000):

            m2 = build_model()
            optsolver = SolverFactory('multistart')
            optsolver.solve(m2, iterations=10)
            print(value(m2.obj.expr))
            print(value(m2.x1))
            print(value(m.obj.expr))
            self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr) - .001))
            #self.assertTrue(value(m2.obj.expr) >= value(m.obj.expr))
            del m2

    #test that other strategies do no worse
    # def test_as_good_with_iteration_other_strategies(self):
    #     # initialize model with data
    #     m = build_model()
    #
    #     # create ipopt solver
    #     optsolver = SolverFactory('ipopt')
    #     optsolver.solve(m)
    #     for i in range(10):
    #
    #         m2 = build_model()
    #         optsolver = SolverFactory('multistart')
    #         optsolver.solve(m2, strategy='rand_distributed')
    #
    #         self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr)))
    #         del m2
    #     for i in range(10):
    #
    #         m2 = build_model()
    #         optsolver = SolverFactory('multistart')
    #         optsolver.solve(m2, strategy='midpoint_guess_and_bound')
    #
    #         self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr)))
    #         del m2
    #     for i in range(10):
    #
    #         m2 = build_model()
    #         optsolver = SolverFactory('multistart')
    #         optsolver.solve(m2, strategy='rand_guess_and_bound')
    #
    #         self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr)))
    #         del m2
    # # test that the high confidence stopping rule with very lenient parameters
    # # does no worse.
    #
    # def test_as_good_with_HCS_rule(self):
    #     # initialize model with data
    #     m = build_model()
    #
    #     # create ipopt solver
    #     optsolver = SolverFactory('ipopt')
    #     optsolver.solve(m)
    #     for i in range(5):
    #
    #         m2 = build_model()
    #         optsolver = SolverFactory('multistart')
    #         optsolver.solve(m2, iterations=-1, HCS_param=(.99, .99))
    #
    #         self.assertTrue((value(m2.obj.expr)) >= (value(m.obj.expr)))
    #         del m2

# simple non-convex model with many multiple minima


def build_model():
    model = ConcreteModel()
    model.x1 = Var(initialize=1, bounds=(0, 100))

    def obj_rule(amodel):
        return model.x1 * sin(model.x1)





    model.obj = Objective(rule=obj_rule, sense=maximize)
    return model


if __name__ == '__main__':
    unittest.main()
