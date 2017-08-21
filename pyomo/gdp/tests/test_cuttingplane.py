import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random

from nose.tools import set_trace

# TODO:
#     - test that deactivated objectives on the model don't get used by the
#       transformation

class TwoTermDisj(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)
    
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.a = Var(bounds=(2,7))
        m.x = Var(bounds=(4,9))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c1 = Constraint(expr=m.a == 0)
                disjunct.c2 = Constraint(expr=m.x <= 7)
            else:
                disjunct.c = Constraint(expr=m.a >= 5)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disjunction = Disjunction(rule=disj_rule)

        m.obj = Objective(expr=m.x - m.a)
        return m

    def test_something(self):
        # TODO: this is very much not a test yet! (and coverage is a lie!)
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        set_trace()
