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
        m.x = Var(bounds=(1,4))
        m.y = Var(bounds=(1,4))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c1 = Constraint(expr=1 <= m.x <= 2)
                disjunct.c2 = Constraint(expr=3 <= m.y <= 4)
            else:
                disjunct.c1 = Constraint(expr=3 <= m.x <= 4)
                disjunct.c2 = Constraint(expr=1 <= m.y <= 2)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disjunction = Disjunction(rule=disj_rule)

        m.obj = Objective(expr=4*m.y - m.x, sense=maximize)
        return m

    def test_transformation_block(self):
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we created the block
        transBlock = m._pyomo_gdp_cuttingplane_relaxation
        self.assertIsInstance(transBlock, Block)
        # the cuts are on it
        cuts = transBlock.cuts
        self.assertIsInstance(cuts, Constraint)
        # this one's tiny, so we've just added one cut
        self.assertEqual(len(cuts), 1)

    def test_cut_constraint(self):
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cut = m._pyomo_gdp_cuttingplane_relaxation.cuts[0]
        self.assertEqual(cut.lower, 0)
        self.assertIsNone(cut.upper)

        # test body
        #set_trace()
