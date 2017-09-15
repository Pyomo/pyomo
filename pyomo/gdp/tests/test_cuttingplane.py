import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random

from nose.tools import set_trace
from pyomo.opt import SolverFactory

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
        m.x = Var(bounds=(0,5))
        m.y = Var(bounds=(0,5))
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

        m.obj = Objective(expr=m.x + 2*m.y)
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
        self.assertEqual(len(cut.body._coef), 4)
        self.assertEqual(len(cut.body._args), 4)
        self.assertEqual(cut.body._const, 0)
        
        coefs = {
            0: 0.45,
            1: 0.55,
            2: -0.1,
            3: 0.1
        }

        xhat = {
            0: 2.7,
            1: 1.3,
            2: 0.15,
            3: 0.85
        }

        variables = {
            0: m.x,
            1: m.y,
            2: m.d[1].indicator_var,
            3: m.d[0].indicator_var
        }

        for i in range(4):
            self.assertAlmostEqual(cut.body._coef[i], coefs[i])
            self.assertEqual(len(cut.body._args[i]._coef), 1)
            self.assertEqual(len(cut.body._args[i]._args), 1)
            self.assertAlmostEqual(cut.body._args[i]._const, -1*xhat[i])
            self.assertEqual(cut.body._args[i]._coef[0], 1)
            self.assertIs(cut.body._args[i]._args[0], variables[i])
