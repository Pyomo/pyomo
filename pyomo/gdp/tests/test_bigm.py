import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

# DEBUG
from nose.tools import set_trace

class TestBigM_2TermDisj(unittest.TestCase):
    def test_twoTerm_disjunction(self):
        m = ConcreteModel()
        # m.BigM = Suffix(direction=Suffix.LOCAL)
        # m.BigM[None] = 78
        m.a = Var(bounds=(2,7))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a == 0)
            else:
                disjunct.c = Constraint(expr=m.a >= 5)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disj = Disjunction(rule=disj_rule)

        TransformationFactory('gdp.bigm').apply_to(m)#, default_bigM=78)
        
        # we have a transformation block
        gdpblock = m.component("_gdp_relax")
        self.assertIsInstance(gdpblock, Block)
        # have indicator variables
        dblock = m._gdp_relax.component("d")
        self.assertIsInstance(dblock, Block)
        self.assertIsInstance(dblock[0].indicator_var, Var)
        self.assertTrue(dblock[0].indicator_var.is_binary())
        self.assertIsInstance(dblock[1].indicator_var, Var)
        self.assertTrue(dblock[1].indicator_var.is_binary())

        # old constraint there, deactivated, new constraint active
        oldc = dblock[0].component("c")
        self.assertIsInstance(oldc, Constraint)
        self.assertFalse(oldc.active)
        newc = dblock[0].component("c_eq")
        self.assertIsInstance(newc, Constraint)
        self.assertTrue(newc.active)

        # new constraint is right
        self.assertIs(oldc.lower, newc.lower)
        self.assertIs(oldc.upper, newc.upper)
        self.assertIs(oldc.body, newc.body._args[0])
        self.assertIs(dblock[0].indicator_var, newc.body._args[1]._args[0])
        set_trace()
        
