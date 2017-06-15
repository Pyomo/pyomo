import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

# DEBUG
from nose.tools import set_trace

class TestBigM(unittest.TestCase):
    def test_twoTerm_disjunction(self):
        m = ConcreteModel()
        m.a = Var(within=NonNegativeReals)
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

        TransformationFactory('gdp.bigm').apply_to(m, default_bigM=78)
        
        # we have a transformation block
        gdpblock = m.component("_gdp_relax")
        self.assertIsInstance(gdpblock, Block)
        # have XOR constraints
        # set_trace()
        
