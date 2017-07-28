import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *

# DEBUG
from nose.tools import set_trace

class TwoTermDisj(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
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
        m.disjunction = Disjunction(rule=disj_rule)
        return m

    def test_something(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        set_trace()

