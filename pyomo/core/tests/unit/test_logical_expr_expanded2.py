import pyutilib.th as unittest
from pyomo.environ import *
from pyomo.core.expr.logical_expr import (LogicalExpressionBase, NotExpression, 
	AndExpression, OrExpression, Implication, EquivalenceExpression, XorExpression, 
	ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalence, 
	LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor
    )

'''

Test file 2

'''
class TestLogicalClasses(unittest.TestCase):

    def test_elementary_nodes(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar() 
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()

        m.Y1.value = True
        m.Y2.value = True
        m.Y3.value = True
        #try alterbate way to set value
        #m.Y1.set_value(True)
        #m.Y2.set_value(True)
        a = LogicalAnd(m.Y1, m.Y2, m.Y3)
        b = LogicalOr(m.Y1, m.Y2, m.Y3)
        c = m.Y1 and m.Y2 and m.Y3 
        d = m.Y1 or m.Y2 or m.Y3	
        self.assertTrue(value(a))
        self.assertTrue(value(b))
        self.assertTrue(value(a.implies(b)))
        self.assertTrue(value(Implies(a,b)))



        m.Y1.value, m.Y2.value, m.Y3.value, = False, True, True
        a = LogicalAnd(m.Y1, m.Y2, m.Y3)
        b = LogicalOr(m.Y1, m.Y2, m.Y3)
        c = m.Y1 and m.Y2 and m.Y3 
        d = m.Y1 or m.Y2 or m.Y3	
        self.assertFalse(value(a))
        self.assertTrue(value(b))
        self.assertTrue(value(a.implies(b)))
        self.assertTrue(value(Implies(a,b)))


if __name__ == "__main__":
    unittest.main()