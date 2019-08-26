import pyutilib.th as unittest
from pyomo.environ import *
from pyomo.core.expr.logical_expr import LogicalAnd, LogicalOr, LogicalXor, Implies

'''
First testing file.
'''

class TestLogicalClasses(unittest.TestCase):

    def test_elementary_nodes(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar() 
        m.Y2 = BooleanVar()
        
        m.Y1.value = True
        m.Y2.value = True
        #try alterbate way to set value
        #m.Y1.set_value(True)
        #m.Y2.set_value(True)
        self.assertTrue(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(Implies(m.Y1, m.Y2)))
        self.assertFalse(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = False, True
        self.assertFalse(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(Implies(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = True, False
        self.assertFalse(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertFalse(value(Implies(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = False, False
        self.assertFalse(value(logical_expr.LogicalAnd(m.Y1, m.Y2)))
        self.assertFalse(value(logical_expr.LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(logical_expr.Implies(m.Y1, m.Y2)))
        self.assertFalse(value(logical_expr.LogicalXor(m.Y1, m.Y2)))


if __name__ == "__main__":
    unittest.main()