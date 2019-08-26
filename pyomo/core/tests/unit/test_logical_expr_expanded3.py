import pyutilib.th as unittest
from pyomo.environ import *
#from pyomo.core.epxr.logical_expr import Expressions

from pyomo.core.expr.logical_expr import (LogicalExpressionBase, NotExpression, 
	AndExpression, OrExpression, Implication, EquivalenceExpression, XorExpression, 
	ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalence, 
	LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor
    )

####------------------########
'''
In this file, all binary operation will be test at a very
basic level

In this file, the test nodes takes the name 'OperationNmae_MethodType'
Expressions tested include 
Equivalence, Xor, Implies

Somehow, none of the class method works due to an attribute error
The challenge here is if a node is a leaf node, both class method and 
operator method will raise an attribute error.


I'm thinking about resolving this issue by create an unary node that does
nothing but turn a leafnode into a logical_expression if possible.
'''

class TestLogicalClasses(unittest.TestCase):

    def test_elementary_nodes(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar() 
        m.Y2 = BooleanVar()
        
        #######-----------------------------########
        m.Y1.value, m.Y2.value = True, True

        eq_static =  Equivalence(m.Y1, m.Y2)
        #eq_class = m.Y1.equals(m.Y2)
        #eq_operator = (m.Y1 == m.Y2)
        self.assertTrue(value(eq_static))
        #self.assertTrue(value(eq_class))
        #self.assertTrue(value(eq_operator))

        xor_static = LogicalXor(m.Y1, m.Y2)
        #xor_class = m.Y1.Xor(m.Y2)
        #xor_operator =  m.Y1 ^ m.Y2
        self.assertFalse(value(xor_static))
        #self.assertFalse(value(xor_class))
        #self.assertFalse(value(xor_operator))

        implication_static = Implies(m.Y1, m.Y2)
        #implication_class = m.Y1.implies(m.Y2)
        self.assertTrue(value(implication_static))
        #self.assertTrue(value(implication_class))

       	#######-----------------------------########
       	m.Y1.value, m.Y2.value = True, False

        eq_static =  Equivalence(m.Y1, m.Y2)
        #eq_class = m.Y1.equals(m.Y2)
        #eq_operator = (m.Y1 == m.Y2)
        self.assertFalse(value(eq_static))
        #self.assertFalse(value(eq_class))
        #self.assertFalse(value(eq_operator))

        xor_static = LogicalXor(m.Y1, m.Y2)
        #xor_class = m.Y1.Xor(m.Y2)
        #xor_operator =  m.Y1 ^ m.Y2
        self.assertTrue(value(xor_static))
        #self.assertTrue(value(xor_class))
        #self.assertTrue(value(xor_operator))

        implication_static = Implies(m.Y1, m.Y2)
        #implication_class = m.Y1.implies(m.Y2)
        self.assertFalse(value(implication_static))
        #self.assertFalse(value(implication_class))


        #######-----------------------------########
        m.Y1.value, m.Y2.value = False, True

        eq_static =  Equivalence(m.Y1, m.Y2)
        #eq_class = m.Y1.equals(m.Y2)
        #eq_operator = (m.Y1 == m.Y2)
        self.assertFalse(value(eq_static))
        #self.assertFalse(value(eq_class))
        #self.assertFalse(value(eq_operator))

        xor_static = LogicalXor(m.Y1, m.Y2)
        #xor_class = m.Y1.Xor(m.Y2)
        #xor_operator =  m.Y1 ^ m.Y2
        self.assertTrue(value(xor_static))
        #self.assertTrue(value(xor_class))
        #self.assertTrue(value(xor_operator))

        implication_static = Implies(m.Y1, m.Y2)
        #implication_class = m.Y1.implies(m.Y2)
        self.assertTrue(value(implication_static))
        #self.assertTrue(value(implication_class))

        ######----------------------------#######

        m.Y1.value, m.Y2.value = False, False
        eq_static =  Equivalence(m.Y1, m.Y2)
        #eq_class = m.Y1.equals(m.Y2)
        #eq_operator = (m.Y1 == m.Y2)
        self.assertTrue(value(eq_static))
        #self.assertTrue(value(eq_class))
        #self.assertTrue(value(eq_operator))

        xor_static = LogicalXor(m.Y1, m.Y2)
        #xor_class = m.Y1.Xor(m.Y2)
        #xor_operator =  m.Y1 ^ m.Y2
        self.assertFalse(value(xor_static))
        #self.assertFalse(value(xor_class))
        #self.assertFalse(value(xor_operator))

        implication_static = Implies(m.Y1, m.Y2)
        #implication_class = m.Y1.implies(m.Y2)
        self.assertTrue(value(implication_static))
        #self.assertTrue(value(implication_class))

        #######-----------------------------########


if __name__ == "__main__":
    unittest.main()