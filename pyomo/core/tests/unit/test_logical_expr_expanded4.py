import pyutilib.th as unittest
from pyomo.environ import *
#from pyomo.core.epxr.logical_expr import Expressions

from pyomo.core.expr.logical_expr import (LogicalExpressionBase, NotExpression, 
	AndExpression, OrExpression, Implication, EquivalenceExpression, XorExpression, 
	ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalence, 
	LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor
    )

########-----------------------------------#########

"""

In this file, MultiArgExpression will be tested, the same problem as the 
last test file is expected.

The Expression that will be tested include 
 
And, Or, Exactly, AtMost, AtLeast

"""

########-----------------------------------#########

class TestLogicalClasses(unittest.TestCase):

  def test_elementary_nodes(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar() 
    m.Y2 = BooleanVar()
    m.Y3 = BooleanVar()
    m.Y4 = BooleanVar()

    ###########----------------------###########

    """
    Test for AndExpression and OrExpression
    """


    m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, True
    And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
    And_operator = m.Y1 and m.Y2 and m.Y3 and m.Y4
    self.assertTrue(value(And_static)) 
    self.assertTrue(value(And_operator))

    Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
    Or_operator = m.Y1 or m.Y2 or m.Y3 or m.Y4
    self.assertTrue(value(Or_static))
    self.assertTrue(value(Or_operator))

    m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, False
    And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
    And_operator = m.Y1 and m.Y2 and m.Y3 and m.Y4
    self.assertFalse(value(And_static)) 
    self.assertFalse(value(And_operator))

    Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
    Or_operator = m.Y1 or m.Y2 or m.Y3 or m.Y4
    self.assertTrue(value(Or_static))
    self.assertTrue(value(Or_operator))

    m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = False, False, False, False

    And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
    And_operator = m.Y1 and m.Y2 and m.Y3 and m.Y4
    self.assertFalse(value(And_static)) 
    self.assertFalse(value(And_operator))
    Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
    Or_operator = m.Y1 or m.Y2 or m.Y3 or m.Y4
    self.assertFalse(value(Or_static))
    self.assertFalse(value(Or_operator))

    ###########----------------------###########
      
    """
    Test for Exactly, AtMost and AtLeast
    """

    m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, False
    Exactly_0 = Exactly(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_1 = Exactly(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_2 = Exactly(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_3 = Exactly(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_4 = Exactly(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertFalse(Exactly_0)
    self.assertFalse(Exactly_1)
    self.assertFalse(Exactly_2)
    self.assertTrue(Exactly_3)
    self.assertFalse(Exactly_4)

    AtMost_0 = AtMost(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_1 = AtMost(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_2 = AtMost(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_3 = AtMost(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_4 = AtMost(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertTrue(AtMost_0)
    self.assertTrue(AtMost_1)
    self.assertTrue(AtMost_2)
    self.assertTrue(AtMost_3)
    self.assertFalse(AtMost_4)

    AtLeast_0 = AtLeast(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_1 = AtLeast(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_2 = AtLeast(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_3 = AtLeast(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_4 = AtLeast(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertFalse(AtLeast_0)
    self.assertFalse(AtLeast_1)
    self.assertFalse(AtLeast_2)
    self.assertTrue(AtLeast_3)
    self.assertTrue(AtLeast_4)


    m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, False, False, False

    Exactly_0 = Exactly(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_1 = Exactly(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_2 = Exactly(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_3 = Exactly(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    Exactly_4 = Exactly(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertFalse(Exactly_0)
    self.assertTrue(Exactly_1)
    self.assertFalse(Exactly_2)
    self.assertFalse(Exactly_3)
    self.assertFalse(Exactly_4)

    AtMost_0 = AtMost(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_1 = AtMost(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_2 = AtMost(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_3 = AtMost(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtMost_4 = AtMost(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertTrue(AtMost_0)
    self.assertTrue(AtMost_1)
    self.assertFalse(AtMost_2)
    self.assertFalse(AtMost_3)
    self.assertFalse(AtMost_4)

    AtLeast_0 = AtLeast(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_1 = AtLeast(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_2 = AtLeast(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_3 = AtLeast(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    AtLeast_4 = AtLeast(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
    self.assertFalse(AtLeast_0)
    self.assertTrue(AtLeast_1)
    self.assertTrue(AtLeast_2)
    self.assertTrue(AtLeast_3)
    self.assertTrue(AtLeast_4)



if __name__ == "__main__":
    unittest.main()




      













