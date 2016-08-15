import pyutilib.th as unittest
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_expression import (expression,
                                                  expression_dict,
                                                  expression_list)
from pyomo.core.base.component_variable import variable
from pyomo.core.base.expression import Expression
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

class Test_expression(unittest.TestCase):

    def test_init(self):
        e = expression()
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, Expression)
        self.assertEqual(e.expr, None)

class Test_expression_dict(_TestComponentDictBase,
                           unittest.TestCase):
    _container_type = expression_dict
    _ctype_factory = lambda self: expression()

class Test_expression_list(_TestComponentListBase,
                           unittest.TestCase):
    _container_type = expression_list
    _ctype_factory = lambda self: expression()

if __name__ == "__main__":
    unittest.main()
