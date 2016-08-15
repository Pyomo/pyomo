import pyutilib.th as unittest
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_constraint import (constraint,
                                                  constraint_dict,
                                                  constraint_list)
from pyomo.core.base.component_variable import variable
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

class Test_constraint(unittest.TestCase):

    def test_init(self):
        c = constraint()
        self.assertTrue(c.parent is None)
        self.assertEqual(c.ctype, Constraint)
        self.assertEqual(c.body, None)
        self.assertEqual(c.lower, None)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

class Test_constraint_dict(_TestComponentDictBase,
                           unittest.TestCase):
    _container_type = constraint_dict
    _ctype_factory = lambda self: constraint()

class Test_constraint_list(_TestComponentListBase,
                           unittest.TestCase):
    _container_type = constraint_list
    _ctype_factory = lambda self: constraint()

if __name__ == "__main__":
    unittest.main()
