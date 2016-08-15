import pyutilib.th as unittest
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_variable import (variable,
                                                variable_dict,
                                                variable_list)
from pyomo.core.base.var import Var
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

class Test_variable(unittest.TestCase):

    def test_init(self):
        v = variable()
        self.assertTrue(v.parent is None)
        self.assertEqual(v.ctype, Var)
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.lb, -float('inf'))
        self.assertEqual(v.ub, float('inf'))
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)
        self.assertEqual(v.stale, True)
        b = block()
        b.v = v
        self.assertTrue(v.parent is b)
        del b.v
        self.assertTrue(v.parent is None)

class Test_variable_dict(_TestComponentDictBase,
                         unittest.TestCase):
    _container_type = variable_dict
    _ctype_factory = lambda self: variable()

class Test_variable_list(_TestComponentListBase,
                         unittest.TestCase):
    _container_type = variable_list
    _ctype_factory = lambda self: variable()

if __name__ == "__main__":
    unittest.main()
