import pyutilib.th as unittest
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_objective import (objective,
                                                 objective_dict,
                                                 objective_list)
from pyomo.core.base.component_variable import variable
from pyomo.core.base.objective import Objective
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

class Test_objective(unittest.TestCase):

    def test_init(self):
        o = objective()
        self.assertTrue(o.parent is None)
        self.assertEqual(o.ctype, Objective)
        self.assertEqual(o.expr, None)

class Test_objective_dict(_TestComponentDictBase,
                          unittest.TestCase):
    _container_type = objective_dict
    _ctype_factory = lambda self: objective()

class Test_objective_list(_TestComponentListBase,
                          unittest.TestCase):
    _container_type = objective_list
    _ctype_factory = lambda self: objective()

if __name__ == "__main__":
    unittest.main()
