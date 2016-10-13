import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_objective import (IObjective,
                                                 objective,
                                                 objective_dict,
                                                 objective_list,
                                                 minimize,
                                                 maximize)
from pyomo.core.base.numvalue import NumericValue
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
        self.assertEqual(o.sense, minimize)
        self.assertEqual(o.is_minimizing(), True)

    def test_sense(self):
        o = objective()
        o.sense = maximize
        self.assertEqual(o.sense, maximize)
        self.assertEqual(o.is_minimizing(), False)
        with self.assertRaises(ValueError):
            o.sense = 100
        self.assertEqual(o.sense, maximize)
        self.assertEqual(o.is_minimizing(), False)

    def test_type(self):
        o = objective()
        self.assertTrue(isinstance(o, ICategorizedObject))
        self.assertTrue(isinstance(o, IActiveObject))
        self.assertTrue(isinstance(o, IComponent))
        self.assertTrue(isinstance(o, _IActiveComponent))
        self.assertTrue(isinstance(o, IObjective))
        self.assertTrue(isinstance(o, NumericValue))

class Test_objective_dict(_TestActiveComponentDictBase,
                          unittest.TestCase):
    _container_type = objective_dict
    _ctype_factory = lambda self: objective()

class Test_objective_list(_TestActiveComponentListBase,
                          unittest.TestCase):
    _container_type = objective_list
    _ctype_factory = lambda self: objective()

if __name__ == "__main__":
    unittest.main()
