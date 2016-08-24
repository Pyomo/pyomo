import pyutilib.th as unittest
from pyomo.core.base.component_interface import (IObjectWithParent,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer,
                                                 IBlockStorage)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_block import (block,
                                             block_dict,
                                             block_list)
from pyomo.core.base.block import Block

class Test_block(unittest.TestCase):

    def test_init(self):
        b = block()
        self.assertTrue(b.parent is None)
        self.assertEqual(b.ctype, Block)

    def test_type(self):
        b = block()
        self.assertTrue(isinstance(b, IObjectWithParent))
        self.assertTrue(isinstance(b, IActiveObject))
        self.assertTrue(isinstance(b, IComponentContainer))
        self.assertTrue(isinstance(b, _IActiveComponentContainer))
        self.assertTrue(isinstance(b, IBlockStorage))

class Test_block_dict(_TestActiveComponentDictBase,
                      unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: block()

class Test_block_list(_TestActiveComponentListBase,
                      unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: block()

if __name__ == "__main__":
    unittest.main()
