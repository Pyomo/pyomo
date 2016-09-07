import sys
import collections

import pyutilib.th as unittest

from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer,
                                                 IBlockStorage)
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_suffix import suffix
from pyomo.core.base.component_variable import (variable,
                                                variable_dict)
from pyomo.core.base.suffix import Suffix
from pyomo.core.base.component_block import (block,
                                             block_dict)
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

import six
from six import StringIO

class Test_suffix(unittest.TestCase):

    def test_init(self):
        s = suffix()
        self.assertTrue(s.parent is None)
        self.assertEqual(s.ctype, Suffix)
        self.assertEqual(s.direction, suffix.LOCAL)
        self.assertEqual(s.datatype, suffix.FLOAT)

    def test_type(self):
        s = suffix()
        self.assertTrue(isinstance(s, ICategorizedObject))
        self.assertTrue(isinstance(s, IActiveObject))
        self.assertTrue(isinstance(s, IComponent))
        self.assertTrue(isinstance(s, _IActiveComponent))
        self.assertTrue(isinstance(s, collections.Mapping))
        self.assertTrue(isinstance(s, collections.MutableMapping))
        self.assertTrue(issubclass(type(s), collections.Mapping))
        self.assertTrue(issubclass(type(s), collections.MutableMapping))

    def test_import_export_enabled(self):
        s = suffix()
        s.direction = suffix.LOCAL
        self.assertEqual(s.direction, suffix.LOCAL)
        self.assertEqual(s.export_enabled(), False)
        self.assertEqual(s.import_enabled(), False)
        s.direction = suffix.IMPORT
        self.assertEqual(s.direction, suffix.IMPORT)
        self.assertEqual(s.export_enabled(), False)
        self.assertEqual(s.import_enabled(), True)
        s.direction = suffix.EXPORT
        self.assertEqual(s.direction, suffix.EXPORT)
        self.assertEqual(s.export_enabled(), True)
        self.assertEqual(s.import_enabled(), False)
        s.direction = suffix.IMPORT_EXPORT
        self.assertEqual(s.direction, suffix.IMPORT_EXPORT)
        self.assertEqual(s.export_enabled(), True)
        self.assertEqual(s.import_enabled(), True)
        with self.assertRaises(ValueError):
            s.direction = 'export'

    def test_datatype(self):
        s = suffix()
        s.datatype = suffix.FLOAT
        self.assertEqual(s.datatype, suffix.FLOAT)
        s.datatype = suffix.INT
        self.assertEqual(s.datatype, suffix.INT)
        with self.assertRaises(ValueError):
            s.datatype = 'something'

    def test_clear(self):
        x = variable()
        y = variable()

        s = suffix()
        s[x] = 1.0
        s[y] = None

        self.assertEqual(len(s), 2)
        s.clear()
        self.assertEqual(len(s), 0)

    def test_del(self):
        x = variable()

        s = suffix()
        s[x] = 1.0

        self.assertEqual(len(s), 1)
        del s[x]
        self.assertEqual(len(s), 0)
        with self.assertRaises(KeyError):
            del s[x]

    def test_name(self):
        s = suffix()
        self.assertTrue(s.parent is None)
        self.assertTrue(s.parent_block is None)
        self.assertTrue(s.root_block is None)
        self.assertEqual(s.local_name, None)
        self.assertEqual(s.name, None)

        model = block()
        model.s = s
        self.assertTrue(model.parent is None)
        self.assertTrue(model.parent_block is None)
        self.assertTrue(model.root_block is model)
        self.assertTrue(s.parent is model)
        self.assertTrue(s.parent_block is model)
        self.assertTrue(s.root_block is model)
        self.assertEqual(s.local_name, "s")
        self.assertEqual(s.name, "s")

        b = block()
        b.model = model
        self.assertTrue(b.parent is None)
        self.assertTrue(b.parent_block is None)
        self.assertTrue(b.root_block is b)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is b)
        self.assertTrue(s.parent is model)
        self.assertTrue(s.parent_block is model)
        self.assertTrue(s.root_block is b)
        self.assertEqual(s.local_name, "s")
        self.assertEqual(s.name, "model.s")

        bdict = block_dict()
        bdict[0] = b
        self.assertTrue(bdict.parent is None)
        self.assertTrue(bdict.parent_block is None)
        self.assertTrue(bdict.root_block is None)
        self.assertTrue(b.parent is bdict)
        self.assertTrue(b.parent_block is None)
        self.assertTrue(b.root_block is b)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is b)
        self.assertTrue(s.parent is model)
        self.assertTrue(s.parent_block is model)
        self.assertTrue(s.root_block is b)
        self.assertEqual(s.local_name, "s")
        self.assertEqual(s.name, "[0].model.s")

        m = block()
        m.bdict = bdict
        self.assertTrue(m.parent is None)
        self.assertTrue(m.parent_block is None)
        self.assertTrue(m.root_block is m)
        self.assertTrue(bdict.parent is m)
        self.assertTrue(bdict.parent_block is m)
        self.assertTrue(bdict.root_block is m)
        self.assertTrue(b.parent is bdict)
        self.assertTrue(b.parent_block is m)
        self.assertTrue(b.root_block is m)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is m)
        self.assertTrue(s.parent is model)
        self.assertTrue(s.parent_block is model)
        self.assertTrue(s.root_block is m)
        self.assertEqual(s.local_name, "s")
        self.assertEqual(s.name, "bdict[0].model.s")

    def test_active(self):

        s = suffix()
        with self.assertRaises(AttributeError):
            s.active = False

        model = block()
        model.s = s
        b = block()
        b.model = model
        bdict = block_dict()
        bdict[0] = b
        bdict[None] = block()
        m = block()
        m.bdict = bdict

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(s.active, True)

        m.deactivate()

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(s.active, False)

        m.activate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(s.active, True)

    #
    # These methods are deprecated
    #

    def test_set_all_values(self):
        x = variable()
        y = variable()

        s = suffix()
        s[x] = 1.0
        s[y] = None

        self.assertEqual(s[x], 1.0)
        self.assertEqual(s[y], None)
        s.set_all_values(0)
        self.assertEqual(s[x], 0)
        self.assertEqual(s[y], 0)

    def test_clear_all_values(self):
        x = variable()
        y = variable()

        s = suffix()
        s[x] = 1.0
        s[y] = None

        self.assertEqual(len(s), 2)
        s.clear_all_values()
        self.assertEqual(len(s), 0)
        s.clear_all_values()

    def test_clear_value(self):
        x = variable()

        s = suffix()
        s[x] = 1.0

        self.assertEqual(len(s), 1)
        s.clear_value(x)
        self.assertEqual(len(s), 0)
        s.clear_value(x)

    def test_getset_direction(self):
        s = suffix()
        s.set_direction(suffix.LOCAL)
        self.assertEqual(s.get_direction(), suffix.LOCAL)
        s.set_direction(suffix.IMPORT)
        self.assertEqual(s.get_direction(), suffix.IMPORT)
        s.set_direction(suffix.EXPORT)
        self.assertEqual(s.get_direction(), suffix.EXPORT)
        s.set_direction(suffix.IMPORT_EXPORT)
        self.assertEqual(s.get_direction(), suffix.IMPORT_EXPORT)
        with self.assertRaises(ValueError):
            s.set_direction('export')

    def test_getset_datatype(self):
        s = suffix()
        s.set_datatype(suffix.FLOAT)
        self.assertEqual(s.get_datatype(), suffix.FLOAT)
        s.set_datatype(suffix.INT)
        self.assertEqual(s.get_datatype(), suffix.INT)
        with self.assertRaises(ValueError):
            s.set_datatype('something')

if __name__ == "__main__":
    unittest.main()
