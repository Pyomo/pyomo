import sys
import collections
import pickle

import pyutilib.th as unittest
import pyomo.kernel
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.kernel.component_interface import (ICategorizedObject,
                                                   IComponent,
                                                   _ActiveObjectMixin,
                                                   IComponentContainer)
from pyomo.core.kernel.component_suffix import (suffix,
                                                export_suffix_generator,
                                                import_suffix_generator,
                                                local_suffix_generator,
                                                suffix_generator)
from pyomo.core.kernel.component_variable import (variable,
                                                  variable_dict)
from pyomo.core.kernel.component_constraint import (constraint,
                                                    constraint_list)
from pyomo.core.kernel.component_block import (block,
                                               block_dict)
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)
from pyomo.core.base.suffix import Suffix

import six
from six import StringIO

class Test_suffix(unittest.TestCase):

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        clist = constraint_list([constraint()])
        s = suffix()
        s[v] = 1
        s[clist] = None
        pyomo.kernel.pprint(s)
        b = block()
        b.s = s
        pyomo.kernel.pprint(s)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(s)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

        pyomo.kernel.pprint({'a': 1, 'b': 2})

    def test_str(self):
        s = suffix()
        self.assertEqual(str(s), "<suffix>")
        b = block()
        b.s = s
        self.assertEqual(str(s), "s")

    def test_ctype(self):
        s = suffix()
        self.assertIs(s.ctype, Suffix)
        self.assertIs(type(s).ctype, Suffix)
        self.assertIs(suffix.ctype, Suffix)

    def test_pickle(self):
        s = suffix(direction=suffix.EXPORT,
                   datatype=suffix.FLOAT)
        self.assertEqual(s.direction, suffix.EXPORT)
        self.assertEqual(s.datatype, suffix.FLOAT)
        self.assertEqual(s.parent, None)
        sup = pickle.loads(
            pickle.dumps(s))
        self.assertEqual(sup.direction, suffix.EXPORT)
        self.assertEqual(sup.datatype, suffix.FLOAT)
        self.assertEqual(sup.parent, None)
        b = block()
        b.s = s
        self.assertIs(s.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        sup = bup.s
        self.assertEqual(sup.direction, suffix.EXPORT)
        self.assertEqual(sup.datatype, suffix.FLOAT)
        self.assertIs(sup.parent, bup)
        b.v = variable(lb=1)
        b.s[b.v] = 1.0
        bup = pickle.loads(
            pickle.dumps(b))
        sup = bup.s
        vup = bup.v
        self.assertEqual(sup[vup], 1.0)

    def test_init(self):
        s = suffix()
        self.assertTrue(s.parent is None)
        self.assertEqual(s.ctype, Suffix)
        self.assertEqual(s.direction, suffix.LOCAL)
        self.assertEqual(s.datatype, suffix.FLOAT)

    def test_type(self):
        s = suffix()
        self.assertTrue(isinstance(s, ICategorizedObject))
        self.assertTrue(isinstance(s, IComponent))
        self.assertTrue(isinstance(s, _ActiveObjectMixin))
        self.assertTrue(isinstance(s, collections.Mapping))
        self.assertTrue(isinstance(s, collections.MutableMapping))
        self.assertTrue(issubclass(type(s), collections.Mapping))
        self.assertTrue(issubclass(type(s), collections.MutableMapping))

    def test_import_export_enabled(self):
        s = suffix()
        s.direction = suffix.LOCAL
        self.assertEqual(s.direction, suffix.LOCAL)
        self.assertEqual(s.export_enabled, False)
        self.assertEqual(s.import_enabled, False)
        s.direction = suffix.IMPORT
        self.assertEqual(s.direction, suffix.IMPORT)
        self.assertEqual(s.export_enabled, False)
        self.assertEqual(s.import_enabled, True)
        s.direction = suffix.EXPORT
        self.assertEqual(s.direction, suffix.EXPORT)
        self.assertEqual(s.export_enabled, True)
        self.assertEqual(s.import_enabled, False)
        s.direction = suffix.IMPORT_EXPORT
        self.assertEqual(s.direction, suffix.IMPORT_EXPORT)
        self.assertEqual(s.export_enabled, True)
        self.assertEqual(s.import_enabled, True)
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

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(s.active, False)

        m.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(s.active, True)

        m.deactivate()

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(s.active, True)

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(s.active, False)

        m.activate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(s.active, False)

        m.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(s.active, True)

    def test_export_suffix_generator(self):
        m = block()
        m.s0 = suffix(direction=suffix.LOCAL)
        m.s0i = suffix(direction=suffix.LOCAL,
                      datatype=suffix.INT)
        m.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                       datatype=suffix.INT)
        m.s2 = suffix(direction=suffix.IMPORT)
        m.s2i = suffix(direction=suffix.IMPORT,
                      datatype=suffix.INT)
        m.s3 = suffix(direction=suffix.EXPORT)
        m.s3i = suffix(direction=suffix.EXPORT,
                       datatype=suffix.INT)
        m.b = block()
        m.b.s0 = suffix(direction=suffix.LOCAL)
        m.b.s0i = suffix(direction=suffix.LOCAL,
                         datatype=suffix.INT)
        m.b.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.b.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                         datatype=suffix.INT)
        m.b.s2 = suffix(direction=suffix.IMPORT)
        m.b.s2i = suffix(direction=suffix.IMPORT,
                         datatype=suffix.INT)
        m.b.s3 = suffix(direction=suffix.EXPORT)
        m.b.s3i = suffix(direction=suffix.EXPORT,
                         datatype=suffix.INT)
        # default
        self.assertEqual([id(c_) for c_
                          in export_suffix_generator(m)],
                         [id(m.s1), id(m.s1i),
                          id(m.s3), id(m.s3i),
                          id(m.b.s1), id(m.b.s1i),
                          id(m.b.s3), id(m.b.s3i)])
        # descend_into=False
        self.assertEqual([id(c_) for c_
                          in export_suffix_generator(m,
                                                     descend_into=False)],
                         [id(m.s1), id(m.s1i),
                          id(m.s3), id(m.s3i)])
        # datatype=INT
        self.assertEqual([id(c_) for c_
                          in export_suffix_generator(m,
                                                     datatype=suffix.INT)],
                         [id(m.s1i),
                          id(m.s3i),
                          id(m.b.s1i),
                          id(m.b.s3i)])
        # active=True
        m.s1.deactivate()
        m.b.deactivate()
        self.assertEqual([id(c_) for c_ in export_suffix_generator(m,
                                                                   active=True)],
                         [id(m.s1i), id(m.s3), id(m.s3i)])

    def test_import_suffix_generator(self):
        m = block()
        m.s0 = suffix(direction=suffix.LOCAL)
        m.s0i = suffix(direction=suffix.LOCAL,
                      datatype=suffix.INT)
        m.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                       datatype=suffix.INT)
        m.s2 = suffix(direction=suffix.IMPORT)
        m.s2i = suffix(direction=suffix.IMPORT,
                      datatype=suffix.INT)
        m.s3 = suffix(direction=suffix.EXPORT)
        m.s3i = suffix(direction=suffix.EXPORT,
                       datatype=suffix.INT)
        m.b = block()
        m.b.s0 = suffix(direction=suffix.LOCAL)
        m.b.s0i = suffix(direction=suffix.LOCAL,
                         datatype=suffix.INT)
        m.b.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.b.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                         datatype=suffix.INT)
        m.b.s2 = suffix(direction=suffix.IMPORT)
        m.b.s2i = suffix(direction=suffix.IMPORT,
                         datatype=suffix.INT)
        m.b.s3 = suffix(direction=suffix.EXPORT)
        m.b.s3i = suffix(direction=suffix.EXPORT,
                         datatype=suffix.INT)
        # default
        self.assertEqual([id(c_) for c_
                          in import_suffix_generator(m)],
                         [id(m.s1), id(m.s1i),
                          id(m.s2), id(m.s2i),
                          id(m.b.s1), id(m.b.s1i),
                          id(m.b.s2), id(m.b.s2i)])
        # descend_into=False
        self.assertEqual([id(c_) for c_
                          in import_suffix_generator(m,
                                                     descend_into=False)],
                         [id(m.s1), id(m.s1i),
                          id(m.s2), id(m.s2i)])
        # datatype=INT
        self.assertEqual([id(c_) for c_
                          in import_suffix_generator(m,
                                                     datatype=suffix.INT)],
                         [id(m.s1i),
                          id(m.s2i),
                          id(m.b.s1i),
                          id(m.b.s2i)])
        # active=True
        m.s1.deactivate()
        m.b.deactivate()
        self.assertEqual([id(c_) for c_
                          in import_suffix_generator(m,
                                                     active=True)],
                         [id(m.s1i), id(m.s2), id(m.s2i)])

    def test_local_suffix_generator(self):
        m = block()
        m.s0 = suffix(direction=suffix.LOCAL)
        m.s0i = suffix(direction=suffix.LOCAL,
                      datatype=suffix.INT)
        m.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                       datatype=suffix.INT)
        m.s2 = suffix(direction=suffix.IMPORT)
        m.s2i = suffix(direction=suffix.IMPORT,
                      datatype=suffix.INT)
        m.s3 = suffix(direction=suffix.EXPORT)
        m.s3i = suffix(direction=suffix.EXPORT,
                       datatype=suffix.INT)
        m.b = block()
        m.b.s0 = suffix(direction=suffix.LOCAL)
        m.b.s0i = suffix(direction=suffix.LOCAL,
                         datatype=suffix.INT)
        m.b.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.b.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                         datatype=suffix.INT)
        m.b.s2 = suffix(direction=suffix.IMPORT)
        m.b.s2i = suffix(direction=suffix.IMPORT,
                         datatype=suffix.INT)
        m.b.s3 = suffix(direction=suffix.EXPORT)
        m.b.s3i = suffix(direction=suffix.EXPORT,
                         datatype=suffix.INT)
        # default
        self.assertEqual([id(c_) for c_
                          in local_suffix_generator(m)],
                         [id(m.s0), id(m.s0i),
                          id(m.b.s0), id(m.b.s0i)])
        # descend_into=False
        self.assertEqual([id(c_) for c_
                          in local_suffix_generator(m,
                                                    descend_into=False)],
                         [id(m.s0), id(m.s0i)])
        # datatype=INT
        self.assertEqual([id(c_) for c_
                          in local_suffix_generator(m,
                                                    datatype=suffix.INT)],
                         [id(m.s0i),
                          id(m.b.s0i)])
        # active=True
        m.s0.deactivate()
        m.b.deactivate()
        self.assertEqual([id(c_) for c_
                          in local_suffix_generator(m,
                                                    active=True)],
                         [id(m.s0i)])

    def test_suffix_generator(self):
        m = block()
        m.s0 = suffix(direction=suffix.LOCAL)
        m.s0i = suffix(direction=suffix.LOCAL,
                      datatype=suffix.INT)
        m.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                       datatype=suffix.INT)
        m.s2 = suffix(direction=suffix.IMPORT)
        m.s2i = suffix(direction=suffix.IMPORT,
                      datatype=suffix.INT)
        m.s3 = suffix(direction=suffix.EXPORT)
        m.s3i = suffix(direction=suffix.EXPORT,
                       datatype=suffix.INT)
        m.b = block()
        m.b.s0 = suffix(direction=suffix.LOCAL)
        m.b.s0i = suffix(direction=suffix.LOCAL,
                         datatype=suffix.INT)
        m.b.s1 = suffix(direction=suffix.IMPORT_EXPORT)
        m.b.s1i = suffix(direction=suffix.IMPORT_EXPORT,
                         datatype=suffix.INT)
        m.b.s2 = suffix(direction=suffix.IMPORT)
        m.b.s2i = suffix(direction=suffix.IMPORT,
                         datatype=suffix.INT)
        m.b.s3 = suffix(direction=suffix.EXPORT)
        m.b.s3i = suffix(direction=suffix.EXPORT,
                         datatype=suffix.INT)
        # default
        self.assertEqual([id(c_) for c_
                          in suffix_generator(m)],
                         [id(m.s0), id(m.s0i),
                          id(m.s1), id(m.s1i),
                          id(m.s2), id(m.s2i),
                          id(m.s3), id(m.s3i),
                          id(m.b.s0), id(m.b.s0i),
                          id(m.b.s1), id(m.b.s1i),
                          id(m.b.s2), id(m.b.s2i),
                          id(m.b.s3), id(m.b.s3i)])
        # descend_into=False
        self.assertEqual([id(c_) for c_
                          in suffix_generator(m,
                                              descend_into=False)],
                         [id(m.s0), id(m.s0i),
                          id(m.s1), id(m.s1i),
                          id(m.s2), id(m.s2i),
                          id(m.s3), id(m.s3i)])
        # datatype=INT
        self.assertEqual([id(c_) for c_
                          in suffix_generator(m,
                                              datatype=suffix.INT)],
                         [id(m.s0i),
                          id(m.s1i),
                          id(m.s2i),
                          id(m.s3i),
                          id(m.b.s0i),
                          id(m.b.s1i),
                          id(m.b.s2i),
                          id(m.b.s3i)])
        # active=True
        m.s1.deactivate()
        m.b.deactivate()
        self.assertEqual([id(c_) for c_
                          in suffix_generator(m,
                                              active=True)],
                         [id(m.s0),
                          id(m.s0i),
                          id(m.s1i),
                          id(m.s2),
                          id(m.s2i),
                          id(m.s3),
                          id(m.s3i)])

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
