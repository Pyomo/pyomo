import pickle

import pyutilib.th as unittest
import pyomo.kernel
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_tuple import \
    _TestActiveComponentTupleBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.kernel.component_interface import (ICategorizedObject,
                                                   IComponent,
                                                   _ActiveObjectMixin)
from pyomo.core.kernel.component_sos import (ISOS,
                                             sos,
                                             sos1,
                                             sos2,
                                             sos_dict,
                                             sos_tuple,
                                             sos_list)
from pyomo.core.kernel.component_block import block
from pyomo.core.kernel.component_variable import (variable,
                                                  variable_list)
from pyomo.core.kernel.component_parameter import parameter
from pyomo.core.kernel.component_expression import (expression,
                                                    data_expression)
from pyomo.core.base.sos import SOSConstraint

class Test_sos(unittest.TestCase):

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        vlist = variable_list([variable(), variable()])
        s = sos(vlist)
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

    def test_ctype(self):
        s = sos([])
        self.assertIs(s.ctype, SOSConstraint)
        self.assertIs(type(s).ctype, SOSConstraint)
        self.assertIs(sos.ctype, SOSConstraint)

    def test_pickle(self):
        v = variable()
        s = sos([v],weights=[1])
        self.assertEqual(len(s), 1)
        self.assertIs(s.variables[0], v)
        self.assertTrue(v in s)
        self.assertEqual(s.weights[0], 1)
        self.assertEqual(s.level, 1)
        self.assertEqual(s.parent, None)
        sup = pickle.loads(
            pickle.dumps(s))
        self.assertEqual(len(sup), 1)
        self.assertIsNot(sup.variables[0], v)
        self.assertFalse(v in sup)
        self.assertEqual(sup.weights[0], 1)
        self.assertEqual(sup.level, 1)
        self.assertEqual(sup.parent, None)

        b = block()
        b.v = v
        self.assertIs(v.parent, b)
        b.s = s
        self.assertIs(s.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        sup = bup.s
        self.assertEqual(len(sup), 1)
        self.assertIs(sup.variables[0], bup.v)
        self.assertTrue(bup.v in sup)
        self.assertEqual(sup.weights[0], 1)
        self.assertEqual(sup.level, 1)
        self.assertIs(sup.parent, bup)

    def test_init(self):
        s = sos([])
        self.assertTrue(s.parent is None)
        self.assertEqual(s.ctype, SOSConstraint)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.variables, ())
        self.assertEqual(s.weights, ())
        self.assertEqual(s.level, 1)

        vlist = tuple([variable(), variable()])
        s = sos(vlist)
        self.assertTrue(s.parent is None)
        self.assertEqual(s.ctype, SOSConstraint)
        self.assertEqual(len(s), 2)
        self.assertEqual(len(s.variables), 2)
        for v in vlist:
            self.assertTrue(v in s)
        self.assertEqual(s.weights, tuple([1,2]))
        self.assertEqual(s.level, 1)

        vlist = tuple([variable(), variable()])
        s = sos(vlist, weights=[3.5,4.5], level=2)
        self.assertTrue(s.parent is None)
        self.assertEqual(s.ctype, SOSConstraint)
        self.assertEqual(len(s), 2)
        self.assertEqual(len(s.variables), 2)
        for v in vlist:
            self.assertTrue(v in s)
        self.assertEqual(s.weights, tuple([3.5, 4.5]))
        self.assertEqual(s.level, 2)
        for i, (v,w) in enumerate(s.items()):
            self.assertIs(v, vlist[i])
            self.assertEqual(w, s.weights[i])

    def test_type(self):
        s = sos([])
        self.assertTrue(isinstance(s, ICategorizedObject))
        self.assertTrue(isinstance(s, IComponent))
        self.assertTrue(isinstance(s, _ActiveObjectMixin))
        self.assertTrue(isinstance(s, ISOS))

        s = sos1([])
        self.assertTrue(isinstance(s, ICategorizedObject))
        self.assertTrue(isinstance(s, IComponent))
        self.assertTrue(isinstance(s, _ActiveObjectMixin))
        self.assertTrue(isinstance(s, ISOS))

        s = sos2([])
        self.assertTrue(isinstance(s, ICategorizedObject))
        self.assertTrue(isinstance(s, IComponent))
        self.assertTrue(isinstance(s, _ActiveObjectMixin))
        self.assertTrue(isinstance(s, ISOS))

    def test_bad_weights(self):
        v = variable()
        with self.assertRaises(ValueError):
            s = sos([v], weights=[v])

        v.fix(1.0)
        with self.assertRaises(ValueError):
            s = sos([v], weights=[v])

        e = expression()
        with self.assertRaises(ValueError):
            s = sos([v], weights=[e])

        de = data_expression()
        s = sos([v], weights=[de])

        p = parameter()
        s = sos([v], weights=[p])

    def test_active(self):
        s = sos([])
        self.assertEqual(s.active, True)
        s.deactivate()
        self.assertEqual(s.active, False)
        s.activate()
        self.assertEqual(s.active, True)

        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.s = s
        self.assertEqual(s.active, True)
        self.assertEqual(b.active, False)
        s.deactivate()
        self.assertEqual(s.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(s.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(s.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(s.active, False)
        self.assertEqual(b.active, False)

class Test_sos_dict(_TestActiveComponentDictBase,
                    unittest.TestCase):
    _container_type = sos_dict
    _ctype_factory = lambda self: sos([variable()])

class Test_sos_tuple(_TestActiveComponentTupleBase,
                     unittest.TestCase):
    _container_type = sos_tuple
    _ctype_factory = lambda self: sos([variable()])

class Test_sos_list(_TestActiveComponentListBase,
                    unittest.TestCase):
    _container_type = sos_list
    _ctype_factory = lambda self: sos([variable()])

if __name__ == "__main__":
    unittest.main()
