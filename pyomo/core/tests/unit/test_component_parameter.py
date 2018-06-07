import pickle

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import (NumericValue,
                                        is_fixed,
                                        is_constant,
                                        is_potentially_variable)
import pyomo.kernel
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_tuple import \
    _TestComponentTupleBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.kernel.component_interface import (ICategorizedObject,
                                                   IComponent)
from pyomo.core.kernel.component_parameter import (IParameter,
                                                   parameter,
                                                   parameter_dict,
                                                   parameter_tuple,
                                                   parameter_list)
from pyomo.core.kernel.component_variable import variable
from pyomo.core.kernel.component_block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)
from pyomo.core.base.param import Param

class Test_parameter(unittest.TestCase):

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        p = parameter()
        pyomo.kernel.pprint(p)
        b = block()
        b.p = p
        pyomo.kernel.pprint(p)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(p)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_ctype(self):
        p = parameter()
        self.assertIs(p.ctype, Param)
        self.assertIs(type(p).ctype, Param)
        self.assertIs(parameter.ctype, Param)

    def test_pickle(self):
        p = parameter(value=1.0)
        self.assertEqual(p.value, 1.0)
        self.assertEqual(p.parent, None)
        pup = pickle.loads(
            pickle.dumps(p))
        self.assertEqual(pup.value, 1.0)
        self.assertEqual(pup.parent, None)
        b = block()
        b.p = p
        self.assertIs(p.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        pup = bup.p
        self.assertEqual(pup.value, 1.0)
        self.assertIs(pup.parent, bup)

    def test_init(self):
        p = parameter()
        self.assertTrue(p.parent is None)
        self.assertEqual(p.ctype, Param)
        self.assertEqual(p.value, None)
        self.assertEqual(p(), None)
        p.value = 1
        self.assertEqual(p.value, 1)
        self.assertEqual(p(), 1)

    def test_type(self):
        p = parameter()
        self.assertTrue(isinstance(p, ICategorizedObject))
        self.assertTrue(isinstance(p, IComponent))
        self.assertTrue(isinstance(p, IParameter))
        self.assertTrue(isinstance(p, NumericValue))

    def test_is_constant(self):
        p = parameter()
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)
        p.value = 1.0
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)

    def test_is_fixed(self):
        p = parameter()
        self.assertEqual(p.is_fixed(), True)
        self.assertEqual(is_fixed(p), True)
        p.value = 1.0
        self.assertEqual(p.is_fixed(), True)
        self.assertEqual(is_fixed(p), True)

    def test_potentially_variable(self):
        p = parameter()
        self.assertEqual(p.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(p), False)
        p.value = 1.0
        self.assertEqual(p.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(p), False)

    def test_polynomial_degree(self):
        p = parameter()
        self.assertEqual(p.polynomial_degree(), 0)
        self.assertEqual((p**2).polynomial_degree(), 0)
        self.assertEqual(p.value, None)
        with self.assertRaises(ValueError):
            (p**2)()
        p.value = 1.0
        self.assertEqual(p.polynomial_degree(), 0)
        self.assertEqual((p**2).polynomial_degree(), 0)
        self.assertEqual(p.value, 1.0)
        self.assertEqual((p**2)(), 1.0)

class Test_parameter_dict(_TestComponentDictBase,
                          unittest.TestCase):
    _container_type = parameter_dict
    _ctype_factory = lambda self: parameter()

class Test_parameter_tuple(_TestComponentTupleBase,
                           unittest.TestCase):
    _container_type = parameter_tuple
    _ctype_factory = lambda self: parameter()

class Test_parameter_list(_TestComponentListBase,
                           unittest.TestCase):
    _container_type = parameter_list
    _ctype_factory = lambda self: parameter()

if __name__ == "__main__":
    unittest.main()
