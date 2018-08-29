import pickle

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import (NumericValue,
                                      is_fixed,
                                      is_constant,
                                      is_potentially_variable)
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import (IParameter,
                                                   parameter,
                                                   parameter_dict,
                                                   parameter_tuple,
                                                   parameter_list)
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)

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
        self.assertIs(p.ctype, IParameter)
        self.assertIs(type(p), parameter)
        self.assertIs(type(p)._ctype, IParameter)

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
        self.assertEqual(p.ctype, IParameter)
        self.assertEqual(p.value, None)
        self.assertEqual(p(), None)
        p.value = 1
        self.assertEqual(p.value, 1)
        self.assertEqual(p(), 1)

    def test_type(self):
        p = parameter()
        self.assertTrue(isinstance(p, ICategorizedObject))
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

    def test_is_expression_type(self):
        p = parameter()
        self.assertEqual(p.is_expression_type(), False)

    def test_is_parameter_type(self):
        p = parameter()
        # GH: apparently is_parameter_type has something
        #     to do with mutability...
        self.assertEqual(p.is_parameter_type(), False)

class Test_parameter_dict(_TestActiveDictContainerBase,
                          unittest.TestCase):
    _container_type = parameter_dict
    _ctype_factory = lambda self: parameter()

class Test_parameter_tuple(_TestActiveTupleContainerBase,
                           unittest.TestCase):
    _container_type = parameter_tuple
    _ctype_factory = lambda self: parameter()

class Test_parameter_list(_TestActiveListContainerBase,
                           unittest.TestCase):
    _container_type = parameter_list
    _ctype_factory = lambda self: parameter()

if __name__ == "__main__":
    unittest.main()
