import sys
import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.component_variable import (IVariable,
                                                variable,
                                                variable_dict,
                                                variable_list)
from pyomo.core.base.var import Var
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

import six
from six import StringIO

class Test_variable(unittest.TestCase):

    def test_pickle(self):
        v = variable(lb=1,
                     ub=2,
                     domain_type=IntegerSet,
                     fixed=True)
        self.assertEqual(v.lb, 1)
        self.assertEqual(v.ub, 2)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.parent, None)
        vup = pickle.loads(
            pickle.dumps(v))
        self.assertEqual(vup.lb, 1)
        self.assertEqual(vup.ub, 2)
        self.assertEqual(vup.domain_type, IntegerSet)
        self.assertEqual(vup.fixed, True)
        self.assertEqual(vup.parent, None)
        b = block()
        b.v = v
        self.assertIs(v.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        vup = bup.v
        self.assertEqual(vup.lb, 1)
        self.assertEqual(vup.ub, 2)
        self.assertEqual(vup.domain_type, IntegerSet)
        self.assertEqual(vup.fixed, True)
        self.assertIs(vup.parent, bup)

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

        v = variable(domain_type=IntegerSet,
                     value=1,
                     lb=0,
                     ub=2,
                     fixed=True)
        self.assertTrue(v.parent is None)
        self.assertEqual(v.ctype, Var)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 2)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1)
        self.assertEqual(v.stale, True)

    def test_type(self):
        v = variable()
        self.assertTrue(isinstance(v, ICategorizedObject))
        self.assertTrue(isinstance(v, IComponent))
        self.assertTrue(isinstance(v, IVariable))
        self.assertTrue(isinstance(v, NumericValue))

    def test_to_string(self):
        b = block()
        v = variable()
        self.assertEqual(str(v), "<variable>")
        v.to_string()
        out = StringIO()
        v.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "<variable>")
        v.to_string(verbose=False)
        out = StringIO()
        v.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "<variable>")
        v.to_string(verbose=True)
        out = StringIO()
        v.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "<variable>")

        b.v = v
        self.assertEqual(str(v), "v")
        v.to_string()
        out = StringIO()
        v.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "v")
        v.to_string(verbose=False)
        out = StringIO()
        v.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "v")
        v.to_string(verbose=True)
        out = StringIO()
        v.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "v")

    def test_domain_type(self):
        v = variable()
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)

        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)

    def test_binary_type(self):
        v = variable()
        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, -float('inf'))
        self.assertEqual(v.ub, float('inf'))

        v.lb = 0
        v.ub = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.lb = 0
        v.ub = 0
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 0)
        self.assertEqual(v.bounds, (0,0))

        v.lb = 1
        v.ub = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 1)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (1,1))

    def test_bounds_setter(self):
        v = variable()
        v.lb = 0
        v.ub = 1
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.bounds = (2, 3)
        self.assertEqual(v.lb, 2)
        self.assertEqual(v.ub, 3)
        self.assertEqual(v.bounds, (2,3))

        v.lb = -1
        v.ub = 0
        self.assertEqual(v.lb, -1)
        self.assertEqual(v.ub, 0)
        self.assertEqual(v.bounds, (-1,0))

    def test_fix_free(self):
        v = variable()
        self.assertEqual(v.value, None)
        self.assertEqual(v.fixed, False)

        v.fix(1)
        self.assertEqual(v.value, 1)
        self.assertEqual(v.fixed, True)

        v.free()
        self.assertEqual(v.value, 1)
        self.assertEqual(v.fixed, False)

        v.value = 0
        self.assertEqual(v.value, 0)
        self.assertEqual(v.fixed, False)

        v.fix()
        self.assertEqual(v.value, 0)
        self.assertEqual(v.fixed, True)

        with self.assertRaises(TypeError):
            v.fix(1,2)
        self.assertEqual(v.value, 0)
        self.assertEqual(v.fixed, True)

        v.free()
        with self.assertRaises(TypeError):
            v.fix(1,2)
        self.assertEqual(v.value, 0)
        self.assertEqual(v.fixed, False)

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
