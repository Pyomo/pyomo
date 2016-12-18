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
from pyomo.core.base.numvalue import (NumericValue,
                                      is_fixed,
                                      is_constant,
                                      potentially_variable)
from pyomo.core.base.component_parameter import parameter
from pyomo.core.base.component_variable import \
    (IVariable,
     variable,
     variable_dict,
     variable_list,
     _extract_domain_type_and_bounds)
from pyomo.core.base.var import Var
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet,
                                       Binary,
                                       NonNegativeReals,
                                       NegativeReals,
                                       Reals,
                                       RealInterval,
                                       Integers,
                                       NonNegativeIntegers,
                                       NegativeIntegers,
                                       IntegerInterval,
                                       BooleanSet)

import six
from six import StringIO

class Test_variable(unittest.TestCase):

    def test_extract_domain_type_and_bounds(self):
        # test an edge case
        domain_type, lb, ub = _extract_domain_type_and_bounds(None, None, None, None)
        self.assertIs(domain_type, RealSet)
        self.assertIs(lb, None)
        self.assertIs(ub, None)

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
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
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

    def test_is_constant(self):
        v = variable()
        self.assertEqual(v.is_constant(), False)
        self.assertEqual(is_constant(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)
        v.value = 1.0
        self.assertEqual(v.is_constant(), False)
        self.assertEqual(is_constant(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, 1.0)
        v.fix()
        self.assertEqual(v.is_constant(), False)
        self.assertEqual(is_constant(v), False)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1.0)
        v.value = None
        self.assertEqual(v.is_constant(), False)
        self.assertEqual(is_constant(v), False)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, None)
        v.free()
        self.assertEqual(v.is_constant(), False)
        self.assertEqual(is_constant(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)

    def test_is_fixed(self):
        v = variable()
        self.assertEqual(v.is_fixed(), False)
        self.assertEqual(is_fixed(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)
        v.value = 1.0
        self.assertEqual(v.is_fixed(), False)
        self.assertEqual(is_fixed(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, 1.0)
        v.fix()
        self.assertEqual(v.is_fixed(), True)
        self.assertEqual(is_fixed(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1.0)
        v.value = None
        self.assertEqual(v.is_fixed(), True)
        self.assertEqual(is_fixed(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, None)
        v.free()
        self.assertEqual(v.is_fixed(), False)
        self.assertEqual(is_fixed(v), False)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)

    def test_potentially_variable(self):
        v = variable()
        self.assertEqual(v._potentially_variable(), True)
        self.assertEqual(potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)
        v.value = 1.0
        self.assertEqual(v._potentially_variable(), True)
        self.assertEqual(potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, 1.0)
        v.fix()
        self.assertEqual(v._potentially_variable(), True)
        self.assertEqual(potentially_variable(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1.0)
        v.value = None
        self.assertEqual(v._potentially_variable(), True)
        self.assertEqual(potentially_variable(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, None)
        v.free()
        self.assertEqual(v._potentially_variable(), True)
        self.assertEqual(potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)

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

    def test_domain(self):
        v = variable(domain=Reals)
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (None, None))

        v = variable(domain=Reals, lb=0, ub=1)
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0, 1))

        lb_param = parameter()
        ub_param = parameter()
        lb = lb_param
        ub = ub_param
        v = variable(domain=Reals, lb=lb, ub=ub)
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertIs(v.lb, lb)
        self.assertIs(v.ub, ub)
        self.assertIs(v.bounds[0], lb)
        self.assertIs(v.bounds[1], ub)

        lb = lb**2
        ub = ub**2
        v.lb = lb
        v.ub = ub
        lb_param.value = 2.0
        ub_param.value = 3.0
        self.assertIs(v.lb, lb)
        self.assertEqual(v.lb(), 4.0)
        self.assertIs(v.ub, ub)
        self.assertEqual(v.ub(), 9.0)
        self.assertIs(v.bounds[0], lb)
        self.assertEqual(v.bounds[0](), 4.0)
        self.assertIs(v.bounds[1], ub)
        self.assertEqual(v.bounds[1](), 9.0)

        # if the domain has a finite bound
        # then you can not use the same bound
        # keyword
        variable(domain=NonNegativeReals, ub=0)
        with self.assertRaises(ValueError):
            variable(domain=NonNegativeReals, lb=0)

        variable(domain=NonNegativeReals, ub=ub)
        with self.assertRaises(ValueError):
            variable(domain=NonNegativeReals, lb=lb)

        variable(domain=NonNegativeIntegers, ub=0)
        with self.assertRaises(ValueError):
            variable(domain=NonNegativeIntegers, lb=0)

        variable(domain=NonNegativeIntegers, ub=ub)
        with self.assertRaises(ValueError):
            variable(domain=NonNegativeIntegers, lb=lb)

        variable(domain=NegativeReals, lb=0)
        with self.assertRaises(ValueError):
            variable(domain=NegativeReals, ub=0)

        variable(domain=NegativeReals, lb=lb)
        with self.assertRaises(ValueError):
            variable(domain=NegativeReals, ub=ub)

        variable(domain=NegativeIntegers, lb=0)
        with self.assertRaises(ValueError):
            variable(domain=NegativeIntegers, ub=0)

        variable(domain=NegativeIntegers, lb=lb)
        with self.assertRaises(ValueError):
            variable(domain=NegativeIntegers, ub=ub)

        unit_interval = RealInterval(bounds=(0,1))
        self.assertEqual(unit_interval.bounds(), (0,1))
        v = variable(domain=unit_interval)
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0, 1))
        with self.assertRaises(ValueError):
            variable(domain=unit_interval, lb=0)
        with self.assertRaises(ValueError):
            variable(domain=unit_interval, ub=0)

        v = variable()
        v.domain = unit_interval
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0, 1))

        binary = IntegerInterval(bounds=(0,1))
        self.assertEqual(binary.bounds(), (0,1))
        v = variable(domain=binary)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0, 1))
        with self.assertRaises(ValueError):
            variable(domain=binary, lb=0)
        with self.assertRaises(ValueError):
            variable(domain=binary, ub=0)

        v = variable()
        v.domain = binary
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0, 1))

        variable(domain_type=RealSet)
        variable(domain=Reals)
        with self.assertRaises(ValueError):
            variable(domain_type=RealSet,
                     domain=Reals)
        with self.assertRaises(ValueError):
            variable(domain_type=BooleanSet)

    def test_domain_type(self):
        v = variable()
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (None, None))

        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (None, None))

        v = variable(domain_type=IntegerSet, lb=0)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (0, None))

        v = variable(domain_type=IntegerSet, ub=0)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, 0)
        self.assertEqual(v.bounds, (None, 0))

        v.domain_type = RealSet
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, 0)
        self.assertEqual(v.bounds, (None, 0))

        with self.assertRaises(ValueError):
            v.domain_type = BooleanSet

    def test_binary_type(self):
        v = variable()
        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (None, None))

        v.lb = 0
        v.ub = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.lb = 0
        v.ub = 0
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 0)
        self.assertEqual(v.bounds, (0,0))

        v.lb = 1
        v.ub = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 1)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (1,1))

        v = variable(domain=Binary)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.ub = 2
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 2)
        self.assertEqual(v.bounds, (0,2))

        v.lb = -1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, -1)
        self.assertEqual(v.ub, 2)
        self.assertEqual(v.bounds, (-1,2))

        v.domain = Binary
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.domain_type = RealSet
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 1)
        self.assertEqual(v.bounds, (0,1))

        v.domain = Reals
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.ub, None)
        self.assertEqual(v.bounds, (None, None))

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
