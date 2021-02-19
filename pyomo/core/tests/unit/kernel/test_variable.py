#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import (NumericValue,
                                      is_fixed,
                                      is_constant,
                                      is_potentially_variable)
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.variable import \
    (IVariable,
     variable,
     variable_dict,
     variable_tuple,
     variable_list,
     _extract_domain_type_and_bounds)
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)
from pyomo.core.base.set import(Binary,
                                NonNegativeReals,
                                NegativeReals,
                                Reals,
                                NonNegativeIntegers,
                                NegativeIntegers,
                                RealInterval,
                                IntegerInterval)

class Test_variable(unittest.TestCase):

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        pyomo.kernel.pprint(v)
        b = block()
        b.v = v
        pyomo.kernel.pprint(v)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(v)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_bad_bounds(self):
        v = variable()
        self.assertIs(v.lb, None)
        self.assertIs(v.ub, None)
        v.lb = 1.0
        v.ub = 1.0
        self.assertEqual(v.lb, 1.0)
        self.assertEqual(v.ub, 1.0)
        with self.assertRaises(ValueError):
            v.lb = "string"
        self.assertEqual(v.lb, 1.0)
        self.assertEqual(v.ub, 1.0)
        with self.assertRaises(ValueError):
            v.ub = "string"
        self.assertEqual(v.lb, 1.0)
        self.assertEqual(v.ub, 1.0)

    def test_extract_domain_type_and_bounds(self):
        # test an edge case
        domain_type, lb, ub = _extract_domain_type_and_bounds(None,
                                                              None,
                                                              None,
                                                              None)
        self.assertIs(domain_type, RealSet)
        self.assertIs(lb, None)
        self.assertIs(ub, None)

    def test_polynomial_degree(self):
        v = variable()
        self.assertEqual(v.polynomial_degree(), 1)
        v.fix(0)
        self.assertEqual(v.polynomial_degree(), 0)
        v.free()
        self.assertEqual(v.polynomial_degree(), 1)

    def test_ctype(self):
        v = variable()
        self.assertIs(v.ctype, IVariable)
        self.assertIs(type(v), variable)
        self.assertIs(type(v)._ctype, IVariable)

    def test_pickle(self):
        v = variable(lb=1,
                     ub=2,
                     domain_type=IntegerSet,
                     fixed=True)
        self.assertEqual(v.lb, 1)
        self.assertEqual(type(v.lb), int)
        self.assertEqual(v.ub, 2)
        self.assertEqual(type(v.ub), int)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.parent, None)
        vup = pickle.loads(
            pickle.dumps(v))
        self.assertEqual(vup.lb, 1)
        self.assertEqual(type(vup.lb), int)
        self.assertEqual(vup.ub, 2)
        self.assertEqual(type(vup.ub), int)
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
        self.assertEqual(v.ctype, IVariable)
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
        self.assertEqual(v.ctype, IVariable)
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub, 2)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1)
        self.assertEqual(v.stale, True)

    def test_type(self):
        v = variable()
        self.assertTrue(isinstance(v, ICategorizedObject))
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
        self.assertEqual(v.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)
        v.value = 1.0
        self.assertEqual(v.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, 1.0)
        v.fix()
        self.assertEqual(v.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, 1.0)
        v.value = None
        self.assertEqual(v.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(v), True)
        self.assertEqual(v.fixed, True)
        self.assertEqual(v.value, None)
        v.free()
        self.assertEqual(v.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(v), True)
        self.assertEqual(v.fixed, False)
        self.assertEqual(v.value, None)

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
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.has_ub(), False)

        v.lb = 0
        v.ub = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(type(v.lb), int)
        self.assertEqual(v.ub, 1)
        self.assertEqual(type(v.ub), int)
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
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.has_ub(), False)

        v.bounds = (float('-inf'), float('inf'))
        self.assertEqual(v.domain_type, RealSet)
        self.assertEqual(v.is_continuous(), True)
        self.assertEqual(v.is_discrete(), False)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), False)
        self.assertEqual(v.lb, float('-inf'))
        self.assertEqual(v.ub, float('inf'))
        self.assertEqual(v.bounds, (float('-inf'),float('inf')))
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.has_ub(), False)

        pL = parameter()
        v.lb = pL
        pU = parameter()
        v.ub = pU

        v.domain_type = IntegerSet
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        with self.assertRaises(ValueError):
            v.is_binary()
        self.assertEqual(v.is_integer(), True)
        self.assertIs(v.lb, pL)
        with self.assertRaises(ValueError):
            v.has_lb()
        self.assertIs(v.ub, pU)
        with self.assertRaises(ValueError):
            v.has_ub()

        pL.value = 0
        pU.value = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb(), 0)
        self.assertEqual(v.ub(), 1)
        self.assertEqual(v.has_lb(), True)
        self.assertEqual(v.has_ub(), True)

        pL.value = 0
        pU.value = 0
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb(), 0)
        self.assertEqual(v.ub(), 0)

        pL.value = 1
        pU.value = 1
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), True)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb(), 1)
        self.assertEqual(v.ub(), 1)

        v.domain = Binary
        pU.value = 2
        v.ub = pU
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.ub(), 2)

        pL.value = -1
        v.lb = pL
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb(), -1)
        self.assertEqual(v.ub(), 2)

        pL.value = float('-inf')
        pU.value = float('inf')
        self.assertEqual(v.domain_type, IntegerSet)
        self.assertEqual(v.is_continuous(), False)
        self.assertEqual(v.is_discrete(), True)
        self.assertEqual(v.is_binary(), False)
        self.assertEqual(v.is_integer(), True)
        self.assertEqual(v.lb(), float('-inf'))
        self.assertEqual(v.ub(), float('inf'))
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.has_ub(), False)

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

    def test_has_lb_ub(self):
        v = variable()
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.lb, None)
        self.assertEqual(v.has_ub(), False)
        self.assertEqual(v.ub, None)

        v.lb = float('-inf')
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.lb, float('-inf'))
        self.assertEqual(v.has_ub(), False)
        self.assertEqual(v.ub, None)

        v.ub = float('inf')
        self.assertEqual(v.has_lb(), False)
        self.assertEqual(v.lb, float('-inf'))
        self.assertEqual(v.has_ub(), False)
        self.assertEqual(v.ub, float('inf'))

        v.lb = 0
        self.assertEqual(v.has_lb(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.has_ub(), False)
        self.assertEqual(v.ub, float('inf'))

        v.ub = 0
        self.assertEqual(v.has_lb(), True)
        self.assertEqual(v.lb, 0)
        self.assertEqual(v.has_ub(), True)
        self.assertEqual(v.ub, 0)

        #
        # edge cases
        #

        v.lb = float('inf')
        self.assertEqual(v.has_lb(), True)
        self.assertEqual(v.lb, float('inf'))
        self.assertEqual(v.has_ub(), True)
        self.assertEqual(v.ub, 0)

        v.ub = float('-inf')
        self.assertEqual(v.has_lb(), True)
        self.assertEqual(v.lb, float('inf'))
        self.assertEqual(v.has_ub(), True)
        self.assertEqual(v.ub, float('-inf'))

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

    def test_active(self):
        v = variable()
        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.v = v
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(b.active, True)

    def test_call(self):
        v = variable()
        self.assertEqual(v.value, None)
        with self.assertRaises(ValueError):
            v()
        with self.assertRaises(ValueError):
            v(exception=True)
        self.assertEqual(v(exception=False), None)

        v.value = 2
        self.assertEqual(v.value, 2)
        self.assertEqual(v(), 2)
        self.assertEqual(v(exception=True), 2)
        self.assertEqual(v(exception=False), 2)

    def test_slack_methods(self):
        x = variable(value=2)
        L = 1
        U = 5

        # equality
        x.bounds = L,L
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, -3)
        self.assertEqual(x.lslack, 3)
        self.assertEqual(x.uslack, -3)
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, -5)
        self.assertEqual(x.lslack, 5)
        self.assertEqual(x.uslack, -5)
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, -1)
        self.assertEqual(x.uslack, 1)
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # equality
        x.bounds = U, U
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, -1)
        self.assertEqual(x.uslack, 1)
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, 1)
        self.assertEqual(x.uslack, -1)
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, -5)
        self.assertEqual(x.lslack, -5)
        self.assertEqual(x.uslack, 5)
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # lower finite
        x.bounds = L, None
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, 3)
        self.assertEqual(x.lslack, 3)
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, 5)
        self.assertEqual(x.lslack, 5)
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, -1)
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # lower unbounded
        x.bounds = float('-inf'), None
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # upper finite
        x.bounds = None, U
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, 1)
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, 1)
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, -1)
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, 5)
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, 5)
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # upper unbounded
        x.bounds = None, float('inf')
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # range finite
        x.bounds = L, U
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, 1)
        self.assertEqual(x.lslack, 3)
        self.assertEqual(x.uslack, 1)
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, 5)
        self.assertEqual(x.uslack, -1)
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, -1)
        self.assertEqual(x.uslack, 5)
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # range unbounded (None)
        x.bounds = None, None
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # range unbounded
        x.bounds = float('-inf'), float('inf')
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # range finite (parameter)
        x.bounds = parameter(L), parameter(U)
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, 1)
        self.assertEqual(x.lslack, 3)
        self.assertEqual(x.uslack, 1)
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, 5)
        self.assertEqual(x.uslack, -1)
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, -1)
        self.assertEqual(x.lslack, -1)
        self.assertEqual(x.uslack, 5)
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

        # range unbounded (parameter)
        x.bounds = parameter(float('-inf')), parameter(float('inf'))
        x.value = 4
        self.assertEqual(x.value, 4)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 6
        self.assertEqual(x.value, 6)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = 0
        self.assertEqual(x.value, 0)
        self.assertEqual(x.slack, float('inf'))
        self.assertEqual(x.lslack, float('inf'))
        self.assertEqual(x.uslack, float('inf'))
        x.value = None
        self.assertEqual(x.value, None)
        self.assertEqual(x.slack, None)
        self.assertEqual(x.lslack, None)
        self.assertEqual(x.uslack, None)

class _variable_subclass(variable):
    pass

class Test_variable_dict(_TestActiveDictContainerBase,
                         unittest.TestCase):
    _container_type = variable_dict
    _ctype_factory = lambda self: variable()

class Test_variable_tuple(_TestActiveTupleContainerBase,
                          unittest.TestCase):
    _container_type = variable_tuple
    _ctype_factory = lambda self: variable()

class Test_variable_list(_TestActiveListContainerBase,
                         unittest.TestCase):
    _container_type = variable_list
    _ctype_factory = lambda self: variable()

if __name__ == "__main__":
    unittest.main()
