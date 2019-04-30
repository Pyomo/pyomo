import pickle
import math

import pyutilib.th as unittest
import pyomo.kernel
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (IConstraint,
                                          constraint_dict,
                                          constraint_tuple,
                                          constraint_list)
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import (expression,
                                          data_expression)
from pyomo.core.kernel.conic import (quadratic,
                                     rotated_quadratic,
                                     primal_exponential,
                                     primal_power,
                                     dual_exponential,
                                     dual_power)
from pyomo.kernel import IntegerSet

class _conic_tester_base(object):

    _object_factory = None

    def setUp(self):
        assert self._object_factory is not None

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        c = self._object_factory()
        pyomo.kernel.pprint(c)
        b = block()
        b.c = c
        pyomo.kernel.pprint(c)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(c)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_type(self):
        c = self._object_factory()
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IConstraint))

    def test_ctype(self):
        c = self._object_factory()
        self.assertIs(c.ctype, IConstraint)
        self.assertIs(type(c)._ctype, IConstraint)

    def test_pickle(self):
        c = self._object_factory()
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, 0)
        self.assertIsNot(c.body, None)
        self.assertIs(c.parent, None)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertIs(cup.lb, None)
        self.assertEqual(cup.ub, 0)
        self.assertIsNot(cup.body, None)
        self.assertIs(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        cup = bup.c
        self.assertIs(cup.lb, None)
        self.assertEqual(cup.ub, 0)
        self.assertIsNot(cup.body, None)
        self.assertIs(cup.parent, bup)

    def test_properties(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        self.assertIs(c.parent, None)
        self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.check_convexity_conditions(),
                         True)
        self.assertEqual(c.check_convexity_conditions(relax=False),
                         True)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        with self.assertRaises(AttributeError):
            c.lb = 1
        with self.assertRaises(AttributeError):
            c.ub = 1
        with self.assertRaises(AttributeError):
            c.rhs = 1
        with self.assertRaises(ValueError):
            c.rhs
        with self.assertRaises(AttributeError):
            c.equality = True

        self.assertIs(c._body, None)
        self.assertIs(c.parent, None)
        self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.check_convexity_conditions(),
                         True)
        self.assertEqual(c.check_convexity_conditions(relax=False),
                         True)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        self.assertEqual(c.active, True)
        with self.assertRaises(AttributeError):
            c.active = False
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        c.activate()
        self.assertEqual(c.active, True)

    def test_containers(self):
        c = self._object_factory()
        self.assertIs(c.parent, None)
        cdict = constraint_dict()
        cdict[None] = c
        self.assertIs(c.parent, cdict)
        del cdict[None]
        self.assertIs(c.parent, None)
        clist = constraint_list()
        clist.append(c)
        self.assertIs(c.parent, clist)
        clist.remove(c)
        self.assertIs(c.parent, None)
        ctuple = constraint_tuple((c,))
        self.assertIs(c.parent, ctuple)

class Test_quadratic(_conic_tester_base,
                     unittest.TestCase):

    _object_factory = lambda self: quadratic(
        x=[variable(),
           variable()],
        r=variable(lb=0))

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x[0].value = 5
        c.x[1].value = 2
        c.r.value = 3
        val = 5**2 + 2**2 - 3**2
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(c.body(), val)
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

class Test_rotated_quadratic(_conic_tester_base,
                             unittest.TestCase):

    _object_factory = lambda self: rotated_quadratic(
        x=[variable(),
           variable()],
        r1=variable(lb=0),
        r2=variable(lb=0))

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x[0].value = 2
        c.x[1].value = 3
        c.r1.value = 5
        c.r2.value = 7
        val = 2**2 + 3**2 - 2*5*7
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(c.body(), val)
        self.assertEqual(c(), val)
        self.assertEqual(c.slack, -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.r1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r1.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r1.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.r2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r2.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r2.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

class Test_primal_exponential(_conic_tester_base,
                              unittest.TestCase):

    _object_factory = lambda self: primal_exponential(
        x1=variable(lb=0),
        x2=variable(),
        r=variable(lb=0))

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x1.value = 1.1
        c.x2.value = 2.3
        c.r.value = 8
        val = round(1.1*math.exp(2.3/1.1) - 8, 9)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(round(c.body(),9), val)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.x1.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.x1.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.x2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.r.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

class Test_primal_power(_conic_tester_base,
                        unittest.TestCase):

    _object_factory = lambda self: primal_power(
        x=[variable(),
           variable()],
        r1=variable(lb=0),
        r2=variable(lb=0),
        alpha=parameter(value=0.4))

    def test_bad_alpha_type(self):
        c = primal_power(
            x=[variable(),
               variable()],
            r1=variable(lb=0),
            r2=variable(lb=0),
            alpha=parameter())
        c = primal_power(
            x=[variable(),
               variable()],
            r1=variable(lb=0),
            r2=variable(lb=0),
            alpha=data_expression())
        with self.assertRaises(TypeError):
            c = primal_power(
                x=[variable(),
                   variable()],
                r1=variable(lb=0),
                r2=variable(lb=0),
                alpha=variable())
        with self.assertRaises(TypeError):
            c = primal_power(
                x=[variable(),
                   variable()],
                r1=variable(lb=0),
                r2=variable(lb=0),
                alpha=expression())

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x[0].value = 1.1
        c.x[1].value = -2.3
        c.r1.value = 5.9
        c.r2.value = 3.4
        val = round((1.1**2 + (-2.3)**2)**0.5 - \
                    (5.9**0.4)*(3.4**0.6), 9)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(round(c.body(),9), val)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.r1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r1.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r1.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.r2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r2.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r2.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.alpha.value = 0
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.alpha.value = 1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

class Test_dual_exponential(_conic_tester_base,
                            unittest.TestCase):

    _object_factory = lambda self: dual_exponential(
        x1=variable(),
        x2=variable(ub=0),
        r=variable(lb=0))

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x1.value = 1.2
        c.x2.value = -5.3
        c.r.value = 2.7
        val = round(-(-5.3/math.e)*math.exp(1.2/-5.3) - 2.7, 9)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(round(c.body(),9), val)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.x2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.x2.ub = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.x2.ub = 1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.r.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

class Test_dual_power(_conic_tester_base,
                      unittest.TestCase):

    _object_factory = lambda self: dual_power(
        x=[variable(),
           variable()],
        r1=variable(lb=0),
        r2=variable(lb=0),
        alpha=parameter(value=0.4))

    def test_bad_alpha_type(self):
        c = dual_power(
            x=[variable(),
               variable()],
            r1=variable(lb=0),
            r2=variable(lb=0),
            alpha=parameter())
        c = dual_power(
            x=[variable(),
               variable()],
            r1=variable(lb=0),
            r2=variable(lb=0),
            alpha=data_expression())
        with self.assertRaises(TypeError):
            c = dual_power(
                x=[variable(),
                   variable()],
                r1=variable(lb=0),
                r2=variable(lb=0),
                alpha=variable())
        with self.assertRaises(TypeError):
            c = dual_power(
                x=[variable(),
                   variable()],
                r1=variable(lb=0),
                r2=variable(lb=0),
                alpha=expression())

    def test_expression(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        with self.assertRaises(ValueError):
            self.assertIs(c(), None)
        with self.assertRaises(ValueError):
            self.assertIs(c(exception=True), None)
        self.assertIs(c(exception=False), None)
        self.assertIs(c._body, None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

        c.x[0].value = 1.2
        c.x[1].value = -5.3
        c.r1.value = 2.7
        c.r2.value = 3.7
        val = round((1.2**2 + (-5.3)**2)**0.5 - \
                    ((2.7/0.4)**0.4) * \
                    ((3.7/0.6)**0.6), 9)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIs(c._body, None)
        # check body
        self.assertEqual(round(c.body(),9), val)
        self.assertEqual(round(c(),9), val)
        self.assertEqual(round(c.slack,9), -val)
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(round(c.uslack,9), -val)
        self.assertIsNot(c._body, None)

    def test_check_convexity_conditions(self):
        c = self._object_factory()
        self.assertEqual(c.check_convexity_conditions(),
                         True)

        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.r1.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r1.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r1.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.r2.domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)
        c = self._object_factory()
        c.r2.lb = None
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.r2.lb = -1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

        c = self._object_factory()
        c.alpha.value = 0
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.alpha.value = 1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

if __name__ == "__main__":
    unittest.main()
