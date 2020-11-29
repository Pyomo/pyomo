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
import math

import pyutilib.th as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (IConstraint,
                                          linear_constraint,
                                          constraint,
                                          constraint_dict,
                                          constraint_tuple,
                                          constraint_list)
from pyomo.core.kernel.variable import (variable,
                                        variable_tuple)
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import (expression,
                                          data_expression)
from pyomo.core.kernel.conic import (_build_linking_constraints,
                                     quadratic,
                                     rotated_quadratic,
                                     primal_exponential,
                                     primal_power,
                                     dual_exponential,
                                     dual_power)

class _conic_tester_base(object):

    _object_factory = None

    def setUp(self):
        assert self._object_factory is not None

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        c = self._object_factory()
        pprint(c)
        b = block()
        b.c = c
        pprint(c)
        pprint(b)
        m = block()
        m.b = b
        pprint(c)
        pprint(b)
        pprint(m)

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
        r=variable(lb=0),
        x=[variable(),
           variable()])

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

        c.r.value = 3
        c.x[0].value = 5
        c.x[1].value = 2
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

        c = self._object_factory()
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

    def test_as_domain(self):
        ret = quadratic.as_domain(
            r=3,x=[1,2])
        self.assertIs(type(ret), block)
        q,c,r,x = ret.q,ret.c,ret.r,ret.x
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), quadratic)
        self.assertIs(type(x), variable_tuple)
        self.assertIs(type(r), variable)
        self.assertEqual(len(x), 2)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0].rhs, 3)
        r.value = 3
        self.assertEqual(c[0].slack, 0)
        r.value = None
        self.assertEqual(c[1].rhs, 1)
        x[0].value = 1
        self.assertEqual(c[1].slack, 0)
        x[0].value = None
        self.assertEqual(c[2].rhs, 2)
        x[1].value = 2
        self.assertEqual(c[2].slack, 0)
        x[1].value = None

class Test_rotated_quadratic(_conic_tester_base,
                             unittest.TestCase):

    _object_factory = lambda self: rotated_quadratic(
        r1=variable(lb=0),
        r2=variable(lb=0),
        x=[variable(),
           variable()])

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

        c.r1.value = 5
        c.r2.value = 7
        c.x[0].value = 2
        c.x[1].value = 3
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
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

    def test_as_domain(self):
        ret = rotated_quadratic.as_domain(
            r1=3,r2=4,x=[1,2])
        self.assertIs(type(ret), block)
        q,c,r1,r2,x = ret.q,ret.c,ret.r1,ret.r2,ret.x
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), rotated_quadratic)
        self.assertIs(type(x), variable_tuple)
        self.assertIs(type(r1), variable)
        self.assertIs(type(r2), variable)
        self.assertEqual(len(x), 2)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 4)
        self.assertEqual(c[0].rhs, 3)
        r1.value = 3
        self.assertEqual(c[0].slack, 0)
        r1.value = None
        self.assertEqual(c[1].rhs, 4)
        r2.value = 4
        self.assertEqual(c[1].slack, 0)
        r2.value = None
        self.assertEqual(c[2].rhs, 1)
        x[0].value = 1
        self.assertEqual(c[2].slack, 0)
        x[0].value = None
        self.assertEqual(c[3].rhs, 2)
        x[1].value = 2
        self.assertEqual(c[3].slack, 0)
        x[1].value = None

class Test_primal_exponential(_conic_tester_base,
                              unittest.TestCase):

    _object_factory = lambda self: primal_exponential(
        r=variable(lb=0),
        x1=variable(lb=0),
        x2=variable())

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

        c.r.value = 8
        c.x1.value = 1.1
        c.x2.value = 2.3
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

    def test_as_domain(self):
        ret = primal_exponential.as_domain(
            r=3,x1=1,x2=2)
        self.assertIs(type(ret), block)
        q,c,r,x1,x2 = ret.q,ret.c,ret.r,ret.x1,ret.x2
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), primal_exponential)
        self.assertIs(type(r), variable)
        self.assertIs(type(x1), variable)
        self.assertIs(type(x2), variable)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0].rhs, 3)
        r.value = 3
        self.assertEqual(c[0].slack, 0)
        r.value = None
        self.assertEqual(c[1].rhs, 1)
        x1.value = 1
        self.assertEqual(c[1].slack, 0)
        x1.value = None
        self.assertEqual(c[2].rhs, 2)
        x2.value = 2
        self.assertEqual(c[2].slack, 0)
        x2.value = None

class Test_primal_power(_conic_tester_base,
                        unittest.TestCase):

    _object_factory = lambda self: primal_power(
        r1=variable(lb=0),
        r2=variable(lb=0),
        x=[variable(),
           variable()],
        alpha=parameter(value=0.4))

    def test_bad_alpha_type(self):
        c = primal_power(
            r1=variable(lb=0),
            r2=variable(lb=0),
            x=[variable(),
               variable()],
            alpha=parameter())
        c = primal_power(
            r1=variable(lb=0),
            r2=variable(lb=0),
            x=[variable(),
               variable()],
            alpha=data_expression())
        with self.assertRaises(TypeError):
            c = primal_power(
                r1=variable(lb=0),
                r2=variable(lb=0),
                x=[variable(),
                   variable()],
                alpha=variable())
        with self.assertRaises(TypeError):
            c = primal_power(
                r1=variable(lb=0),
                r2=variable(lb=0),
                x=[variable(),
                   variable()],
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

        c.r1.value = 5.9
        c.r2.value = 3.4
        c.x[0].value = 1.1
        c.x[1].value = -2.3
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
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.alpha.value = 0
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.alpha.value = 1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

    def test_as_domain(self):
        ret = primal_power.as_domain(
            r1=3,r2=4,x=[1,2],alpha=0.5)
        self.assertIs(type(ret), block)
        q,c,r1,r2,x = ret.q,ret.c,ret.r1,ret.r2,ret.x
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), primal_power)
        self.assertIs(type(r1), variable)
        self.assertIs(type(r2), variable)
        self.assertIs(type(x), variable_tuple)
        self.assertEqual(len(x), 2)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 4)
        self.assertEqual(c[0].rhs, 3)
        r1.value = 3
        self.assertEqual(c[0].slack, 0)
        r1.value = None
        self.assertEqual(c[1].rhs, 4)
        r2.value = 4
        self.assertEqual(c[1].slack, 0)
        r2.value = None
        self.assertEqual(c[2].rhs, 1)
        x[0].value = 1
        self.assertEqual(c[2].slack, 0)
        x[0].value = None
        self.assertEqual(c[3].rhs, 2)
        x[1].value = 2
        self.assertEqual(c[3].slack, 0)
        x[1].value = None

class Test_dual_exponential(_conic_tester_base,
                            unittest.TestCase):

    _object_factory = lambda self: dual_exponential(
        r=variable(lb=0),
        x1=variable(),
        x2=variable(ub=0))

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

        c.r.value = 2.7
        c.x1.value = 1.2
        c.x2.value = -5.3
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

    def test_as_domain(self):
        ret = dual_exponential.as_domain(
            r=3,x1=1,x2=2)
        self.assertIs(type(ret), block)
        q,c,r,x1,x2 = ret.q,ret.c,ret.r,ret.x1,ret.x2
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), dual_exponential)
        self.assertIs(type(x1), variable)
        self.assertIs(type(x2), variable)
        self.assertIs(type(r), variable)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0].rhs, 3)
        r.value = 3
        self.assertEqual(c[0].slack, 0)
        r.value = None
        self.assertEqual(c[1].rhs, 1)
        x1.value = 1
        self.assertEqual(c[1].slack, 0)
        x1.value = None
        self.assertEqual(c[2].rhs, 2)
        x2.value = 2
        self.assertEqual(c[2].slack, 0)
        x2.value = None

class Test_dual_power(_conic_tester_base,
                      unittest.TestCase):

    _object_factory = lambda self: dual_power(
        r1=variable(lb=0),
        r2=variable(lb=0),
        x=[variable(),
           variable()],
        alpha=parameter(value=0.4))

    def test_bad_alpha_type(self):
        c = dual_power(
            r1=variable(lb=0),
            r2=variable(lb=0),
            x=[variable(),
               variable()],
            alpha=parameter())
        c = dual_power(
            r1=variable(lb=0),
            r2=variable(lb=0),
            x=[variable(),
               variable()],
            alpha=data_expression())
        with self.assertRaises(TypeError):
            c = dual_power(
                r1=variable(lb=0),
                r2=variable(lb=0),
                x=[variable(),
                   variable()],
                alpha=variable())
        with self.assertRaises(TypeError):
            c = dual_power(
                r1=variable(lb=0),
                r2=variable(lb=0),
                x=[variable(),
                   variable()],
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

        c.r1.value = 2.7
        c.r2.value = 3.7
        c.x[0].value = 1.2
        c.x[1].value = -5.3
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
        c.x[0].domain_type = IntegerSet
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        self.assertEqual(c.check_convexity_conditions(relax=True),
                         True)

        c = self._object_factory()
        c.alpha.value = 0
        self.assertEqual(c.check_convexity_conditions(),
                         False)
        c = self._object_factory()
        c.alpha.value = 1
        self.assertEqual(c.check_convexity_conditions(),
                         False)

    def test_as_domain(self):
        ret = dual_power.as_domain(
            r1=3,r2=4,x=[1,2],alpha=0.5)
        self.assertIs(type(ret), block)
        q,c,r1,r2,x = ret.q,ret.c,ret.r1,ret.r2,ret.x
        self.assertEqual(q.check_convexity_conditions(), True)
        self.assertIs(type(q), dual_power)
        self.assertIs(type(r1), variable)
        self.assertIs(type(r2), variable)
        self.assertIs(type(x), variable_tuple)
        self.assertEqual(len(x), 2)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 4)
        self.assertEqual(c[0].rhs, 3)
        r1.value = 3
        self.assertEqual(c[0].slack, 0)
        r1.value = None
        self.assertEqual(c[1].rhs, 4)
        r2.value = 4
        self.assertEqual(c[1].slack, 0)
        r2.value = None
        self.assertEqual(c[2].rhs, 1)
        x[0].value = 1
        self.assertEqual(c[2].slack, 0)
        x[0].value = None
        self.assertEqual(c[3].rhs, 2)
        x[1].value = 2
        self.assertEqual(c[3].slack, 0)
        x[1].value = None

class TestMisc(unittest.TestCase):

    def test_build_linking_constraints(self):
        c = _build_linking_constraints([],[])
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 0)
        c = _build_linking_constraints([None],[variable()])
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 0)
        v = [1,
             data_expression(),
             variable(),
             expression(expr=1.0)]
        vaux = [variable(),
                variable(),
                variable(),
                variable()]
        c = _build_linking_constraints(v, vaux)
        self.assertIs(type(c), constraint_tuple)
        self.assertEqual(len(c), 4)
        self.assertIs(type(c[0]), linear_constraint)
        self.assertEqual(c[0].rhs, 1)
        self.assertEqual(len(list(c[0].terms)), 1)
        self.assertIs(list(c[0].terms)[0][0], vaux[0])
        self.assertEqual(list(c[0].terms)[0][1], 1)
        self.assertIs(type(c[1]), linear_constraint)
        self.assertIs(c[1].rhs, v[1])
        self.assertEqual(len(list(c[1].terms)), 1)
        self.assertIs(list(c[1].terms)[0][0], vaux[1])
        self.assertEqual(list(c[1].terms)[0][1], 1)
        self.assertIs(type(c[2]), linear_constraint)
        self.assertEqual(c[2].rhs, 0)
        self.assertEqual(len(list(c[2].terms)), 2)
        self.assertIs(list(c[2].terms)[0][0], vaux[2])
        self.assertEqual(list(c[2].terms)[0][1], 1)
        self.assertIs(list(c[2].terms)[1][0], v[2])
        self.assertEqual(list(c[2].terms)[1][1], -1)
        self.assertIs(type(c[3]), constraint)
        self.assertEqual(c[3].rhs, 0)
        from pyomo.repn import generate_standard_repn
        repn = generate_standard_repn(c[3].body)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], vaux[3])
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.constant, -1)

if __name__ == "__main__":
    unittest.main()
