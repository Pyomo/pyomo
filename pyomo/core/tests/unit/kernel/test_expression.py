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
                                      is_potentially_variable,
                                      value)
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.expression import (IIdentityExpression,
                                          noclone,
                                          IExpression,
                                          expression,
                                          data_expression,
                                          expression_dict,
                                          expression_tuple,
                                          expression_list)
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.objective import objective
from pyomo.core.kernel.block import block

import six

try:
    import numpy
    numpy_available = True
except:
    numpy_available = False

class Test_noclone(unittest.TestCase):

    def test_is_named_expression_type(self):
        e = expression()
        self.assertEqual(e.is_named_expression_type(), True)

    def test_arg(self):
        e = expression()
        self.assertEqual(e.arg(0), None)
        e.expr = 1
        self.assertEqual(e.arg(0), 1)
        with self.assertRaises(KeyError):
            e.arg(1)

    def test_init_non_NumericValue(self):
        types = [None, 1, 1.1, True, ""]
        if numpy_available:
            types.extend([numpy.float32(1), numpy.bool_(True), numpy.int32(1)])
        types.append(block())
        types.append(block)
        for obj in types:
            self.assertEqual(noclone(obj), obj)
            self.assertIs(type(noclone(obj)), type(obj))

    def test_init_NumericValue(self):
        v = variable()
        p = parameter()
        e = expression()
        d = data_expression()
        o = objective()
        for obj in (v, v+1, v**2,
                    p, p+1, p**2,
                    e, e+1, e**2,
                    d, d+1, d**2,
                    o, o+1, o**2):
            self.assertTrue(isinstance(noclone(obj), NumericValue))
            self.assertTrue(isinstance(noclone(obj), IIdentityExpression))
            self.assertTrue(isinstance(noclone(obj), noclone))
            self.assertIs(noclone(obj).expr, obj)

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        e = noclone(v**2)
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(e, indent=1)
        b = block()
        b.e = expression(expr=e)
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)
        # tests compatibility with _ToStringVisitor
        pyomo.kernel.pprint(noclone(v)+1)
        pyomo.kernel.pprint(noclone(v+1))
        x = variable()
        y = variable()
        pyomo.kernel.pprint(y + x*noclone(noclone(x*y)))
        pyomo.kernel.pprint(y + noclone(noclone(x*y))*x)

    def test_pickle(self):
        v = variable()
        e = noclone(v)
        self.assertEqual(type(e), noclone)
        self.assertIs(type(e.expr), variable)
        eup = pickle.loads(
            pickle.dumps(e))
        self.assertEqual(type(eup), noclone)
        self.assertTrue(e is not eup)
        self.assertIs(type(eup.expr), variable)
        self.assertIs(type(e.expr), variable)
        self.assertTrue(eup.expr is not e.expr)

        del e
        del v

        v = variable(value=1)
        b = block()
        b.v = v
        eraw = b.v + 1
        b.e = 1 + noclone(eraw)
        bup = pickle.loads(
            pickle.dumps(b))
        self.assertTrue(isinstance(bup.e, NumericValue))
        self.assertEqual(value(bup.e), 3.0)
        b.v.value = 2
        self.assertEqual(value(b.e), 4.0)
        self.assertEqual(value(bup.e), 3.0)
        bup.v.value = -1
        self.assertEqual(value(b.e), 4.0)
        self.assertEqual(value(bup.e), 1.0)

        self.assertIs(b.v.parent, b)
        self.assertIs(bup.v.parent, bup)

        del b.v

    def test_call(self):
        e = noclone(None)
        self.assertIs(e, None)
        e = noclone(1)
        self.assertEqual(e, 1)
        p = parameter()
        p.value = 2
        e = noclone(p + 1)
        self.assertEqual(e(), 3)

    def test_is_constant(self):
        v = variable()
        e = noclone(v)
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        v.fix(1)
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)

        p = parameter()
        e = noclone(p)
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)

        self.assertEqual(is_constant(noclone(1)), True)

    def test_is_fixed(self):
        v = variable()
        e = noclone(v + 1)
        self.assertEqual(e.is_fixed(), False)
        self.assertEqual(is_fixed(e), False)
        v.fix()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)

        e = noclone(parameter())
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)

    def testis_potentially_variable(self):
        e = noclone(variable())
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = noclone(parameter())
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        e = noclone(expression())
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = noclone(data_expression())
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)

    def test_polynomial_degree(self):
        e = noclone(parameter())
        self.assertEqual(e.polynomial_degree(), 0)
        e = noclone(parameter(value=1))
        self.assertEqual(e.polynomial_degree(), 0)
        v = variable()
        v.value = 2
        e = noclone(v + 1)
        self.assertEqual(e.polynomial_degree(), 1)
        e = noclone(v**2 + v + 1)
        self.assertEqual(e.polynomial_degree(), 2)
        v.fix()
        self.assertEqual(e.polynomial_degree(), 0)
        e = noclone(v**v)
        self.assertEqual(e.polynomial_degree(), 0)
        v.free()
        self.assertEqual(e.polynomial_degree(), None)

    def test_is_expression_type(self):
        for obj in (variable(), parameter(), objective(),
                    expression(), data_expression()):
            self.assertEqual(noclone(obj).is_expression_type(), True)

    def test_is_parameter_type(self):
        for obj in (variable(), parameter(), objective(),
                    expression(), data_expression()):
            self.assertEqual(noclone(obj).is_parameter_type(), False)

    def test_args(self):
        e = noclone(parameter() + 1)
        self.assertEqual(e.nargs(), 1)
        self.assertTrue(e.arg(0) is e.expr)

    def test_aruments(self):
        e = noclone(parameter() + 1)
        self.assertEqual(len(tuple(e.args)), 1)
        self.assertTrue(tuple(e.args)[0] is e.expr)

    def test_clone(self):

        p = parameter()
        e = noclone(p)
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is p)
        sube = p**2 + 1
        e = noclone(sube)
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is sube)

    def test_division_behavior(self):
        # make sure integers involved in Pyomo expression
        # use __future__ behavior
        e = noclone(parameter(value=2))
        self.assertIs(type(e.expr), parameter)
        self.assertEqual((1/e)(), 0.5)
        self.assertEqual((parameter(1)/e)(), 0.5)
        # since the type returned is int, this should result
        # in the behavior used by the interpreter
        if six.PY3:
            self.assertEqual((1/e.expr()), 0.5)
        else:
            self.assertEqual((1/e.expr()), 0)

    def test_to_string(self):
        b = block()
        p = parameter()
        e = noclone(p**2)
        self.assertEqual(str(e.expr), "<parameter>**2")
        self.assertEqual(str(e), "{(<parameter>**2)}")
        self.assertEqual(e.to_string(), "(<parameter>**2)")
        self.assertEqual(e.to_string(verbose=False), "(<parameter>**2)")
        self.assertEqual(e.to_string(verbose=True), "{pow(<parameter>, 2)}")
        b.e = e
        b.p = p
        self.assertNotEqual(p.name, None)
        self.assertEqual(e.to_string(verbose=True), "{pow("+p.name+", 2)}")
        self.assertEqual(e.to_string(verbose=True), "{pow(p, 2)}")
        del b.e
        del b.p

class _Test_expression_base(object):

    _ctype_factory = None

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        p = parameter()
        e = self._ctype_factory(p**2)
        pyomo.kernel.pprint(e)
        b = block()
        b.e = e
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_pickle(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(type(e.expr), float)
        self.assertEqual(e.expr, 1.0)
        self.assertEqual(e.parent, None)
        eup = pickle.loads(
            pickle.dumps(e))
        self.assertEqual(type(eup.expr), float)
        self.assertEqual(eup.expr, 1.0)
        self.assertEqual(eup.parent, None)
        b = block()
        b.e = e
        self.assertIs(e.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        eup = bup.e
        self.assertEqual(type(eup.expr), float)
        self.assertEqual(eup.expr, 1.0)
        self.assertIs(eup.parent, bup)

    def test_init_no_args(self):
        e = self._ctype_factory()
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, IExpression)
        self.assertTrue(e.expr is None)

    def test_init_args(self):
        e = self._ctype_factory(1.0)
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, IExpression)
        self.assertTrue(e.expr is not None)

    def test_type(self):
        e = self._ctype_factory()
        self.assertTrue(isinstance(e, ICategorizedObject))
        self.assertTrue(isinstance(e, IExpression))
        self.assertTrue(isinstance(e, NumericValue))
        self.assertTrue(isinstance(e, IIdentityExpression))

    def test_call(self):
        e = self._ctype_factory()
        self.assertEqual(e(), None)
        e.expr = 1
        self.assertEqual(e(), 1)
        p = parameter()
        p.value = 2
        e.expr = p + 1
        self.assertEqual(e(), 3)

    def test_is_constant(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        e.expr = 1
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        p = parameter()
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)
        p.value = 2
        e.expr = p + 1
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)

    def test_is_expression_type(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_expression_type(), True)

    def test_is_parameter_type(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_parameter_type(), False)

    def test_args(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(e.nargs(), 1)
        self.assertTrue(e.arg(0) is e.expr)

    def test_aruments(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(len(tuple(e.args)), 1)
        self.assertTrue(tuple(e.args)[0] is e.expr)

    def test_clone(self):
        e = self._ctype_factory()
        self.assertTrue(e.clone() is e)
        p = parameter()
        e.expr = p
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is p)
        sube = p**2 + 1
        e.expr = sube
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is sube)

    def test_division_behavior(self):
        # make sure integers involved in Pyomo expression
        # use __future__ behavior
        e = self._ctype_factory()
        e.expr = 2
        self.assertIs(type(e.expr), int)
        self.assertEqual((1/e)(), 0.5)
        self.assertEqual((parameter(1)/e)(), 0.5)
        # since the type returned is int, this should result
        # in the behavior used by the interpreter
        if six.PY3:
            self.assertEqual((1/e.expr), 0.5)
        else:
            self.assertEqual((1/e.expr), 0)

    def test_to_string(self):
        b = block()
        e = self._ctype_factory()
        label = str(e)
        self.assertNotEqual(label, None)
        self.assertEqual(e.name, None)

        self.assertEqual(str(e.expr), "None")
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), label+"{Undefined}")
        self.assertEqual(e.to_string(verbose=False), label+"{Undefined}")
        self.assertEqual(e.to_string(verbose=True), label+"{Undefined}")
        b.e = e
        self.assertNotEqual(e.name, None)
        self.assertEqual(e.to_string(verbose=True), "e{Undefined}")
        del b.e
        self.assertEqual(e.name, None)

        e.expr = 1
        self.assertEqual(str(e.expr), "1")
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), "1")
        self.assertEqual(e.to_string(verbose=False), "1")
        self.assertEqual(e.to_string(verbose=True), label+"{1}")
        b.e = e
        self.assertNotEqual(e.name, None)
        self.assertEqual(e.to_string(verbose=True), "e{1}")
        del b.e
        self.assertEqual(e.name, None)


        p = parameter()
        e.expr = p**2
        self.assertEqual(str(e.expr), "<parameter>**2")
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), "(<parameter>**2)")
        self.assertEqual(e.to_string(verbose=False), "(<parameter>**2)")
        self.assertEqual(e.to_string(verbose=True), label+"{pow(<parameter>, 2)}")
        b.e = e
        b.p = p
        self.assertNotEqual(e.name, None)
        self.assertNotEqual(p.name, None)
        self.assertEqual(e.to_string(verbose=True), e.name+"{pow("+p.name+", 2)}")
        self.assertEqual(e.to_string(verbose=True), "e{pow(p, 2)}")
        del b.e
        del b.p

    def test_iadd(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        e = self._ctype_factory(1.0)
        expr = 0.0
        for v in [1.0,e]:
            expr += v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), 2)
        expr = 0.0
        for v in [e,1.0]:
            expr += v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), 2)

    def test_isub(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        e = self._ctype_factory(1.0)
        expr = 0.0
        for v in [1.0,e]:
            expr -= v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), -2)
        expr = 0.0
        for v in [e,1.0]:
            expr -= v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), -2)

    def test_imul(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        e = self._ctype_factory(3.0)
        expr = 1.0
        for v in [2.0,e]:
            expr *= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 6)
        expr = 1.0
        for v in [e,2.0]:
            expr *= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 6)

    def test_idiv(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        # floating point division
        e = self._ctype_factory(3.0)
        expr = e
        for v in [2.0,1.0]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        expr = e
        for v in [1.0,2.0]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        # note that integer division does not occur within
        # Pyomo expressions
        e = self._ctype_factory(3)
        expr = e
        for v in [2,1]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        expr = e
        for v in [1,2]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)

    def test_ipow(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        e = self._ctype_factory(3.0)
        expr = e
        for v in [2.0,1.0]:
            expr **= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 9)
        expr = e
        for v in [1.0,2.0]:
            expr **= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 9)

class Test_expression(_Test_expression_base,
                      unittest.TestCase):
    _ctype_factory = expression

    def test_associativity(self):
        x = variable()
        y = variable()
        pyomo.kernel.pprint(y + x*expression(expression(x*y)))
        pyomo.kernel.pprint(y + expression(expression(x*y))*x)

    def test_ctype(self):
        e = expression()
        self.assertIs(e.ctype, IExpression)
        self.assertIs(type(e), expression)
        self.assertIs(type(e)._ctype, IExpression)

    def test_is_fixed(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        e.expr = 1
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_fixed(), False)
        self.assertEqual(is_fixed(e), False)
        v.fix()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(e(), 3)

    def testis_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e.expr = 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        v.fix()
        e.expr = v + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        self.assertEqual(e(), 3)

    def test_polynomial_degree(self):
        e = self._ctype_factory()
        e.expr = 1
        self.assertEqual(e.polynomial_degree(), 0)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.polynomial_degree(), 1)
        e.expr = v**2 + v + 1
        self.assertEqual(e.polynomial_degree(), 2)
        v.fix()
        self.assertEqual(e.polynomial_degree(), 0)
        e.expr = v**v
        self.assertEqual(e.polynomial_degree(), 0)
        v.free()
        self.assertEqual(e.polynomial_degree(), None)

class Test_data_expression(_Test_expression_base,
                           unittest.TestCase):

    _ctype_factory = data_expression

    def test_associativity(self):
        x = parameter()
        y = parameter()
        pyomo.kernel.pprint(
            y + x*data_expression(data_expression(x*y)))
        pyomo.kernel.pprint(
            y + data_expression(data_expression(x*y))*x)

    def test_ctype(self):
        e = data_expression()
        self.assertIs(e.ctype, IExpression)
        self.assertIs(type(e), data_expression)
        self.assertIs(type(e)._ctype, IExpression)

    def test_bad_init(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(e.expr, 1.0)

        v = variable()
        with self.assertRaises(ValueError):
            e = self._ctype_factory(expr=v)

    def test_bad_assignment(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(e.expr, 1.0)

        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def test_is_fixed(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        e.expr = 1
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        p = parameter()
        e.expr = p**2
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        a = self._ctype_factory()
        e.expr = (a*p)**2/(p + 5)
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(e(), 10.0)

        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def testis_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        e.expr = 1
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        p = parameter()
        e.expr = p**2
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        a = self._ctype_factory()
        e.expr = (a*p)**2/(p + 5)
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        self.assertEqual(e(), 10.0)

        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def test_polynomial_degree(self):
        e = self._ctype_factory()
        self.assertEqual(e.polynomial_degree(), 0)
        e.expr = 1
        self.assertEqual(e.polynomial_degree(), 0)
        p = parameter()
        e.expr = p**2
        self.assertEqual(e.polynomial_degree(), 0)
        a = self._ctype_factory()
        e.expr = (a*p)**2/(p + 5)
        self.assertEqual(e.polynomial_degree(), 0)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(e(), 10.0)

        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

class Test_expression_dict(_TestActiveDictContainerBase,
                           unittest.TestCase):
    _container_type = expression_dict
    _ctype_factory = lambda self: expression()

class Test_expression_tuple(_TestActiveTupleContainerBase,
                           unittest.TestCase):
    _container_type = expression_tuple
    _ctype_factory = lambda self: expression()

class Test_expression_list(_TestActiveListContainerBase,
                           unittest.TestCase):
    _container_type = expression_list
    _ctype_factory = lambda self: expression()

if __name__ == "__main__":
    unittest.main()
