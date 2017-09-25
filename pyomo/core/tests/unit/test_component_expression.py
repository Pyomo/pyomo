import pickle

import pyutilib.th as unittest
import pyomo.kernel
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_tuple import \
    _TestComponentTupleBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.kernel.component_interface import (ICategorizedObject,
                                                   IActiveObject,
                                                   IComponent,
                                                   IComponentContainer)
from pyomo.core.kernel.component_expression import (IIdentityExpression,
                                                    noclone,
                                                    IExpression,
                                                    expression,
                                                    data_expression,
                                                    expression_dict,
                                                    expression_tuple,
                                                    expression_list)
from pyomo.core.kernel.numvalue import (NumericValue,
                                        is_fixed,
                                        is_constant,
                                        potentially_variable,
                                        value)
from pyomo.core.kernel.component_variable import variable
from pyomo.core.kernel.component_parameter import parameter
from pyomo.core.kernel.component_objective import objective
from pyomo.core.kernel.component_block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)
from pyomo.core.base.expression import Expression

import six
from six import StringIO

try:
    import numpy
    numpy_available = True
except:
    numpy_available = False

class Test_noclone(unittest.TestCase):

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
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        e = noclone(v**2)
        pyomo.core.kernel.pprint(e)
        pyomo.core.kernel.pprint(e, indent=1)
        b = block()
        b.e = expression(expr=e)
        pyomo.core.kernel.pprint(e)
        pyomo.core.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.core.kernel.pprint(e)
        pyomo.core.kernel.pprint(b)
        pyomo.core.kernel.pprint(m)

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

    def test_potentially_variable(self):
        e = noclone(variable())
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
        e = noclone(parameter())
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
        e = noclone(expression())
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
        e = noclone(data_expression())
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)

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

    def test_is_expression(self):
        for obj in (variable(), parameter(), objective(),
                    expression(), data_expression()):
            self.assertEqual(noclone(obj).is_expression(), True)

    def test_args(self):
        e = noclone(parameter() + 1)
        self.assertEqual(len(e._args), 1)
        self.assertTrue(e._args[0] is e.expr)

    def test_aruments(self):
        e = noclone(parameter() + 1)
        self.assertEqual(len(tuple(e._arguments())), 1)
        self.assertTrue(tuple(e._arguments())[0] is e.expr)

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
        self.assertEqual(str(e.expr), "<parameter>**2.0")
        self.assertEqual(str(e), "{<parameter>**2.0}")
        e.to_string()
        out = StringIO()
        e.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "<parameter>**2.0")
        e.to_string(verbose=False)
        out = StringIO()
        e.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "<parameter>**2.0")
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         "{pow( <parameter> , 2.0 )}")
        b.e = e
        b.p = p
        self.assertNotEqual(p.name, None)
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         "{pow( "+p.name+" , 2.0 )}")
        self.assertEqual(out.getvalue(),
                         "{pow( p , 2.0 )}")
        del b.e
        del b.p


class _Test_expression_base(object):

    _ctype_factory = None

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        p = parameter()
        e = self._ctype_factory(p**2)
        pyomo.core.kernel.pprint(e)
        b = block()
        b.e = e
        pyomo.core.kernel.pprint(e)
        pyomo.core.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.core.kernel.pprint(e)
        pyomo.core.kernel.pprint(b)
        pyomo.core.kernel.pprint(m)

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
        self.assertEqual(e.ctype, Expression)
        self.assertTrue(e.expr is None)

    def test_init_args(self):
        e = self._ctype_factory(1.0)
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, Expression)
        self.assertTrue(e.expr is not None)

    def test_type(self):
        e = self._ctype_factory()
        self.assertTrue(isinstance(e, ICategorizedObject))
        self.assertTrue(isinstance(e, IComponent))
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

    def test_is_expression(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_expression(), True)

    def test_args(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(len(e._args), 1)
        self.assertTrue(e._args[0] is e.expr)

    def test_aruments(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(len(tuple(e._arguments())), 1)
        self.assertTrue(tuple(e._arguments())[0] is e.expr)

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
        e.to_string()
        out = StringIO()
        e.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "Undefined")
        e.to_string(verbose=False)
        out = StringIO()
        e.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "Undefined")
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), label+"{Undefined}")
        b.e = e
        self.assertNotEqual(e.name, None)
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "e{Undefined}")
        del b.e
        self.assertEqual(e.name, None)

        e.expr = 1
        self.assertEqual(str(e.expr), "1")
        self.assertEqual(str(e), label)
        e.to_string()
        out = StringIO()
        e.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "1.0")
        e.to_string(verbose=False)
        out = StringIO()
        e.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "1.0")
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), label+"{1.0}")
        b.e = e
        self.assertNotEqual(e.name, None)
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "e{1.0}")
        del b.e
        self.assertEqual(e.name, None)


        p = parameter()
        e.expr = p**2
        self.assertEqual(str(e.expr), "<parameter>**2.0")
        self.assertEqual(str(e), label)
        e.to_string()
        out = StringIO()
        e.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "<parameter>**2.0")
        e.to_string(verbose=False)
        out = StringIO()
        e.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "<parameter>**2.0")
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         label+"{pow( <parameter> , 2.0 )}")
        b.e = e
        b.p = p
        self.assertNotEqual(e.name, None)
        self.assertNotEqual(p.name, None)
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         e.name+"{pow( "+p.name+" , 2.0 )}")
        self.assertEqual(out.getvalue(),
                         "e{pow( p , 2.0 )}")
        del b.e
        del b.p

class Test_expression(_Test_expression_base,
                      unittest.TestCase):
    _ctype_factory = expression

    def test_ctype(self):
        e = expression()
        self.assertIs(e.ctype, Expression)
        self.assertIs(type(e).ctype, Expression)
        self.assertIs(expression.ctype, Expression)

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

    def test_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
        e.expr = 1
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
        v.fix()
        e.expr = v + 1
        self.assertEqual(e._potentially_variable(), True)
        self.assertEqual(potentially_variable(e), True)
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

    def test_ctype(self):
        e = data_expression()
        self.assertIs(e.ctype, Expression)
        self.assertIs(type(e).ctype, Expression)
        self.assertIs(data_expression.ctype, Expression)

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

    def test_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
        e.expr = 1
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
        p = parameter()
        e.expr = p**2
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
        a = self._ctype_factory()
        e.expr = (a*p)**2/(p + 5)
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e._potentially_variable(), False)
        self.assertEqual(potentially_variable(e), False)
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

class Test_expression_dict(_TestComponentDictBase,
                           unittest.TestCase):
    _container_type = expression_dict
    _ctype_factory = lambda self: expression()

class Test_expression_tuple(_TestComponentTupleBase,
                           unittest.TestCase):
    _container_type = expression_tuple
    _ctype_factory = lambda self: expression()

class Test_expression_list(_TestComponentListBase,
                           unittest.TestCase):
    _container_type = expression_list
    _ctype_factory = lambda self: expression()

if __name__ == "__main__":
    unittest.main()
