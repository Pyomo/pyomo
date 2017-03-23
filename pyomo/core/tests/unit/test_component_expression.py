import pickle

import pyutilib.th as unittest
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
from pyomo.core.kernel.component_expression import (IExpression,
                                                    expression,
                                                    data_expression,
                                                    expression_dict,
                                                    expression_tuple,
                                                    expression_list)
from pyomo.core.kernel.numvalue import (NumericValue,
                                        is_fixed,
                                        is_constant,
                                        potentially_variable)
from pyomo.core.kernel.component_variable import variable
from pyomo.core.kernel.component_parameter import parameter
from pyomo.core.kernel.component_block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)
from pyomo.core.base.expression import Expression

import six
from six import StringIO

class _Test_expression_base(object):

    _ctype_factory = None

    def test_pickle(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(e.expr(), 1.0)
        self.assertEqual(e.parent, None)
        eup = pickle.loads(
            pickle.dumps(e))
        self.assertEqual(eup.expr(), 1.0)
        self.assertEqual(eup.parent, None)
        b = block()
        b.e = e
        self.assertIs(e.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        eup = bup.e
        self.assertEqual(eup.expr(), 1.0)
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
        self.assertEqual(str(e.expr), "1.0")
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
