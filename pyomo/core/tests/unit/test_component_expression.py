import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer,
                                                 IBlockStorage)
from pyomo.core.tests.unit.test_component_dict import \
    _TestComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestComponentListBase
from pyomo.core.base.component_expression import (IExpression,
                                                  expression,
                                                  expression_dict,
                                                  expression_list)
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.component_variable import variable
from pyomo.core.base.expression import Expression
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

import six
from six import StringIO

class Test_expression(unittest.TestCase):

    def test_init_no_args(self):
        e = expression()
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, Expression)
        self.assertTrue(e.expr is None)

    def test_init_args(self):
        e = expression(1.0)
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, Expression)
        self.assertTrue(e.expr is not None)

    def test_type(self):
        e = expression()
        self.assertTrue(isinstance(e, ICategorizedObject))
        self.assertTrue(isinstance(e, IComponent))
        self.assertTrue(isinstance(e, IExpression))
        self.assertTrue(isinstance(e, NumericValue))

    def test_call(self):
        e = expression()
        self.assertEqual(e(), None)
        e.expr = 1
        self.assertEqual(e(), 1)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e(), 3)

    def test_is_constant(self):
        e = expression()
        self.assertEqual(e.is_constant(), False)
        e.expr = 1
        self.assertEqual(e.is_constant(), False)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_constant(), False)

    def test_is_fixed(self):
        e = expression()
        self.assertEqual(e.is_fixed(), True)
        e.expr = 1
        self.assertEqual(e.is_fixed(), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_fixed(), False)
        v.fix()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(e(), 3)

    def test_is_expression(self):
        e = expression()
        self.assertEqual(e.is_expression(), True)

    def test_args(self):
        e = expression()
        v = variable()
        e.expr = v + 1
        self.assertEqual(len(e._args), 1)
        self.assertTrue(e._args[0] is e.expr)

    def test_aruments(self):
        e = expression()
        v = variable()
        e.expr = v + 1
        self.assertEqual(len(tuple(e._arguments())), 1)
        self.assertTrue(tuple(e._arguments())[0] is e.expr)

    def test_clone(self):
        e = expression()
        self.assertTrue(e.clone() is e)

    def test_polynomial_degree(self):
        e = expression()
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

    def test_to_string(self):
        b = block()
        e = expression()
        self.assertEqual(str(e.expr), "None")
        self.assertEqual(str(e), "<expression>")
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
        self.assertEqual(out.getvalue(), "<expression>{Undefined}")
        b.e = e
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "e{Undefined}")
        del b.e

        e.expr = 1
        self.assertEqual(str(e.expr), "1.0")
        self.assertEqual(str(e), "<expression>")
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
        self.assertEqual(out.getvalue(), "<expression>{1.0}")
        b.e = e
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(), "e{1.0}")
        del b.e


        v = variable()
        e.expr = v**2
        self.assertEqual(str(e.expr), "<variable>**2.0")
        self.assertEqual(str(e), "<expression>")
        e.to_string()
        out = StringIO()
        e.to_string(ostream=out)
        self.assertEqual(out.getvalue(), "<variable>**2.0")
        e.to_string(verbose=False)
        out = StringIO()
        e.to_string(ostream=out, verbose=False)
        self.assertEqual(out.getvalue(), "<variable>**2.0")
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         "<expression>{pow( <variable> , 2.0 )}")
        b.e = e
        b.v = v
        e.to_string(verbose=True)
        out = StringIO()
        e.to_string(ostream=out, verbose=True)
        self.assertEqual(out.getvalue(),
                         "e{pow( v , 2.0 )}")
        del b.e
        del b.v

class Test_expression_dict(_TestComponentDictBase,
                           unittest.TestCase):
    _container_type = expression_dict
    _ctype_factory = lambda self: expression()

class Test_expression_list(_TestComponentListBase,
                           unittest.TestCase):
    _container_type = expression_list
    _ctype_factory = lambda self: expression()

if __name__ == "__main__":
    unittest.main()
