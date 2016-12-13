import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_constraint import (IConstraint,
                                                  constraint,
                                                  constraint_dict,
                                                  constraint_list)
from pyomo.core.base.component_variable import variable
from pyomo.core.base.component_parameter import parameter
from pyomo.core.base.component_expression import (expression,
                                                  data_expression)
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component_block import block
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

class Test_constraint(unittest.TestCase):

    def test_pickle(self):
        c = constraint()
        self.assertEqual(c.lower, None)
        self.assertEqual(c.body, None)
        self.assertEqual(c.upper, None)
        self.assertEqual(c.parent, None)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertEqual(cup.lower, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.upper, None)
        self.assertEqual(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        cup = bup.c
        self.assertEqual(cup.lower, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.upper, None)
        self.assertIs(cup.parent, bup)

    def test_init(self):
        c = constraint()
        self.assertTrue(c.parent is None)
        self.assertEqual(c.ctype, Constraint)
        self.assertEqual(c.body, None)
        self.assertEqual(c.lower, None)
        self.assertEqual(c.upper, None)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)
        self.assertEqual(c(), None)
        self.assertEqual(c.lslack(), None)
        self.assertEqual(c.uslack(), None)

    def test_init_nonexpr(self):
        v = variable()
        c = constraint(lb=0, body=v, ub=1)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        with self.assertRaises(ValueError):
            constraint(lb=0, expr=v <= 1)
        with self.assertRaises(ValueError):
            constraint(body=v, expr=v <= 1)
        with self.assertRaises(ValueError):
            constraint(ub=1, expr=v <= 1)
        c = constraint(expr=v <= 1)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)

    def test_type(self):
        c = constraint()
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IActiveObject))
        self.assertTrue(isinstance(c, IComponent))
        self.assertTrue(isinstance(c, _IActiveComponent))
        self.assertTrue(isinstance(c, IConstraint))

    def test_equality(self):
        v = variable()
        c = constraint(v == 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lower, 1)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

        c = constraint(1 == v)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lower, 1)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

        c = constraint(v - 1 == 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

        c = constraint(0 == v - 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

    def test_nondata_bounds(self):
        c = constraint()
        e = expression()

        eL = expression()
        eU = expression()
        with self.assertRaises(ValueError):
            c.expr = (eL <= e <= eU)
        e.expr = 1.0
        eL.expr = 1.0
        eU.expr = 1.0
        with self.assertRaises(ValueError):
            c.expr = (eL <= e <= eU)
        with self.assertRaises(ValueError):
            c.lb = eL
        with self.assertRaises(ValueError):
            c.ub = eU

        vL = variable()
        vU = variable()
        with self.assertRaises(ValueError):
            c.expr = (vL <= e <= vU)
        with self.assertRaises(ValueError):
            c.lb = vL
        with self.assertRaises(ValueError):
            c.ub = vU

        e.expr = 1.0
        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL <= e <= vU)
        with self.assertRaises(ValueError):
            c.lb = vL
        with self.assertRaises(ValueError):
            c.ub = vU

        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL <= 0.0 <= vU)

    def test_fixed_variable_stays_in_body(self):
        c = constraint()
        x = variable(value=0.5)
        c.expr = (0 <= x <= 1)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.upper(), 1)
        x.value = 2
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.upper(), 1)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.fix(0.5)
        c.expr = (0 <= x <= 1)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.upper(), 1)
        x.value = 2
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.upper(), 1)

        x.free()
        x.value = 1
        c.expr = (0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)
        c.expr = (x == 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.fix()
        c.expr = (0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)
        c.expr = (x == 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.free()
        c.expr = (0 == x)
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)
        x.free()
        c.expr = (x == 0)
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.upper(), 0)

    def test_data_bounds(self):
        c = constraint()
        e = expression(expr=1.0)

        pL = parameter()
        pU = parameter()
        c.expr = (pL <= e <= pU)
        e.expr = None
        c.expr = (pL <= e <= pU)

        e.expr = 1.0
        eL = data_expression()
        eU = data_expression()
        c.expr = (eL <= e <= eU)
        e.expr = None
        c.expr = (eL <= e <= eU)

    # make sure we can use a mutable param that
    # has not been given a value in the upper bound
    # of an inequality constraint
    def test_mutable_novalue_param_lower_bound(self):
        x = variable()
        p = parameter()
        p.value = None

        c = constraint(expr=0 <= x - p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p <= x)
        self.assertTrue(c.lower is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p <= x + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 <= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1)**2 <= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p <= x <= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x - p >= 0)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= p)
        self.assertTrue(c.lower is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x + 1 >= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= (p + 1)**2)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 >= x >= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p, x, None))
        self.assertTrue(c.lower is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p, x + 1, None))
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1, x, None))
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p, x, 1))
        self.assertEqual(c.equality, False)

    # make sure we can use a mutable param that
    # has not been given a value in the lower bound
    # of an inequality constraint
    def test_mutable_novalue_param_upper_bound(self):
        x = variable()
        p = parameter()
        p.value = None

        c = constraint(expr=x - p <= 0)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x <= p)
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x + 1 <= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x <= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x <= (p + 1)**2)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 <= x <= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=0 >= x - p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p >= x)
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p >= x + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 >= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1)**2 >= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p >= x >= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(None, x, p))
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(None, x + 1, p))
        self.assertEqual(c.equality, False)

        c = constraint(expr=(None, x, p + 1))
        self.assertEqual(c.equality, False)

        c = constraint(expr=(1, x, p))
        self.assertEqual(c.equality, False)

    # make sure we can use a mutable param that
    # has not been given a value in the rhs of
    # of an equality constraint
    def test_mutable_novalue_param_equality(self):
        x = variable()
        p = parameter()
        p.value = None

        c = constraint(expr=x - p == 0)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x == p)
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x + 1 == p)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x + 1 == (p + 1)**2)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x == p + 1)
        self.assertEqual(c.equality, True)

        c = constraint(expr=p <= x <= p)
        self.assertTrue(c.upper is p)
        # GH: Not sure if we are supposed to detect equality
        #     in this situation. I would rather us not, for
        #     the sake of making the code less complicated.
        #     Either way, I am not going to test for it here.
        #self.assertEqual(c.equality, <blah>)

        c = constraint(expr=(x, p))
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, True)

        c = constraint(expr=(p, x))
        self.assertTrue(c.upper is p)
        self.assertEqual(c.equality, True)

    def test_tuple_construct_equality(self):
        x = variable()
        c = constraint((0.0, x))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.upper, 0)

        c = constraint((x, 0.0))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.upper, 0)

    def test_tuple_construct_inf_equality(self):
        x = variable()
        with self.assertRaises(ValueError):
            constraint((x, float('inf')))

        with self.assertRaises(ValueError):
            constraint((float('inf'), x))

    def test_tuple_construct_1sided_inequality(self):
        y = variable()
        c = constraint((None, y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, 1)

        c = constraint((0, y, None))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, 0)
        self.assertIs   (c.body, y)
        self.assertEqual(c.upper, None)

    def test_tuple_construct_1sided_inf_inequality(self):
        y = variable()
        c = constraint((float('-inf'), y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, 1)

        c = constraint((0, y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

    def test_tuple_construct_unbounded_inequality(self):
        y = variable()
        c = constraint((None, y, None))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint((float('-inf'), y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

    def test_tuple_construct_invalid_1sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint((x, y, None))

        with self.assertRaises(ValueError):
            constraint((None, y, z))

    def test_tuple_construct_2sided_inequality(self):
        y = variable()
        c = constraint((0, y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, 1)

    def test_tuple_construct_invalid_2sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint((x, y, 1))

        with self.assertRaises(ValueError):
            constraint((0, y, z))

    def test_expr_construct_equality(self):
        x = variable(value=1)
        y = variable(value=1)
        c = constraint(0.0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.upper, 0)

        c = constraint(x == 0.0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.upper, 0)

        c = constraint(x == y)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lower, 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c(), 0)
        self.assertEqual(c.body(), 0)
        self.assertEqual(c.upper, 0)

        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = (x == float('inf'))
        with self.assertRaises(ValueError):
            c.expr = (float('inf') == x)

    def test_strict_inequality_failure(self):
        x = variable()
        y = variable()
        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = (x < 0)
        with self.assertRaises(ValueError):
            c.expr = (x > 0)
        with self.assertRaises(ValueError):
            c.expr = (x < y)
        with self.assertRaises(ValueError):
            c.expr = (x > y)

    def test_expr_construct_inf_equality(self):
        x = variable()
        with self.assertRaises(ValueError):
            constraint(x == float('inf'))

        with self.assertRaises(ValueError):
            constraint(float('inf') == x)

    def test_expr_construct_1sided_inequality(self):
        y = variable()
        c = constraint(y <= 1)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, 1)

        c = constraint(0 <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, 0)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint(y >= 1)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, 1)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint(0 >= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, 0)

    def test_expr_construct_unbounded_inequality(self):
        y = variable()
        c = constraint(y <= float('inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint(float('-inf') <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint(y >= float('-inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

        c = constraint(float('inf') >= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lower, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.upper, None)

    def test_expr_construct_invalid_unbounded_inequality(self):
        y = variable()
        with self.assertRaises(ValueError):
            constraint(y <= float('-inf'))

        with self.assertRaises(ValueError):
            constraint(float('inf') <= y)

        with self.assertRaises(ValueError):
            constraint(y >= float('inf'))

        with self.assertRaises(ValueError):
            constraint(float('-inf') >= y)

    def test_expr_invalid_double_sided_inequality(self):
        x = variable()
        y = variable()
        c = constraint()
        c.expr = (0 <= x - y <= 1)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (x <= x - y <= 1)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (0 <= x - y <= y)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (x >= x - y >= 1)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (0 >= x - y >= y)

    def test_equality_infinite(self):
        c = constraint()
        v = variable()
        c.expr = (v == 1)
        with self.assertRaises(ValueError):
            c.expr = (v == float('inf'))
        with self.assertRaises(ValueError):
            c.expr = (v, float('inf'))
        with self.assertRaises(ValueError):
            c.expr = (float('inf') == v)
        with self.assertRaises(ValueError):
            c.expr = (float('inf'), v)
        with self.assertRaises(ValueError):
            c.expr = (v == float('-inf'))
        with self.assertRaises(ValueError):
            c.expr = (v, float('-inf'))
        with self.assertRaises(ValueError):
            c.expr = (float('-inf') == v)
        with self.assertRaises(ValueError):
            c.expr = (float('-inf'), v)

    def test_equality_infinite(self):
        c = constraint()
        v = variable()
        c.expr = (v == 1)
        with self.assertRaises(ValueError):
            c.expr = (v == float('inf'))
        with self.assertRaises(ValueError):
            c.expr = (v, float('inf'))
        with self.assertRaises(ValueError):
            c.expr = (float('inf') == v)
        with self.assertRaises(ValueError):
            c.expr = (float('inf'), v)
        with self.assertRaises(ValueError):
            c.expr = (v == float('-inf'))
        with self.assertRaises(ValueError):
            c.expr = (v, float('-inf'))
        with self.assertRaises(ValueError):
            c.expr = (float('-inf') == v)
        with self.assertRaises(ValueError):
            c.expr = (float('-inf'), v)

    def test_equality_nonnumeric(self):
        c = constraint()
        v = variable()
        c.expr = (v == 1)
        with self.assertRaises(TypeError):
            c.expr = (v, 'x')
        with self.assertRaises(TypeError):
            c.expr = ('x', v)

    def test_slack_methods(self):
        x = variable(value=2)
        L = -1.0
        U = 5.0
        cL = constraint(x**2 >= L)
        self.assertEqual(cL.lslack(), -5.0)
        self.assertEqual(cL.uslack(), float('inf'))
        cU = constraint(x**2 <= U)
        self.assertEqual(cU.lslack(), float('-inf'))
        self.assertEqual(cU.uslack(), 1.0)
        cR = constraint(L <= x**2 <= U)
        self.assertEqual(cR.lslack(), -5.0)
        self.assertEqual(cR.uslack(), 1.0)

    def test_expr(self):

        x = variable(value=1.0)
        c = constraint()
        c.expr = (2 >= x >= 0)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lower(), 0)
        self.assertEqual(c.upper(), 2)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

        c.expr = (0 >= x >= -2)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lower(), -2)
        self.assertEqual(c.upper(), 0)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.strict_lower, False)
        self.assertEqual(c.strict_upper, False)

    def test_expr_getter(self):
        c = constraint()
        self.assertIs(c.expr, None)

        v = variable()

        c.expr = 0 <= v
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, None)
        self.assertEqual(c.equality, False)

        c.expr = v <= 1
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)

        c.expr = 0 <= v <= 1
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)

        c.expr = v == 1
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 1)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, True)

        c.expr = None
        self.assertIs(c.expr, None)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, None)
        self.assertEqual(c.ub, None)
        self.assertEqual(c.equality, False)

    def test_expr_wrong_type(self):
        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = (2)
        with self.assertRaises(ValueError):
            c.expr = (True)

    def test_chainedInequalityError(self):
        x = variable()
        c = constraint()
        a = x <= 0
        if x <= 0:
            pass
        def f():
            c.expr = a
        self.assertRaisesRegexp(
            TypeError, "Relational expression used in an unexpected "
            "Boolean context.", f)

    def test_tuple_constraint_create(self):
        x = variable()
        y = variable()
        z = variable()
        c = constraint((0.0,x))
        with self.assertRaises(ValueError):
            constraint((y,x,z))
        with self.assertRaises(ValueError):
            constraint((0,x,z))
        with self.assertRaises(ValueError):
            constraint((y,x,0))
        with self.assertRaises(ValueError):
            constraint((x,0,0,0))

        c = constraint((x, y))
        self.assertEqual(c.upper, 0)
        self.assertEqual(c.lower, 0)
        self.assertTrue(c.body is not None)

    def test_expression_constructor_coverage(self):
        x = variable()
        y = variable()
        z = variable()
        L = parameter(value=0)
        U = parameter(value=1)

        expr = U >= x
        expr = expr >= L
        c = constraint(expr)

        expr = x <= z
        expr = expr >= y
        with self.assertRaises(ValueError):
            constraint(expr)

        expr = x >= z
        expr = y >= expr
        with self.assertRaises(ValueError):
            constraint(expr)

        expr = y <= x
        expr = y >= expr
        with self.assertRaises(ValueError):
            constraint(expr)

        L.value = 0
        c = constraint(x >= L)

        U.value = 0
        c = constraint(U >= x)

        L.value = 0
        U.value = 1
        expr = U <= x
        expr = expr <= L
        c = constraint(expr)

        expr = x >= z
        expr = expr <= y
        with self.assertRaises(ValueError):
            constraint(expr)

        expr = x <= z
        expr = y <= expr
        with self.assertRaises(ValueError):
            constraint(expr)

        expr = y >= x
        expr = y <= expr
        with self.assertRaises(ValueError):
            constraint(expr)

        L.value = 0
        expr = x <= L
        c = constraint(expr)

        U.value = 0
        expr = U <= x
        c = constraint(expr)


        x = variable()
        with self.assertRaises(ValueError):
            constraint(x+x)

class Test_constraint_dict(_TestActiveComponentDictBase,
                           unittest.TestCase):
    _container_type = constraint_dict
    _ctype_factory = lambda self: constraint()

class Test_constraint_list(_TestActiveComponentListBase,
                           unittest.TestCase):
    _container_type = constraint_list
    _ctype_factory = lambda self: constraint()

if __name__ == "__main__":
    unittest.main()
