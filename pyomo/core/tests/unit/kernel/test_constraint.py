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
from pyomo.core.expr import logical_expr
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (IConstraint,
                                          constraint,
                                          linear_constraint,
                                          constraint_dict,
                                          constraint_tuple,
                                          constraint_list)
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import (expression,
                                          data_expression)
from pyomo.core.kernel.block import block

class Test_constraint(unittest.TestCase):

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        c = constraint((1, v**2, 2))
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

    def test_ctype(self):
        c = constraint()
        self.assertIs(c.ctype, IConstraint)
        self.assertIs(type(c), constraint)
        self.assertIs(type(c)._ctype, IConstraint)

    def test_pickle(self):
        c = constraint()
        self.assertIs(c.lb, None)
        self.assertIs(c.body, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.parent, None)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.ub, None)
        self.assertEqual(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        cup = bup.c
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.ub, None)
        self.assertIs(cup.parent, bup)

    def test_init(self):
        c = constraint()
        self.assertTrue(c.parent is None)
        self.assertEqual(c.ctype, IConstraint)
        self.assertIs(c.body, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            self.assertEqual(c(), None)
        with self.assertRaises(ValueError):
            self.assertEqual(c(exception=True), None)
        self.assertEqual(c(exception=False), None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

    def test_has_lb_ub(self):
        c = constraint()
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)

        c.lb = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), False)
        self.assertIs(c.ub, None)

        c.ub = float('inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)

        c.lb = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)

        c.ub = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)

        c.lb = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)

        c.ub = float('-inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(type(c.ub), float)

        c.rhs = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)

        c.rhs = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(type(c.ub), float)

        c.equality = False
        pL = parameter()
        c.lb = pL
        self.assertIs(c.lb, pL)
        pU = parameter()
        c.ub = pU
        self.assertIs(c.ub, pU)

        with self.assertRaises(ValueError):
            self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lb, pL)
        with self.assertRaises(ValueError):
            self.assertEqual(c.has_ub(), False)
        self.assertIs(c.ub, pU)

        pL.value = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb(), float('-inf'))
        with self.assertRaises(ValueError):
            self.assertEqual(c.has_ub(), False)
        self.assertIs(c.ub, pU)

        pU.value = float('inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb(), float('-inf'))
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub(), float('inf'))

        pL.value = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb(), 0)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub(), float('inf'))

        pU.value = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb(), 0)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub(), 0)

        pL.value = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb(), float('inf'))
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub(), 0)

        pU.value = float('-inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb(), float('inf'))
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub(), float('-inf'))

        pL.value = float('inf')
        c.rhs = pL
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb(), float('inf'))
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub(), float('inf'))

        pL.value = float('-inf')
        c.rhs = pL
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb(), float('-inf'))
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub(), float('-inf'))

    def test_bounds_getter_setter(self):
        c = constraint()
        self.assertEqual(c.bounds, (None, None))
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, None)

        c.bounds = (1,2)
        self.assertEqual(c.bounds, (1,2))
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 2)

        c.rhs = 3
        self.assertEqual(c.bounds, (3,3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (3,3)
        self.assertEqual(c.bounds, (3,3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (2,2)
        self.assertEqual(c.bounds, (3,3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (1,2)
        self.assertEqual(c.bounds, (3,3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)

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
        with self.assertRaises(ValueError):
            constraint(rhs=1, expr=v <= 1)
        c = constraint(expr=v <= 1)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub(), 1)

        with self.assertRaises(ValueError):
            constraint(rhs=1, lb=1)
        with self.assertRaises(ValueError):
            constraint(rhs=1, ub=1)
        c = constraint(rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertIs(c.body, None)

    def test_type(self):
        c = constraint()
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IConstraint))

    def test_active(self):
        c = constraint()
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        c.activate()
        self.assertEqual(c.active, True)

        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.c = c
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, False)
        c.deactivate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)

    def test_equality(self):
        v = variable()
        c = constraint(v == 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb(), 1)
        self.assertEqual(c.ub(), 1)
        self.assertEqual(c.rhs(), 1)
        self.assertEqual(c.equality, True)

        c = constraint(1 == v)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb(), 1)
        self.assertEqual(c.ub(), 1)
        self.assertEqual(c.rhs(), 1)
        self.assertEqual(c.equality, True)

        c = constraint(v - 1 == 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb(), 0)
        self.assertEqual(c.ub(), 0)
        self.assertEqual(c.rhs(), 0)
        self.assertEqual(c.equality, True)

        c = constraint(0 == v - 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb(), 0)
        self.assertEqual(c.ub(), 0)
        self.assertEqual(c.rhs(), 0)
        self.assertEqual(c.equality, True)

        c = constraint(rhs=1)
        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)

        # can not set when equality is True
        with self.assertRaises(ValueError):
            c.lb = 2
        # can not set when equality is True
        with self.assertRaises(ValueError):
            c.ub = 2

        c.equality = False

        # can not get when equality is False
        with self.assertRaises(ValueError):
            c.rhs

        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)

        # can not set to True, must set rhs to a value
        with self.assertRaises(ValueError):
            c.equality = True

        c.rhs = 3
        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        self.assertEqual(c.equality, True)

        with self.assertRaises(TypeError):
            c.rhs = 'a'

        with self.assertRaises(ValueError):
            c.rhs = None

    def test_nondata_bounds(self):
        c = constraint()
        e = expression()

        eL = expression()
        eU = expression()
        with self.assertRaises(ValueError):
            c.expr = (eL, e, eU)
        e.expr = 1.0
        eL.expr = 1.0
        eU.expr = 1.0
        with self.assertRaises(ValueError):
            c.expr = (eL, e, eU)
        with self.assertRaises(TypeError):
            c.lb = eL
        with self.assertRaises(TypeError):
            c.ub = eU

        vL = variable()
        vU = variable()
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU

        e.expr = 1.0
        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU
        with self.assertRaises(TypeError):
            c.rhs = vL

        # the fixed status of a variable
        # does not change this restriction
        vL.fixed = True
        vU.fixed = True
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU
        with self.assertRaises(TypeError):
            c.rhs = vL

        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL, 0.0, vU)
        c.body = -2.0
        c.lb = 1.0
        c.ub = 1.0
        self.assertEqual(c.slack, -3.0)
        self.assertEqual(c.lslack, -3.0)
        self.assertEqual(c.uslack, 3.0)

        with self.assertRaises(TypeError):
            c.lb = 'a'
        with self.assertRaises(TypeError):
            c.ub = 'a'
        self.assertEqual(c.lb, 1.0)
        self.assertEqual(c.ub, 1.0)

        vL.value = 2
        vU.value = 1
        c.expr = (vL <= vU)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = (vU >= vL)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = (vU <= vL)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), -1)
        self.assertEqual(c.ub, 0)
        c.expr = (vL >= vU)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), -1)
        self.assertEqual(c.ub, 0)

    def test_fixed_variable_stays_in_body(self):
        c = constraint()
        x = variable(value=0.5)
        c.expr = (0, x, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.ub, 1)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.ub, 1)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.fix(0.5)
        c.expr = (0, x, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.ub, 1)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.ub, 1)

        x.free()
        x.value = 1
        c.expr = (0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = (x == 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.fix()
        c.expr = (0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = (x == 0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)

        # ensure the variable is not moved into the upper or
        # lower bound expression (this used to be a bug)
        x.free()
        c.expr = (0 == x)
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        x.free()
        c.expr = (x == 0)
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)

    def test_data_bounds(self):
        c = constraint()
        e = expression(expr=1.0)

        pL = parameter()
        pU = parameter()
        c.expr = (pL, e, pU)
        self.assertIs(c.body, e)
        self.assertIs(c.lb, pL)
        self.assertIs(c.ub, pU)
        e.expr = None
        self.assertIs(c.body, e)
        self.assertIs(c.lb, pL)
        self.assertIs(c.ub, pU)
        c.expr = (pL, e, pU)
        self.assertIs(c.body, e)
        self.assertIs(c.lb, pL)
        self.assertIs(c.ub, pU)

        e.expr = 1.0
        eL = data_expression()
        eU = data_expression()
        c.expr = (eL, e, eU)
        self.assertIs(c.body, e)
        self.assertIs(c.lb, eL)
        self.assertIs(c.ub, eU)
        e.expr = None
        self.assertIs(c.body, e)
        self.assertIs(c.lb, eL)
        self.assertIs(c.ub, eU)
        c.expr = (eL, e, eU)
        self.assertIs(c.body, e)
        self.assertIs(c.lb, eL)
        self.assertIs(c.ub, eU)

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
        self.assertTrue(c.lb is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p <= x + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 <= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1)**2 <= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p, x, p + 1))
        self.assertEqual(c.equality, False)

        c = constraint(expr=x - p >= 0)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= p)
        self.assertTrue(c.lb is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x + 1 >= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x >= (p + 1)**2)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p, x, None))
        self.assertTrue(c.lb is p)
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
        self.assertTrue(c.ub is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x + 1 <= p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x <= p + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=x <= (p + 1)**2)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1, x, p))
        self.assertEqual(c.equality, False)

        c = constraint(expr=0 >= x - p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p >= x)
        self.assertTrue(c.ub is p)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p >= x + 1)
        self.assertEqual(c.equality, False)

        c = constraint(expr=p + 1 >= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(p + 1)**2 >= x)
        self.assertEqual(c.equality, False)

        c = constraint(expr=(None, x, p))
        self.assertTrue(c.ub is p)
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
        self.assertTrue(c.ub is p)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x + 1 == p)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x + 1 == (p + 1)**2)
        self.assertEqual(c.equality, True)

        c = constraint(expr=x == p + 1)
        self.assertEqual(c.equality, True)

        c = constraint(expr=(x, p))
        self.assertTrue(c.ub is p)
        self.assertTrue(c.lb is p)
        self.assertTrue(c.rhs is p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)

        c = constraint(expr=(p, x))
        self.assertTrue(c.ub is p)
        self.assertTrue(c.lb is p)
        self.assertTrue(c.rhs is p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)

        c = constraint(expr=logical_expr.EqualityExpression((p, x)))
        self.assertTrue(c.ub is p)
        self.assertTrue(c.lb is p)
        self.assertTrue(c.rhs is p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)

        c = constraint(expr=logical_expr.EqualityExpression((x, p)))
        self.assertTrue(c.ub is p)
        self.assertTrue(c.lb is p)
        self.assertTrue(c.rhs is p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)

    def test_tuple_construct_equality(self):
        x = variable()
        c = constraint((0.0, x))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), float)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), float)

        c = constraint((x, 0))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)

    def test_tuple_construct_inf_equality(self):
        x = variable()
        c = constraint((x, float('inf')))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)
        self.assertEqual(c.rhs, float('inf'))
        self.assertEqual(type(c.rhs), float)
        self.assertIs(c.body, x)
        c = constraint((float('inf'), x))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)
        self.assertEqual(c.rhs, float('inf'))
        self.assertEqual(type(c.rhs), float)
        self.assertIs(c.body, x)

    def test_tuple_construct_1sided_inequality(self):
        y = variable()
        c = constraint((None, y, 1))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)

        c = constraint((0, y, None))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertIs   (c.body, y)
        self.assertIs(c.ub, None)

    def test_tuple_construct_1sided_inf_inequality(self):
        y = variable()
        c = constraint((float('-inf'), y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(type(c.lb), float)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)
        self.assertEqual(type(c.ub), int)

        c = constraint((0, y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)

    def test_tuple_construct_unbounded_inequality(self):
        y = variable()
        c = constraint((None, y, None))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

        c = constraint((float('-inf'), y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(type(c.lb), float)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(type(c.ub), float)

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
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)

    def test_construct_invalid_2sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint((x, y, 1))

        with self.assertRaises(ValueError):
            constraint((0, y, z))

    def test_tuple_construct_invalid_2sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint(logical_expr.RangedExpression((x, y, 1), (False, False)))

        with self.assertRaises(ValueError):
            constraint(logical_expr.RangedExpression((0, y, z), (False, False)))

    def test_expr_construct_equality(self):
        x = variable(value=1)
        y = variable(value=1)
        c = constraint(0.0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb(), 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub(), 0)

        c = constraint(x == 0.0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb(), 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub(), 0)

        c = constraint(x == y)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb(), 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c(), 0)
        self.assertEqual(c.body(), 0)
        self.assertEqual(c.ub(), 0)

        c = constraint()
        c.expr = (x == float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)
        c.expr = (float('inf') == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)

    def test_strict_inequality_failure(self):
        x = variable()
        y = variable()
        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = (x < 0)
        with self.assertRaises(ValueError):
            c.expr = logical_expr.inequality(body=x, upper=0, strict=True)
        c.expr = (x <= 0)
        c.expr = logical_expr.inequality(body=x, upper=0, strict=False)
        with self.assertRaises(ValueError):
            c.expr = (x > 0)
        with self.assertRaises(ValueError):
            c.expr = logical_expr.inequality(body=x, lower=0, strict=True)
        c.expr = (x >= 0)
        c.expr = logical_expr.inequality(body=x, lower=0, strict=False)
        with self.assertRaises(ValueError):
            c.expr = (x < y)
        with self.assertRaises(ValueError):
            c.expr = logical_expr.inequality(body=x, upper=y, strict=True)
        c.expr = (x <= y)
        c.expr = logical_expr.inequality(body=x, upper=y, strict=False)
        with self.assertRaises(ValueError):
            c.expr = (x > y)
        with self.assertRaises(ValueError):
            c.expr = logical_expr.inequality(body=x, lower=y, strict=True)
        c.expr = (x >= y)
        c.expr = logical_expr.inequality(body=x, lower=y, strict=False)
        with self.assertRaises(ValueError):
            c.expr = logical_expr.RangedExpression((0, x, 1), (True, True))
        with self.assertRaises(ValueError):
            c.expr = logical_expr.RangedExpression((0, x, 1), (False, True))
        with self.assertRaises(ValueError):
            c.expr = logical_expr.RangedExpression((0, x, 1), (True, False))
        c.expr = logical_expr.RangedExpression((0, x, 1), (False, False))

    def test_expr_construct_inf_equality(self):
        x = variable()
        c = constraint(x == float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)
        c = constraint(float('inf') == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)

    def test_expr_construct_1sided_inequality(self):
        y = variable()
        c = constraint(y <= 1)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub(), 1)

        c = constraint(0 <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb(), 0)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

        c = constraint(y >= 1)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb(), 1)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

        c = constraint(0 >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub(), 0)

    def test_expr_construct_unbounded_inequality(self):
        y = variable()
        c = constraint(y <= float('inf'))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, float('inf'))

        c = constraint(float('-inf') <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

        c = constraint(y >= float('-inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('-inf'))
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

        c = constraint(float('inf') >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, float('inf'))

    def test_expr_construct_unbounded_inequality(self):
        y = variable()
        c = constraint(y <= float('-inf'))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertIs(c.body, y)
        c = constraint(float('inf') <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('inf'))
        self.assertIs(c.ub, None)
        self.assertIs(c.body, y)
        c = constraint(y >= float('inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('inf'))
        self.assertIs(c.ub, None)
        self.assertIs(c.body, y)
        c = constraint(float('-inf') >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertIs(c.body, y)

    def test_expr_invalid_double_sided_inequality(self):
        x = variable()
        y = variable()
        c = constraint()
        c.expr = (0, x - y, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (x, x - y, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (0, x - y, y)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (1, x - y, x)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (y, x-y, 0)

    def test_equality_infinite(self):
        c = constraint()
        v = variable()
        c.expr = (v == 1)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertIs(c.body, v)
        c.expr = (v == float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (v, float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (float('inf') == v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (float('inf'), v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, float('inf'))
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (v == float('-inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = (v, float('-inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = (float('-inf') == v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = (float('-inf'), v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('-inf'))
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)

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
        L = 1
        U = 5

        # equality
        cE = constraint(rhs=L, body=x)
        x.value = 4
        self.assertEqual(cE.body(), 4)
        self.assertEqual(cE.slack, -3)
        self.assertEqual(cE.lslack, 3)
        self.assertEqual(cE.uslack, -3)
        x.value = 6
        self.assertEqual(cE.body(), 6)
        self.assertEqual(cE.slack, -5)
        self.assertEqual(cE.lslack, 5)
        self.assertEqual(cE.uslack, -5)
        x.value = 0
        self.assertEqual(cE.body(), 0)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, -1)
        self.assertEqual(cE.uslack, 1)
        x.value = None
        with self.assertRaises(ValueError):
            cE.body()
        self.assertEqual(cE.body(exception=False), None)
        self.assertEqual(cE.slack, None)
        self.assertEqual(cE.lslack, None)
        self.assertEqual(cE.uslack, None)

        # equality
        cE = constraint(rhs=U, body=x)
        x.value = 4
        self.assertEqual(cE.body(), 4)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, -1)
        self.assertEqual(cE.uslack, 1)
        x.value = 6
        self.assertEqual(cE.body(), 6)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, 1)
        self.assertEqual(cE.uslack, -1)
        x.value = 0
        self.assertEqual(cE.body(), 0)
        self.assertEqual(cE.slack, -5)
        self.assertEqual(cE.lslack, -5)
        self.assertEqual(cE.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cE.body()
        self.assertEqual(cE.body(exception=False), None)
        self.assertEqual(cE.slack, None)
        self.assertEqual(cE.lslack, None)
        self.assertEqual(cE.uslack, None)

        # lower finite
        cL = constraint(lb=L, body=x)
        x.value = 4
        self.assertEqual(cL.body(), 4)
        self.assertEqual(cL.slack, 3)
        self.assertEqual(cL.lslack, 3)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cL.body(), 6)
        self.assertEqual(cL.slack, 5)
        self.assertEqual(cL.lslack, 5)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cL.body(), 0)
        self.assertEqual(cL.slack, -1)
        self.assertEqual(cL.lslack, -1)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cL.body()
        self.assertEqual(cL.body(exception=False), None)
        self.assertEqual(cL.slack, None)
        self.assertEqual(cL.lslack, None)
        self.assertEqual(cL.uslack, None)

        # lower unbounded
        cL = constraint(lb=float('-inf'), body=x)
        x.value = 4
        self.assertEqual(cL.body(), 4)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cL.body(), 6)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cL.body(), 0)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cL.body()
        self.assertEqual(cL.body(exception=False), None)
        self.assertEqual(cL.slack, None)
        self.assertEqual(cL.lslack, None)
        self.assertEqual(cL.uslack, None)

        # upper finite
        cU = constraint(body=x, ub=U)
        x.value = 4
        self.assertEqual(cU.body(), 4)
        self.assertEqual(cU.slack, 1)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, 1)
        x.value = 6
        self.assertEqual(cU.body(), 6)
        self.assertEqual(cU.slack, -1)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, -1)
        x.value = 0
        self.assertEqual(cU.body(), 0)
        self.assertEqual(cU.slack, 5)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cU.body()
        self.assertEqual(cU.body(exception=False), None)
        self.assertEqual(cU.slack, None)
        self.assertEqual(cU.lslack, None)
        self.assertEqual(cU.uslack, None)

        # upper unbounded
        cU = constraint(body=x, ub=float('inf'))
        x.value = 4
        self.assertEqual(cU.body(), 4)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cU.body(), 6)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cU.body(), 0)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cU.body()
        self.assertEqual(cU.body(exception=False), None)
        self.assertEqual(cU.slack, None)
        self.assertEqual(cU.lslack, None)
        self.assertEqual(cU.uslack, None)

        # range finite
        cR = constraint(lb=L, body=x, ub=U)
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, 1)
        self.assertEqual(cR.lslack, 3)
        self.assertEqual(cR.uslack, 1)
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, 5)
        self.assertEqual(cR.uslack, -1)
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, -1)
        self.assertEqual(cR.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

        # range unbounded (None)
        cR = constraint(body=x)
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

        # range unbounded
        cR = constraint(body=x, lb=float('-inf'), ub=float('inf'))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

        # range finite (parameter)
        cR = constraint(body=x,
                        lb=parameter(L),
                        ub=parameter(U))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, 1)
        self.assertEqual(cR.lslack, 3)
        self.assertEqual(cR.uslack, 1)
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, 5)
        self.assertEqual(cR.uslack, -1)
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, -1)
        self.assertEqual(cR.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

        # range unbounded (parameter)
        cR = constraint(body=x,
                        lb=parameter(float('-inf')),
                        ub=parameter(float('inf')))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

    def test_expr(self):

        x = variable(value=1.0)
        c = constraint()
        c.expr = (0, x, 2)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 2)
        self.assertEqual(c.equality, False)

        c.expr = (-2, x, 0)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lb, -2)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)

    def test_expr_getter(self):
        c = constraint()
        self.assertIs(c.expr, None)

        v = variable()

        c.expr = 0 <= v
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb(), 0)
        self.assertIs(c.body, v)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)

        c.expr = v <= 1
        self.assertIsNot(c.expr, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub(), 1)
        self.assertEqual(c.equality, False)

        c.expr = (0, v, 1)
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)

        c.expr = v == 1
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb(), 1)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub(), 1)
        self.assertEqual(c.equality, True)

        c.expr = None
        self.assertIs(c.expr, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)

    def test_expr_wrong_type(self):
        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = (2)
        with self.assertRaises(ValueError):
            c.expr = (True)

    @unittest.skipIf(not logical_expr._using_chained_inequality, "Chained inequalities are not supported.")
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

class Test_linear_constraint(unittest.TestCase):

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        c = linear_constraint(lb=1, terms=[(v,1)], ub=1)
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

    def test_ctype(self):
        c = linear_constraint([],[])
        self.assertIs(c.ctype, IConstraint)
        self.assertIs(type(c), linear_constraint)
        self.assertIs(type(c)._ctype, IConstraint)

    def test_pickle(self):
        c = linear_constraint([],[])
        self.assertIs(c.lb, None)
        self.assertEqual(c.body, 0)
        self.assertIs(c.ub, None)
        self.assertEqual(c.parent, None)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, 0)
        self.assertEqual(cup.ub, None)
        self.assertEqual(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        cup = bup.c
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, 0)
        self.assertEqual(cup.ub, None)
        self.assertIs(cup.parent, bup)

    def test_init(self):
        c = linear_constraint([],[])
        self.assertTrue(c.parent is None)
        self.assertEqual(c.ctype, IConstraint)
        self.assertEqual(c.body, 0)
        self.assertIs(c.lb, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)
        self.assertEqual(c(), 0)
        self.assertEqual(c.slack, float('inf'))
        self.assertEqual(c.lslack, float('inf'))
        self.assertEqual(c.uslack, float('inf'))

    def test_init_nonexpr(self):
        v = variable(value=3)

        c = linear_constraint()
        self.assertEqual(len(list(c.terms)), 0)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body, 0)
        self.assertEqual(c.ub, None)

        c = linear_constraint([v],[1],lb=0,ub=1)
        self.assertEqual(len(list(c.terms)), 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 3)
        self.assertEqual(c(), 3)
        self.assertEqual(c.ub, 1)
        # can't use both terms and variables
        with self.assertRaises(ValueError):
            linear_constraint(terms=(),variables=())
        # can't use both terms and coefficients
        with self.assertRaises(ValueError):
            linear_constraint(terms=(),coefficients=())
        # can't use both all three
        with self.assertRaises(ValueError):
            linear_constraint(terms=(),variables=(),coefficients=())
        # can't use only variables
        with self.assertRaises(ValueError):
            linear_constraint(variables=[v])
        # can't use only coefficients
        with self.assertRaises(ValueError):
            linear_constraint(coefficients=[1])
        # can't use both lb and rhs
        with self.assertRaises(ValueError):
            linear_constraint([v],[1],lb=0,rhs=0)
        # can't use both ub and rhs
        with self.assertRaises(ValueError):
            linear_constraint([v],[1],ub=0,rhs=0)

        c = linear_constraint([v],[1],rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body(), 3)
        self.assertEqual(c(), 3)

        c = linear_constraint([],[],rhs=1)
        c.terms = ((v, 1),)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body(), 3)
        self.assertEqual(c(), 3)

    def test_init_terms(self):
        v = variable(value=3)
        c = linear_constraint([],[],rhs=1)
        c.terms = ((v, 2),)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body(), 6)
        self.assertEqual(c(), 6)

        c = linear_constraint(terms=[(v,2)],rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body(), 6)
        self.assertEqual(c(), 6)

        terms = [(v,2)]
        c = linear_constraint(terms=iter(terms),rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body(), 6)
        self.assertEqual(c(), 6)

        c.terms = ()
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.body, 0)
        self.assertEqual(c(), 0)
        self.assertEqual(tuple(c.terms), ())

    def test_type(self):
        c = linear_constraint([],[])
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IConstraint))

    def test_active(self):
        c = linear_constraint([],[])
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        c.activate()
        self.assertEqual(c.active, True)

        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.c = c
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, False)
        c.deactivate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)

    def test_equality(self):
        v = variable()
        c = linear_constraint([v],[1],rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)

        c = linear_constraint([],[],rhs=1)
        self.assertEqual(c.body, 0)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)

        # can not set when equality is True
        with self.assertRaises(ValueError):
            c.lb = 2
        # can not set when equality is True
        with self.assertRaises(ValueError):
            c.ub = 2

        c.equality = False

        # can not get when equality is False
        with self.assertRaises(ValueError):
            c.rhs

        self.assertEqual(c.body, 0)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)

        # can not set to True, must set rhs to a value
        with self.assertRaises(ValueError):
            c.equality = True

        c.rhs = 3
        self.assertEqual(c.body, 0)
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        self.assertEqual(c.equality, True)

    def test_nondata_bounds(self):
        c = linear_constraint([],[])

        eL = expression()
        eU = expression()
        with self.assertRaises(TypeError):
            c.rhs = eL
        with self.assertRaises(TypeError):
            c.lb = eL
        with self.assertRaises(TypeError):
            c.ub = eU

        vL = variable()
        vU = variable()
        with self.assertRaises(TypeError):
            c.rhs = vL
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU

        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(TypeError):
            c.rhs = vL
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU

        # the fixed status of a variable
        # does not change this restriction
        vL.fixed = True
        vU.fixed = True
        with self.assertRaises(TypeError):
            c.rhs = vL
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU

        vL.value = -1
        vU.value = -1
        c = linear_constraint([vL, vU],
                              [1, 1],
                              lb=1.0,
                              ub=1.0)
        self.assertEqual(c(), -2.0)
        self.assertEqual(c.slack, -3.0)
        self.assertEqual(c.lslack, -3.0)
        self.assertEqual(c.uslack, 3.0)

    def test_fixed_variable_stays_in_body(self):
        x = variable(value=0.5)
        c = linear_constraint([x],[1], lb=0, ub=1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c(), 0.5)
        self.assertEqual(c.ub, 1)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], x)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c(), 2)
        self.assertEqual(c.ub, 1)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], x)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

        x.fix(0.5)
        c = linear_constraint([x],[2], lb=0, ub=1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c(), 1)
        self.assertEqual(c.ub, 1)
        repn = c.canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 1)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 4)
        self.assertEqual(c(), 4)
        self.assertEqual(c.ub, 1)
        repn = c.canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 4)

        x.free()
        x.value = 1
        c = linear_constraint([x],[1], rhs=0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c(), 1)
        self.assertEqual(c.ub, 0)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], x)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

        x.fix()
        c = linear_constraint([x],[1], rhs=0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c(), 1)
        self.assertEqual(c.ub, 0)
        repn = c.canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 1)

        x.free()
        c = linear_constraint([x],[1], rhs=0)
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c(), 1)
        self.assertEqual(c.ub, 0)
        repn = c.canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 1)

    def test_data_bounds(self):
        c = linear_constraint([],[])
        e = expression(expr=1.0)

        c.lb = 1.0
        c.ub = 2.0
        pL = parameter()
        pU = parameter()
        c.lb = pL
        self.assertIs(c.lb, pL)
        c.ub = pU
        self.assertIs(c.ub, pU)

        e.expr = 1.0
        eL = data_expression()
        eU = data_expression()
        c.lb = eL
        self.assertIs(c.lb, eL)
        c.ub = eU
        self.assertIs(c.ub, eU)

    def test_call(self):
        c = linear_constraint([],[])
        self.assertEqual(c(), 0)

        v = variable()
        c = linear_constraint([v],[2])
        with self.assertRaises(ValueError):
            c()
        with self.assertRaises(ValueError):
            c(exception=True)
        self.assertEqual(c(exception=False), None)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (2,))
        self.assertEqual(repn.constant, 0)

        v.value = 2
        self.assertEqual(c(), 4)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (2,))
        self.assertEqual(repn.constant, 0)

        v.value = None
        e = expression(v)
        c = linear_constraint([e],[1])
        with self.assertRaises(ValueError):
            c()
        with self.assertRaises(ValueError):
            c(exception=True)
        self.assertEqual(c(exception=False), None)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

        v.value = 2
        self.assertEqual(c(), 2)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

    def test_canonical_form(self):
        v = variable()
        e = expression()
        p = parameter(value=1)

        c = linear_constraint()
        self.assertEqual(c._linear_canonical_form, True)

        #
        # compute_values = True
        #

        c.terms = [(v,p)]
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

        v.fix(2)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 0)
        self.assertEqual(len(repn.linear_coefs), 0)
        self.assertEqual(repn.constant, 2)

        v.free()
        e.expr = v
        c.terms = [(e,p)]
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(repn.linear_coefs, (1,))
        self.assertEqual(repn.constant, 0)

        v.fix(2)
        repn = c.canonical_form()
        self.assertEqual(len(repn.linear_vars), 0)
        self.assertEqual(len(repn.linear_coefs), 0)
        self.assertEqual(repn.constant, 2)

        #
        # compute_values = False
        #

        v.free()
        c.terms = [(v,p)]
        repn = c.canonical_form(compute_values=False)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(len(repn.linear_coefs), 1)
        self.assertIs(repn.linear_coefs[0], p)
        self.assertEqual(repn.linear_coefs[0](), 1)
        self.assertEqual(repn.constant, 0)

        v.fix(2)
        repn = c.canonical_form(compute_values=False)
        self.assertEqual(len(repn.linear_vars), 0)
        self.assertEqual(len(repn.linear_coefs), 0)
        self.assertEqual(repn.constant(), 2)

        v.free()
        e.expr = v
        c.terms = [(e,p)]
        repn = c.canonical_form(compute_values=False)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], v)
        self.assertEqual(len(repn.linear_coefs), 1)
        self.assertIs(repn.linear_coefs[0], p)
        self.assertEqual(repn.linear_coefs[0](), 1)
        self.assertEqual(repn.constant, 0)

        v.fix(2)
        repn = c.canonical_form(compute_values=False)
        self.assertEqual(len(repn.linear_vars), 0)
        self.assertEqual(len(repn.linear_coefs), 0)
        self.assertEqual(repn.constant(), 2)

class Test_constraint_dict(_TestActiveDictContainerBase,
                           unittest.TestCase):
    _container_type = constraint_dict
    _ctype_factory = lambda self: constraint()

class Test_constraint_tuple(_TestActiveTupleContainerBase,
                            unittest.TestCase):
    _container_type = constraint_tuple
    _ctype_factory = lambda self: constraint()

class Test_constraint_list(_TestActiveListContainerBase,
                           unittest.TestCase):
    _container_type = constraint_list
    _ctype_factory = lambda self: constraint()

if __name__ == "__main__":
    unittest.main()
