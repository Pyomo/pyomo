#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy

from six import StringIO
from pyomo.core.expr import expr_common


import pyutilib.th as unittest
from pyutilib.misc.redirect_io import capture_output

from pyomo.environ import ConcreteModel, AbstractModel, Expression, Var, Set, Param, Objective, value, sum_product
from pyomo.core.base.expression import _GeneralExpressionData

class TestExpressionData(unittest.TestCase):

    def test_exprdata_get_set(self):
        model = ConcreteModel()
        model.e = Expression([1])
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e[1].expr, None)
        model.e.add(1,1)
        self.assertEqual(model.e[1].expr(), 1)
        model.e[1].expr += 2
        self.assertEqual(model.e[1].expr(), 3)

    def test_exprdata_get_set_value(self):
        model = ConcreteModel()
        model.e = Expression([1])
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e[1].expr, None)
        model.e.add(1,1)
        model.e[1].expr = 1
        self.assertEqual(model.e[1].expr(), 1)
        model.e[1].expr += 2
        self.assertEqual(model.e[1].expr(), 3)

    # The copy method must be invoked on expression container to obtain
    # a shallow copy of the class, the underlying expression remains
    # a reference.
    def test_copy(self):
        model = ConcreteModel()
        model.a = Var(initialize=5)
        model.b = Var(initialize=10)

        model.expr1 = Expression(initialize=model.a)

        # Do a shallow copy, the same underlying expression is still referenced
        expr2 = copy.copy(model.expr1)
        self.assertEqual( model.expr1(), 5 )
        self.assertEqual( expr2(), 5 )
        self.assertEqual( id(model.expr1.expr), id(expr2.expr) )

        # Do an in place modification the expression
        model.expr1.expr.set_value(1)
        self.assertEqual( model.expr1(), 1 )
        self.assertEqual( expr2(), 1 )
        self.assertEqual( id(model.expr1.expr), id(expr2.expr) )

        # Update the expression value on expr1 only
        model.expr1.set_value(model.b)
        self.assertEqual( model.expr1(), 10 )
        self.assertEqual( expr2(), 1 )
        self.assertNotEqual( id(model.expr1.expr), id(expr2.expr) )

        model.a.set_value(5)
        model.b.set_value(10)
        model.del_component('expr1')
        model.expr1 = Expression(initialize=model.a + model.b)

        # Do a shallow copy, the same underlying expression is still referenced
        expr2 = copy.copy(model.expr1)
        self.assertEqual( model.expr1(), 15 )
        self.assertEqual( expr2(), 15 )
        self.assertEqual( id(model.expr1.expr), id(expr2.expr) )
        self.assertEqual( id(model.expr1.expr.arg(0)),
                          id(expr2.expr.arg(0)) )
        self.assertEqual( id(model.expr1.expr.arg(1)),
                          id(expr2.expr.arg(1)) )


        # Do an in place modification the expression
        # This causes cloning due to reference counting
        model.a.set_value(0)
        self.assertEqual( model.expr1(), 10 )
        self.assertEqual( expr2(), 10 )
        self.assertEqual( id(model.expr1.expr), id(expr2.expr) )
        self.assertEqual( id(model.expr1.expr.arg(0)),
                          id(expr2.expr.arg(0)) )
        self.assertEqual( id(model.expr1.expr.arg(1)),
                          id(expr2.expr.arg(1)) )


        # Do an in place modification the expression
        # This causes cloning due to reference counting
        model.expr1.expr += 1
        self.assertEqual( model.expr1(), 11 )
        self.assertEqual( expr2(), 10 )
        self.assertNotEqual( id(model.expr1.expr), id(expr2.expr) )

    # test that an object is properly deepcopied when the model is cloned
    def test_model_clone(self):
        model = ConcreteModel()
        model.x = Var(initialize=2.0)
        model.y = Var(initialize=0.0)
        model.ec = Expression(initialize=model.x**2+1)
        model.obj = Objective(expr=model.y+model.ec)
        self.assertEqual(model.obj.expr(),5.0)
        self.assertTrue(id(model.ec) in [id(e) for e in model.obj.expr.args])
        inst = model.clone()
        self.assertEqual(inst.obj.expr(),5.0)
        if not id(inst.ec) in [id(e) for e in inst.obj.expr.args]:
            print("BUG?")
            print(id(inst.ec))
            print(inst.obj.expr.__class__)
            print([id(e) for e in inst.obj.expr.args])
            print([e.__class__ for e in inst.obj.expr.args])
            print([id(e) for e in model.obj.expr.args])
            print([e.__class__ for e in model.obj.expr.args])
        self.assertTrue(id(inst.ec) in [id(e) for e in inst.obj.expr.args])
        self.assertNotEqual(id(model.ec), id(inst.ec))
        self.assertFalse(id(inst.ec) in [id(e) for e in model.obj.expr.args])

    def test_is_constant(self):
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.p = Param(initialize=1.0)
        model.ec = Expression(initialize=model.x)
        self.assertEqual(model.ec.is_constant(), False)
        self.assertEqual(model.ec.expr.is_constant(), False)
        model.ec.set_value(model.p)
        self.assertEqual(model.ec.is_constant(), False)
        self.assertEqual(model.ec.expr.is_constant(), True)

    def test_polynomial_degree(self):
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.ec = Expression(initialize=model.x)
        self.assertEqual( model.ec.polynomial_degree(),
                          model.ec.expr.polynomial_degree() )
        self.assertEqual(model.ec.polynomial_degree(), 1)
        model.ec.set_value(model.x**2)
        self.assertEqual( model.ec.polynomial_degree(),
                          model.ec.expr.polynomial_degree())
        self.assertEqual( model.ec.polynomial_degree(), 2 )


    def test_init_concrete(self):
        model = ConcreteModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)

        model.ec = Expression(expr=0)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=model.y)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.y.set_value(-1)
        model.ec = Expression(initialize=model.y+1.0)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

    def test_init_abstract(self):
        model = AbstractModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)
        model.ec = Expression(initialize=0.0)

        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)
        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)
        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))


class TestExpression(unittest.TestCase):

    def setUp(self):
        TestExpression._save = expr_common.TO_STRING_VERBOSE
        # Tests can choose what they want - this just makes sure that
        #things are restored after the tests run.
        #expr_common.TO_STRING_VERBOSE = True

    def tearDown(self):
        expr_common.TO_STRING_VERBOSE = TestExpression._save

    def test_unconstructed_singleton(self):
        a = Expression()
        self.assertEqual(a._constructed, False)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.expr
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.is_constant()
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.is_fixed()
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.set_value(4)
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        a.construct()
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), None)
        self.assertEqual(a.expr, None)
        self.assertEqual(a.is_constant(), False)
        a.set_value(5)
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 5)
        self.assertEqual(a.expr(), 5)
        self.assertEqual(a.is_constant(), False)
        self.assertEqual(a.is_fixed(), True)


    def test_bad_init_wrong_type(self):
        model = ConcreteModel()
        def _some_rule(model):
            return 1.0
        with self.assertRaises(TypeError):
            model.e = Expression(expr=_some_rule)
        with self.assertRaises(TypeError):
            model.e = Expression([1], expr=_some_rule)
        del _some_rule

    def test_display(self):
        model = ConcreteModel()
        model.e = Expression()
        with capture_output() as out:
            model.e.display()
        self.assertEqual(out.getvalue().strip(), """
e : Size=1
    Key  : Value
    None : Undefined
        """.strip())

        model.e.set_value(1.0)
        with capture_output() as out:
            model.e.display()
        self.assertEqual(out.getvalue().strip(), """
e : Size=1
    Key  : Value
    None :   1.0
        """.strip())

        out = StringIO()
        with capture_output() as no_out:
            model.e.display(ostream=out)
        self.assertEqual(no_out.getvalue(), "")
        self.assertEqual(out.getvalue().strip(), """
e : Size=1
    Key  : Value
    None :   1.0
        """.strip())

        model.E = Expression([1,2])
        with capture_output() as out:
            model.E.display()
        self.assertEqual(out.getvalue().strip(), """
E : Size=2
    Key : Value
      1 : Undefined
      2 : Undefined
        """.strip())

        model.E[1].set_value(1.0)
        with capture_output() as out:
            model.E.display()
        self.assertEqual(out.getvalue().strip(), """
E : Size=2
    Key : Value
      1 :       1.0
      2 : Undefined
        """.strip())

        out = StringIO()
        with capture_output() as no_out:
            model.E.display(ostream=out)
        self.assertEqual(no_out.getvalue(), "")
        self.assertEqual(out.getvalue().strip(), """
E : Size=2
    Key : Value
      1 :       1.0
      2 : Undefined
        """.strip())

    def test_extract_values_store_values(self):
        model = ConcreteModel()
        model.e = Expression()
        self.assertEqual(model.e.extract_values(),
                         {None: None})
        model.e.store_values({None: 1.0})
        self.assertEqual(model.e.extract_values(),
                         {None: 1.0})
        with self.assertRaises(KeyError):
            model.e.store_values({1: 1.0})

        model.E = Expression([1,2])
        self.assertEqual(model.E.extract_values(),
                         {1: None, 2:None})
        model.E.store_values({1: 1.0})
        self.assertEqual(model.E.extract_values(),
                         {1: 1.0, 2: None})
        model.E.store_values({1: None, 2: 2.0})
        self.assertEqual(model.E.extract_values(),
                         {1: None, 2: 2.0})
        with self.assertRaises(KeyError):
            model.E.store_values({3: 3.0})

    def test_setitem(self):
        model = ConcreteModel()
        model.E = Expression([1])
        model.E[1] = 1
        self.assertEqual(model.E[1], 1)
        with self.assertRaises(KeyError):
            model.E[2] = 1
        model.del_component(model.E)
        model.Index = Set(dimen=3, initialize=[(1,2,3)])
        model.E = Expression(model.Index)
        model.E[(1,2,3)] = 1
        self.assertEqual(model.E[(1,2,3)], 1)
        # GH: testing this ludicrous behavior simply for
        #     coverage in expression.py.
        model.E[(1,(2,3))] = 1
        self.assertEqual(model.E[(1,2,3)], 1)
        with self.assertRaises(KeyError):
            model.E[2] = 1

    def test_nonindexed_construct_rule(self):
        model = ConcreteModel()
        def _some_rule(model):
            return 1.0
        model.e = Expression(rule=_some_rule)
        self.assertEqual(value(model.e), 1.0)
        model.del_component(model.e)
        del _some_rule
        def _some_rule(model):
            return Expression.Skip
        # non-indexed Expression does not recognized
        # Expression.Skip
        with self.assertRaises(ValueError):
            model.e = Expression(rule=_some_rule)

    def test_nonindexed_construct_expr(self):
        model = ConcreteModel()
        # non-indexed Expression does not recognized
        # Expression.Skip
        with self.assertRaises(ValueError):
            model.e = Expression(expr=Expression.Skip)
        model.e = Expression()
        self.assertEqual(model.e.extract_values(),
                         {None: None})
        model.del_component(model.e)
        model.e = Expression(expr=1.0)
        self.assertEqual(model.e.extract_values(),
                         {None: 1.0})
        model.del_component(model.e)
        model.e = Expression(expr={None: 1.0})
        self.assertEqual(model.e.extract_values(),
                         {None: 1.0})
        # Even though add can be called with any
        # indexed on indexed Expressions, None must
        # always be used as the index for non-indexed
        # Expressions
        with self.assertRaises(KeyError):
            model.e.add(2, 2)

    def test_indexed_construct_rule(self):
        model = ConcreteModel()
        model.Index = Set(initialize=[1,2,3])
        def _some_rule(model, i):
            if i == 1:
                return Expression.Skip
            else:
                return i
        model.E = Expression(model.Index,
                             rule=_some_rule)
        self.assertEqual(model.E.extract_values(),
                         {2:2, 3:3})
        self.assertEqual(len(model.E), 2)

    def test_implicit_definition(self):
        model = ConcreteModel()
        model.idx = Set(initialize=[1,2,3])
        model.E = Expression(model.idx, rule=lambda m,i: Expression.Skip)
        self.assertEqual(len(model.E), 0)
        expr = model.E[1]
        self.assertIs(type(expr), _GeneralExpressionData)
        self.assertIs(expr.value, None)
        model.E[1] = 5
        self.assertIs(expr, model.E[1])
        self.assertEqual(model.E.extract_values(), {1:5})
        model.E[2] = 6
        self.assertIsNot(expr, model.E[2])
        self.assertEqual(model.E.extract_values(), {1:5, 2:6})

    def test_indexed_construct_expr(self):
        model = ConcreteModel()
        model.Index = Set(initialize=[1,2,3])
        model.E = Expression(model.Index,
                             expr=Expression.Skip)
        self.assertEqual(len(model.E), 0)
        model.E = Expression(model.Index)
        self.assertEqual(model.E.extract_values(),
                         {1:None, 2:None, 3:None})
        model.del_component(model.E)
        model.E = Expression(model.Index, expr=1.0)
        self.assertEqual(model.E.extract_values(),
                         {1:1.0, 2:1.0, 3:1.0})
        model.del_component(model.E)
        model.E = Expression(model.Index,
                             expr={1: Expression.Skip,
                                   2: Expression.Skip,
                                   3: 1.0})
        self.assertEqual(model.E.extract_values(),
                         {3: 1.0})

    def test_bad_init_too_many_keywords(self):
        model = ConcreteModel()
        def _some_rule(model):
            return 1.0
        with self.assertRaises(ValueError):
            model.e = Expression(expr=1.0,
                                 rule=_some_rule)
        del _some_rule
        def _some_indexed_rule(model, i):
            return 1.0
        with self.assertRaises(ValueError):
            model.e = Expression([1],
                                 expr=1.0,
                                 rule=_some_indexed_rule)
        del _some_indexed_rule

    def test_init_concrete_indexed(self):
        model = ConcreteModel()
        model.y = Var(initialize=0.0)
        model.x = Var([1,2,3],initialize=1.0)

        model.ec = Expression([1,2,3],initialize=1.0)
        model.obj = Objective(expr=1.0+sum_product(model.ec, index=[1,2,3]))
        self.assertEqual(model.obj.expr(),4.0)
        model.ec[1].set_value(2.0)
        self.assertEqual(model.obj.expr(),5.0)

    def test_init_concrete_nonindexed(self):
        model = ConcreteModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)

        model.ec = Expression(initialize=0)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=model.y)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.y.set_value(-1)
        model.ec = Expression(initialize=model.y+1.0)
        model.obj = Objective(expr=1.0+model.ec)
        self.assertEqual(model.obj.expr(),1.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e = 1.0
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),2.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        model.ec.set_value(e)
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))
        e += model.x
        self.assertEqual(model.obj.expr(),3.0)
        self.assertEqual(id(model.obj.expr.arg(1)),id(model.ec))

    def test_init_abstract_indexed(self):
        model = AbstractModel()
        model.ec = Expression([1,2,3],initialize=1.0)
        model.obj = Objective(rule=lambda m: 1.0+sum_product(m.ec,index=[1,2,3]))
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),4.0)
        inst.ec[1].set_value(2.0)
        self.assertEqual(inst.obj.expr(),5.0)

    def test_init_abstract_nonindexed(self):
        model = AbstractModel()
        model.y = Var(initialize=0.0)
        model.x = Var(initialize=1.0)
        model.ec = Expression(initialize=0.0)

        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)
        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))

        model.del_component('obj')
        model.del_component('ec')
        model.ec = Expression(initialize=0.0)
        def obj_rule(model):
            return 1.0+model.ec
        model.obj = Objective(rule=obj_rule)
        inst = model.create_instance()
        self.assertEqual(inst.obj.expr(),1.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e = 1.0
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),2.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        inst.ec.set_value(e)
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))
        e += inst.x
        self.assertEqual(inst.obj.expr(),3.0)
        self.assertEqual(id(inst.obj.expr.arg(1)),id(inst.ec))

    def test_pprint_oldStyle(self):
        expr_common.TO_STRING_VERBOSE = True

        model = ConcreteModel()
        model.x = Var()
        model.e = Expression(initialize=model.x+2)
        model.E = Expression([1,2],initialize=model.x**2+1)
        expr = model.e*model.x**2 + model.E[1]

        output = \
"""\
sum(prod(e{sum(x, 2)}, pow(x, 2)), E[1]{sum(pow(x, 2), 1)})
e : Size=1, Index=None
    Key  : Expression
    None : sum(x, 2)
E : Size=2, Index=E_index
    Key : Expression
      1 : sum(pow(x, 2), 1)
      2 : sum(pow(x, 2), 1)
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

        model.e.set_value(1.0)
        model.E[1].set_value(2.0)
        output = \
"""\
sum(prod(e{1.0}, pow(x, 2)), E[1]{2.0})
e : Size=1, Index=None
    Key  : Expression
    None :        1.0
E : Size=2, Index=E_index
    Key : Expression
      1 : 2.0
      2 : sum(pow(x, 2), 1)
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())


        model.e.set_value(None)
        model.E[1].set_value(None)
        output = \
"""\
sum(prod(e{Undefined}, pow(x, 2)), E[1]{Undefined})
e : Size=1, Index=None
    Key  : Expression
    None :  Undefined
E : Size=2, Index=E_index
    Key : Expression
      1 : Undefined
      2 : sum(pow(x, 2), 1)
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())


    def test_pprint_newStyle(self):
        expr_common.TO_STRING_VERBOSE = False

        model = ConcreteModel()
        model.x = Var()
        model.e = Expression(initialize=model.x+2)
        model.E = Expression([1,2],initialize=model.x**2+1)
        expr = model.e*model.x**2 + model.E[1]

        output = \
"""\
(x + 2)*x**2 + (x**2 + 1)
e : Size=1, Index=None
    Key  : Expression
    None : x + 2
E : Size=2, Index=E_index
    Key : Expression
      1 : x**2 + 1
      2 : x**2 + 1
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

        model.e.set_value(1.0)
        model.E[1].set_value(2.0)
        #
        # WEH - the 1.0 seems unnecessary here, but it results from
        # a fixed variable in a sub-expression.  I can't decide if this
        # is the expected behavior or not.
        #
        output = \
"""\
x**2 + 2.0
e : Size=1, Index=None
    Key  : Expression
    None :        1.0
E : Size=2, Index=E_index
    Key : Expression
      1 : 2.0
      2 : x**2 + 1
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())


        model.e.set_value(None)
        model.E[1].set_value(None)
        output = \
"""\
e{None}*x**2 + E[1]{None}
e : Size=1, Index=None
    Key  : Expression
    None :  Undefined
E : Size=2, Index=E_index
    Key : Expression
      1 : Undefined
      2 : x**2 + 1
"""
        out = StringIO()
        out.write(str(expr)+"\n")
        model.e.pprint(ostream=out)
        #model.E[1].pprint(ostream=out)
        model.E.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    def test_len(self):
        model = AbstractModel()
        model.e = Expression()

        self.assertEqual(len(model.e), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.e), 1)

    def test_None_key(self):
        model = AbstractModel()
        model.e = Expression()
        inst = model.create_instance()
        self.assertEqual(id(inst.e), id(inst.e[None]))

    def test_singleton_get_set(self):
        model = ConcreteModel()
        model.e = Expression()
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e.expr, None)
        model.e.expr = 1
        self.assertEqual(model.e.expr(), 1)
        model.e.expr += 2
        self.assertEqual(model.e.expr(), 3)

    def test_singleton_get_set_value(self):
        model = ConcreteModel()
        model.e = Expression()
        self.assertEqual(len(model.e), 1)
        self.assertEqual(model.e.expr, None)
        model.e.expr = 1
        self.assertEqual(model.e.expr(), 1)
        model.e.expr += 2
        self.assertEqual(model.e.expr(), 3)

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Expression(model.C)

    def test_iadd(self):
        # make sure simple for loops that look like they
        # create a new expression do not modify the named
        # expression
        m = ConcreteModel()
        e = m.e = Expression(expr=1.0)
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
        m = ConcreteModel()
        e = m.e = Expression(expr=1.0)
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
        m = ConcreteModel()
        e = m.e = Expression(expr=3.0)
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
        m = ConcreteModel()
        e = m.e = Expression(expr=3.0)
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
        m = ConcreteModel()
        e = m.e = Expression(expr=3.0)
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
        m = ConcreteModel()
        e = m.e = Expression(expr=3.0)
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

if __name__ == "__main__":
    unittest.main()

