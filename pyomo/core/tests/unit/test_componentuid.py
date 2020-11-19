#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for ComponentUID
#
import pickle
import sys
from collections import namedtuple
from datetime import datetime
from six import StringIO, itervalues

import pyutilib.th as unittest
from pyomo.environ import (
    ConcreteModel, Block, Var, Set, Param, Constraint, Any, ComponentUID,
)
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.common.log import LoggingIntercept

_star = slice(None)

_Foo = namedtuple('_Foo', ['x','yy'])

class TestComponentUID(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        m = self.m
        m.a = Param()
        m.s = Set(initialize=[1,'2',3])
        m.b = Block(m.s, m.s)
        m.b[1,1].c = Block()
        m.b[1,'2'].c = Block()
        m.b[1,'2'].c.a = Param(m.s, initialize=3, mutable=True)

    def tearDown(self):
        self.m = None

    def test_genFromComponent_simple(self):
        cuid = ComponentUID(self.m.a)
        self.assertEqual(cuid._cids, (('a',tuple()),))

    def test_genFromComponent_nested(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(3,))) )

    def test_genFromComponent_indexed(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a)
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',())) )

    def test_genFromComponent_nameBuffer(self):
        buf = {}
        cuid = ComponentUID(self.m.b[1,'2'].c.a, cuid_buffer=buf)
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',())) )
        self.assertEqual(len(buf), 9)
        for s1 in self.m.s:
            for s2 in self.m.s:
                _id = id(self.m.b[s1,s2])
                self.assertIn(_id, buf)
                self.assertEqual(buf[_id], ('b',(s1,s2)))

    def test_genFromComponent_context(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a, context=self.m.b[1,'2'])
        self.assertEqual(
            cuid._cids,
            (('c',tuple()), ('a',())) )
        with self.assertRaisesRegex(
                ValueError,
                "Context 'b\[1,2\]' does not apply to component 's'"):
            ComponentUID(self.m.s, context=self.m.b[1,'2'])
        with self.assertRaisesRegex(
                ValueError,
                "Context is not allowed when initializing a ComponentUID "
                "object from a string type"):
            ComponentUID("b[1,2].c.a[2]", context=self.m.b[1,'2'])

    def test_parseFromString(self):
        cuid = ComponentUID('b[1,2].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('b',(1,2)), ('c',tuple()), ('a',(2,))) )

    def test_parseFromString_singleQuote(self):
        cuid = ComponentUID('b[1,\'2\'].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromString_doubleQuote(self):
        cuid = ComponentUID('b[1,\"2\"].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromString_typeID(self):
        cuid = ComponentUID('b[#1,$2].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromString_wildcard_1(self):
        cuid = ComponentUID('b[**].c.a[*]')
        self.assertEqual(
            cuid._cids,
            (('b',(Ellipsis,)), ('c',tuple()), ('a',(_star,))) )

    def test_parseFromString_wildcard_2(self):
        cuid = ComponentUID('b[*,*].c.a[*]')
        self.assertEqual(
            cuid._cids,
            (('b',(_star, _star)), ('c',tuple()), ('a',(_star,))) )

    def test_parseFromRepr1(self):
        cuid = ComponentUID('b:1,2.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('b',(1,2)), ('c',tuple()), ('a',(2,))) )

    def test_parseFromRepr1_singleQuote(self):
        cuid = ComponentUID('b:1,\'2\'.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromRepr1_doubleQuote(self):
        cuid = ComponentUID('b:1,\"2\".c.a:2')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromRepr1_typeID(self):
        cuid = ComponentUID('b:#1,$2.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('b',(1,'2')), ('c',tuple()), ('a',(2,))) )

    def test_parseFromRepr1_wildcard_1(self):
        cuid = ComponentUID('b:**.c.a:*')
        self.assertEqual(
            cuid._cids,
            (('b',(Ellipsis,)), ('c',tuple()), ('a',(_star,))) )

    def test_parseFromRepr1_wildcard_2(self):
        cuid = ComponentUID('b:*,*.c.a:*')
        self.assertEqual(
            cuid._cids,
            (('b',(_star, _star)), ('c',tuple()), ('a',(_star,))) )

    def test_parseFromRepr2_lexError(self):
        cuid = ComponentUID('') # Bogus instance to access parser
        with self.assertRaisesRegex(
                IOError, "ERROR: Token ':' Line 1 Column 1"):
            list(cuid._parse_cuid_v2(':'))
        with self.assertRaisesRegex(
                IOError, "ERROR: Token '\n].b:' Line 1 Column 3"):
            list(cuid._parse_cuid_v2('a[\n].b:'))

    def test_escapeChars(self):
        ref = r"b['a\n.b\\'].x"
        cuid = ComponentUID(ref)
        self.assertEqual(
            cuid._cids,
            (('b',('a\n.b\\',)), ('x',tuple())) )

        m = ConcreteModel()
        m.b = Block(['a\n.b\\'])
        m.b['a\n.b\\'].x = x = Var()
        self.assertTrue(cuid.matches(x))
        self.assertEqual(repr(ComponentUID(x)), ref)
        self.assertEqual(str(ComponentUID(x)), r"b['a\n.b\\'].x")

    def test_nonIntNumber(self):
        inf = float('inf')
        m = ConcreteModel()
        m.b = Block([inf, 'inf'])

        m.b[inf].x = x = Var()
        ref = r"b[inf].x"

        cuid = ComponentUID(x)
        self.assertEqual(
            cuid._cids,
            (('b',(inf,)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(repr(ComponentUID(x)), ref)
        self.assertEqual(str(ComponentUID(x)), ref)

        cuid = ComponentUID(ref)
        self.assertEqual(
            cuid._cids,
            (('b',(inf,)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(repr(ComponentUID(x)), ref)
        self.assertEqual(str(ComponentUID(x)), ref)

        ref = r"b:#inf.x"
        cuid = ComponentUID(ref)
        self.assertEqual(
            cuid._cids,
            (('b',(inf,)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(ComponentUID(x).get_repr(1), ref)
        self.assertEqual(str(ComponentUID(x)), r"b[inf].x")

        #
        m.b['inf'].x = x = Var()
        ref = r"b['inf'].x"
        #

        cuid = ComponentUID(x)
        self.assertEqual(
            cuid._cids,
            (('b',('inf',)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(repr(ComponentUID(x)), ref)
        self.assertEqual(str(ComponentUID(x)), ref)

        cuid = ComponentUID(ref)
        self.assertEqual(
            cuid._cids,
            (('b',('inf',)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(repr(ComponentUID(x)), ref)
        self.assertEqual(str(ComponentUID(x)), ref)

        ref = r"b:$inf.x"
        cuid = ComponentUID(ref)
        self.assertEqual(
            cuid._cids,
            (('b',('inf',)), ('x',tuple())) )

        self.assertTrue(cuid.matches(x))
        self.assertEqual(ComponentUID(x).get_repr(1), ref)
        self.assertEqual(str(ComponentUID(x)), r"b['inf'].x")


    def test_find_component_deprecated(self):
        ref = self.m.b[1,'2'].c.a[3]
        cuid = ComponentUID(ref)
        DEP_OUT = StringIO()
        with LoggingIntercept(DEP_OUT, 'pyomo.core'):
            self.assertTrue(cuid.find_component(self.m) is ref)
        self.assertIn('ComponentUID.find_component() is deprecated.',
                      DEP_OUT.getvalue())

    def test_find_explicit_exists(self):
        ref = self.m.b[1,'2'].c.a[3]
        cuid = ComponentUID(ref)
        self.assertTrue(cuid.find_component_on(self.m) is ref)

    def test_find_component_exists_1(self):
        ref = self.m.b[1,'2'].c.a
        cuid = ComponentUID(ref)
        self.assertTrue(cuid.find_component_on(self.m) is ref)

    def test_find_wildcard(self):
        cuid = ComponentUID('b:1,$2.c.a:*')
        comp = cuid.find_component_on(self.m)
        self.assertIs(comp.ctype, Param)
        cList = list(itervalues(comp))
        self.assertEqual(len(cList), 3)
        self.assertEqual(cList, list(self.m.b[1,'2'].c.a[:]))

        cuid = ComponentUID('b[*,*]')
        comp = cuid.find_component_on(self.m)
        self.assertIs(comp.ctype, Block)
        cList = list(itervalues(comp))
        self.assertEqual(len(cList), 9)
        self.assertEqual(cList, list(self.m.b.values()))

    def test_find_wildcard_partial_exists(self):
        # proper Reference: to ComponentData
        cuid = ComponentUID('b[*,*].c.a[**]')
        comp = cuid.find_component_on(self.m)
        self.assertIs(comp.ctype, Param)
        cList = list(itervalues(comp))
        self.assertEqual(len(cList), 3)
        self.assertEqual(cList, list(self.m.b[1,'2'].c.a[:]))

        # improper Reference: to IndexedComponent
        cuid = ComponentUID('b[*,*].c.a')
        comp = cuid.find_component_on(self.m)
        self.assertIs(comp.ctype, IndexedComponent)
        cList = list(itervalues(comp))
        self.assertEqual(len(cList), 1)
        self.assertIs(cList[0], self.m.b[1,'2'].c.a)

    def test_find_wildcard_not_exists(self):
        cuid = ComponentUID('b[*,*].c.x')
        self.assertIsNone(cuid.find_component_on(self.m))

    # def test_find_implicit_exists(self):
    #     cuid = ComponentUID('b:1,2.c.a:3')
    #     self.assertTrue(cuid.find_component_on(self.m) is
    #                     self.m.b[1,'2'].c.a[3])

    def test_find_implicit_notExists_1(self):
        cuid = ComponentUID('b:1,2.c.a:4')
        self.assertTrue(cuid.find_component_on(self.m) is None)

    def test_find_implicit_notExists_2(self):
        cuid = ComponentUID('b:1,1.c.a:3')
        self.assertTrue(cuid.find_component_on(self.m) is None)

    def test_find_explicit_notExists_1(self):
        cuid = ComponentUID('b:1,2.c.a:$3')
        self.assertTrue(cuid.find_component_on(self.m) is None)

    def test_find_explicit_notExists_2(self):
        cuid = ComponentUID('b:$1,2.c.a:3')
        self.assertTrue(cuid.find_component_on(self.m) is None)

    def test_printers_1(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        s = "b[1,'2'].c.a[3]"
        r1 = "b:#1,$2.c.a:#3"
        r2 = "b[1,'2'].c.a[3]"
        self.assertEqual(str(cuid), s)
        self.assertEqual(repr(cuid), r2)
        self.assertEqual(cuid.get_repr(1), r1)
        self.assertEqual(cuid.get_repr(2), r2)
        with self.assertRaisesRegex(
                ValueError, "Invalid repr version '3'; expected 1 or 2"):
            cuid.get_repr(3)

    def test_printers_2(self):
        cuid = ComponentUID('b:$1,2.c.a:#3')
        s = "b['1',2].c.a[3]"
        r1 = "b:$1,#2.c.a:#3"
        r2 = "b['1',2].c.a[3]"
        self.assertEqual(str(cuid), s)
        self.assertEqual(repr(cuid), r2)
        self.assertEqual(cuid.get_repr(1), r1)
        self.assertEqual(cuid.get_repr(2), r2)

    def test_printers_3(self):
        cuid = ComponentUID('b:**.c.a:*')
        s = 'b[**].c.a[*]'
        r1 = "b:**.c.a:*"
        r2 = "b[**].c.a[*]"
        self.assertEqual(str(cuid), s)
        self.assertEqual(repr(cuid), r2)
        self.assertEqual(cuid.get_repr(1), r1)
        self.assertEqual(cuid.get_repr(2), r2)

    def test_printers_4(self):
        cuid = ComponentUID('b:*,*.c.a:**')
        s = 'b[*,*].c.a[**]'
        r1 = "b:*,*.c.a:**"
        r2 = "b[*,*].c.a[**]"
        self.assertEqual(str(cuid), s)
        self.assertEqual(repr(cuid), r2)
        self.assertEqual(cuid.get_repr(1), r1)
        self.assertEqual(cuid.get_repr(2), r2)

    def test_matches_explicit(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    # def test_matches_implicit(self):
    #     cuid = ComponentUID('b:1,2.c.a:3')
    #     self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
    #     self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_explicit_1(self):
        cuid = ComponentUID('b:#1,$2.c.a:$3')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_explicit_2(self):
        cuid = ComponentUID('b:#1,#2.c.a:#3')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_wildcard_1(self):
        cuid = ComponentUID('b:**.c.a:*')
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_wildcard_2(self):
        cuid = ComponentUID('b:*,*.c.a:**')
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_wildcard_3(self):
        cuid = ComponentUID('b:*,*.c.a:*,*')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_mismatch_name(self):
        cuid = ComponentUID('b:*,*.d')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))

    def test_matches_mismatch_1(self):
        cuid = ComponentUID('b:*,*.c.a:*')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))

    def test_matches_mismatch_2(self):
        cuid = ComponentUID('b:*,*.c')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_mismatch_3(self):
        cuid = ComponentUID('b:*,*,*.c.a:*')
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_ellipsis1(self):
        cuid = ComponentUID('b[**,1].c')
        self.assertTrue(cuid.matches(self.m.b[1,1].c))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))

    def test_matches_ellipsis2(self):
        cuid = ComponentUID('b[**,1,1].c')
        self.assertTrue(cuid.matches(self.m.b[1,1].c))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))

    def test_matches_ellipsis3(self):
        cuid = ComponentUID('b[**,1,1,3].c')
        self.assertFalse(cuid.matches(self.m.b[1,1].c))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c))

    def test_matches_ellipsis4(self):
        cuid = ComponentUID('b[**,1,*].c')
        self.assertTrue(cuid.matches(self.m.b[1,1].c))
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c))

    def test_list_components_dne_1(self):
        cuid = ComponentUID('b:*,*,*.c.a:*')
        ref = []
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_dne_2(self):
        cuid = ComponentUID('b:*,*.c:#1.a:*')
        ref = []
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_scalar(self):
        cuid = ComponentUID('b:1,$2.c.a:3')
        ref = [ str(ComponentUID(self.m.b[1,'2'].c.a[3])) ]
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_wildcard_1(self):
        cuid = ComponentUID('b:**.c.a:3')
        ref = [ str(ComponentUID(self.m.b[1,'2'].c.a[3])) ]
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_wildcard_2(self):
        cuid = ComponentUID('b:*,*.c.a:*')
        ref = [ str(ComponentUID(self.m.b[1,'2'].c.a[1])),
                str(ComponentUID(self.m.b[1,'2'].c.a['2'])),
                str(ComponentUID(self.m.b[1,'2'].c.a[3])) ]
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_wildcard_3(self):
        cuid = ComponentUID('b:1,*.c')
        ref = [ str(ComponentUID(self.m.b[1,1].c)),
                str(ComponentUID(self.m.b[1,'2'].c)) ]
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_list_components_wildcard_4(self):
        cuid = ComponentUID('b:1,*')
        ref = [ str(ComponentUID(self.m.b[1,1])),
                str(ComponentUID(self.m.b[1,'2'])),
                str(ComponentUID(self.m.b[1,3])) ]
        cList = [ str(ComponentUID(x)) for x in cuid.list_components(self.m) ]
        self.assertEqual(sorted(cList), sorted(ref))

    def test_in_container(self):
        a = ComponentUID('foo.bar[*]')
        b = ComponentUID('baz')
        c = ComponentUID('baz.bar')
        D = {a: 1, b: 2}

        self.assertTrue(  a in D )
        self.assertTrue(  b in D )
        self.assertFalse( c in D )

        # Verify that hashing is not being done by id()
        self.assertTrue( ComponentUID('foo.bar[*]') in D )
        self.assertTrue( ComponentUID('baz') in D )

    def test_comparisons(self):
        a = ComponentUID('foo.x[*]')
        b = ComponentUID('baz')

        self.assertFalse( a <  b )
        self.assertFalse( a <= b )
        self.assertTrue ( a >  b )
        self.assertTrue ( a >= b )
        self.assertFalse( a == b )
        self.assertTrue ( a != b )

        self.assertTrue ( b <  a )
        self.assertTrue ( b <= a )
        self.assertFalse( b >  a )
        self.assertFalse( b >= a )
        self.assertFalse( b == a )
        self.assertTrue ( b != a )

        self.assertFalse( a <  ComponentUID('baz') )
        self.assertFalse( a <= ComponentUID('baz') )
        self.assertTrue ( a >  ComponentUID('baz') )
        self.assertTrue ( a >= ComponentUID('baz') )
        self.assertFalse( a == ComponentUID('baz') )
        self.assertTrue ( a != ComponentUID('baz') )

        self.assertTrue ( ComponentUID('baz') <  a )
        self.assertTrue ( ComponentUID('baz') <= a )
        self.assertFalse( ComponentUID('baz') >  a )
        self.assertFalse( ComponentUID('baz') >= a )
        self.assertFalse( ComponentUID('baz') == a )
        self.assertTrue ( ComponentUID('baz') != a )

        self.assertFalse( b <  b )
        self.assertTrue ( b <= b )
        self.assertFalse( b >  b )
        self.assertTrue ( b >= b )
        self.assertTrue ( b == b )
        self.assertFalse( b != b )

        self.assertFalse( ComponentUID('baz') < b )
        self.assertTrue ( ComponentUID('baz') <= b )
        self.assertFalse( ComponentUID('baz') > b )
        self.assertTrue ( ComponentUID('baz') >= b )
        self.assertTrue ( ComponentUID('baz') == b )
        self.assertFalse( ComponentUID('baz') != b )

    def test_comparisons_lt(self):
        a = ComponentUID('foo.x[*]')
        a1 = ComponentUID('foo.x[1]')
        a2 = ComponentUID('foo.x[2]')
        aa = ComponentUID("foo.x['a']")
        a11 = ComponentUID('foo.x[1,1]')
        ae = ComponentUID('foo.x[**]')
        self.assertTrue( a < ae )
        self.assertTrue( a1 < ae )
        self.assertTrue( a1 < a )
        self.assertTrue( a1 < a2 )
        self.assertTrue( a1 < aa )
        self.assertTrue( a1 < a11 )
        self.assertTrue( a11 < a2 )
        self.assertFalse( ae < a )
        self.assertFalse( ae < a1 )
        self.assertFalse( a < a1 )
        self.assertFalse( a2 < a1 )
        self.assertFalse( aa < a1 )
        self.assertFalse( a11 < a1 )
        self.assertFalse( a2 < a11 )

        x = ComponentUID('foo.x')
        xy = ComponentUID('foo.x.y')
        self.assertTrue( x < xy )
        self.assertFalse( xy < x )

        with self.assertRaisesRegex(
                TypeError, "'<' not supported between instances of "
                "'ComponentUID' and 'int'"):
            a < 5

    def test_comparisons_eq(self):
        a = ComponentUID('foo.x[*]')
        a1 = ComponentUID('foo.x[1]')
        b = ComponentUID('foo.x[*]')
        self.assertEqual(a, b)
        self.assertNotEqual(a, a1)
        self.assertNotEqual(a, 5)


    def test_generate_cuid_string_map(self):
        model = Block(concrete=True)
        model.x = Var()
        model.y = Var([1,2])
        model.V = Var([('a','b'),(1,'2'),(3,4)])
        model.b = Block(concrete=True)
        model.b.z = Var([1,'2'])
        setattr(model.b, '.H', Var(['a',2]))
        model.B = Block(['a',2], concrete=True)
        setattr(model.B['a'],'.k', Var())
        model.B[2].b = Block()
        model.B[2].b.x = Var()
        model.add_component('c tuple', Constraint(Any))
        model.component('c tuple')[(1,)] = model.x >= 0

        cuids = (
            ComponentUID.generate_cuid_string_map(model, repr_version=1),
            ComponentUID.generate_cuid_string_map(model),
        )
        self.assertEqual(len(cuids[0]), 29)
        self.assertEqual(len(cuids[1]), 29)
        for obj in [model,
                    model.x,
                    model.y,
                    model.y_index,
                    model.y[1],
                    model.y[2],
                    model.V,
                    model.V_index,
                    model.V['a','b'],
                    model.V[1,'2'],
                    model.V[3,4],
                    model.b,
                    model.b.z,
                    model.b.z_index,
                    model.b.z[1],
                    model.b.z['2'],
                    getattr(model.b, '.H'),
                    getattr(model.b, '.H_index'),
                    getattr(model.b, '.H')['a'],
                    getattr(model.b, '.H')[2],
                    model.B,
                    model.B_index,
                    model.B['a'],
                    getattr(model.B['a'],'.k'),
                    model.B[2],
                    model.B[2].b,
                    model.B[2].b.x,
                    model.component('c tuple')[(1,)]]:
            self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
            self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])

        cuids = (
            ComponentUID.generate_cuid_string_map(model, descend_into=False,
                                                  repr_version=1),
            ComponentUID.generate_cuid_string_map(model, descend_into=False),
        )
        self.assertEqual(len(cuids[0]), 18)
        self.assertEqual(len(cuids[1]), 18)
        for obj in [model,
                    model.x,
                    model.y,
                    model.y_index,
                    model.y[1],
                    model.y[2],
                    model.V,
                    model.V_index,
                    model.V['a','b'],
                    model.V[1,'2'],
                    model.V[3,4],
                    model.b,
                    model.B,
                    model.B_index,
                    model.B['a'],
                    model.B[2],
                    model.component('c tuple')[(1,)]]:
            self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
            self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])

        cuids = (
            ComponentUID.generate_cuid_string_map(model, ctype=Var,
                                                  repr_version=1),
            ComponentUID.generate_cuid_string_map(model, ctype=Var),
        )
        self.assertEqual(len(cuids[0]), 22)
        self.assertEqual(len(cuids[1]), 22)
        for obj in [model,
                    model.x,
                    model.y,
                    model.y[1],
                    model.y[2],
                    model.V,
                    model.V['a','b'],
                    model.V[1,'2'],
                    model.V[3,4],
                    model.b,
                    model.b.z,
                    model.b.z[1],
                    model.b.z['2'],
                    getattr(model.b, '.H'),
                    getattr(model.b, '.H')['a'],
                    getattr(model.b, '.H')[2],
                    model.B,
                    model.B['a'],
                    getattr(model.B['a'],'.k'),
                    model.B[2],
                    model.B[2].b,
                    model.B[2].b.x]:
            self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
            self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])

        cuids = (
            ComponentUID.generate_cuid_string_map(
                model, ctype=Var, descend_into=False, repr_version=1),
            ComponentUID.generate_cuid_string_map(
                model, ctype=Var, descend_into=False),
        )
        self.assertEqual(len(cuids[0]), 9)
        self.assertEqual(len(cuids[1]), 9)
        for obj in [model,
                    model.x,
                    model.y,
                    model.y[1],
                    model.y[2],
                    model.V,
                    model.V['a','b'],
                    model.V[1,'2'],
                    model.V[3,4]]:
            self.assertEqual(ComponentUID(obj).get_repr(1), cuids[0][obj])
            self.assertEqual(repr(ComponentUID(obj)), cuids[1][obj])

    def test_pickle(self):
        a = ComponentUID("b[1,'2'].c")
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

        a = ComponentUID("b[1,*].c")
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

        a = ComponentUID("b[**,*].c")
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_findComponentOn_nestedTuples(self):
        # Tests for #1069
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(Any)
        m.c[0] = m.x >= 0
        m.c[(1,)] = m.x >= 1
        m.c[(2,)] = m.x >= 2
        m.c[2] = m.x >= 3
        self.assertIs(ComponentUID(m.c[0]).find_component_on(m), m.c[0])
        self.assertIs(ComponentUID('c[0]').find_component_on(m), m.c[0])
        self.assertIsNone(ComponentUID('c[(0,)]').find_component_on(m))
        self.assertIs(ComponentUID(m.c[(1,)]).find_component_on(m), m.c[(1,)])
        self.assertIs(ComponentUID('c[(1,)]').find_component_on(m), m.c[(1,)])
        self.assertIsNone(ComponentUID('c[1]').find_component_on(m))
        self.assertIs(ComponentUID('c[(2,)]').find_component_on(m), m.c[(2,)])
        self.assertIs(ComponentUID('c[2]').find_component_on(m), m.c[2])
        self.assertEqual(len(m.c), 4)

        self.assertEqual(repr(ComponentUID(m.c[0])), "c[0]")
        self.assertEqual(repr(ComponentUID(m.c[(1,)])), "c[(1,)]")
        self.assertEqual(str(ComponentUID(m.c[0])), "c[0]")
        self.assertEqual(str(ComponentUID(m.c[(1,)])), "c[(1,)]")

        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint([0,1])
        m.c[0] = m.x >= 0
        m.c[(1,)] = m.x >= 1
        self.assertIs(ComponentUID(m.c[0]).find_component_on(m), m.c[0])
        self.assertIs(ComponentUID(m.c[(0,)]).find_component_on(m), m.c[0])
        self.assertIs(ComponentUID('c[0]').find_component_on(m), m.c[0])
        self.assertIs(ComponentUID('c[(0,)]').find_component_on(m), m.c[0])
        self.assertIs(ComponentUID(m.c[1]).find_component_on(m), m.c[1])
        self.assertIs(ComponentUID(m.c[(1,)]).find_component_on(m), m.c[1])
        self.assertIs(ComponentUID('c[(1,)]').find_component_on(m), m.c[1])
        self.assertIs(ComponentUID('c[1]').find_component_on(m), m.c[1])
        self.assertEqual(len(m.c), 2)

        m = ConcreteModel()
        m.b = Block(Any)
        m.b[0].c = Block(Any)
        m.b[0].c[0].x = Var()
        m.b[(1,)].c = Block(Any)
        m.b[(1,)].c[(1,)].x = Var()
        ref = m.b[0].c[0].x
        self.assertIs(ComponentUID(ref).find_component_on(m), ref)
        ref = 'm.b[0].c[(0,)].x'
        self.assertIsNone(ComponentUID(ref).find_component_on(m))
        ref = m.b[(1,)].c[(1,)].x
        self.assertIs(ComponentUID(ref).find_component_on(m), ref)
        ref = 'm.b[(1,)].c[1].x'
        self.assertIsNone(ComponentUID(ref).find_component_on(m))

        buf = {}
        ref = m.b[0].c[0].x
        self.assertIs(
            ComponentUID(ref, cuid_buffer=buf).find_component_on(m), ref)
        self.assertEqual(len(buf), 3)
        ref = 'm.b[0].c[(0,)].x'
        self.assertIsNone(
            ComponentUID(ref, cuid_buffer=buf).find_component_on(m))
        self.assertEqual(len(buf), 3)
        ref = m.b[(1,)].c[(1,)].x
        self.assertIs(
            ComponentUID(ref, cuid_buffer=buf).find_component_on(m), ref)
        self.assertEqual(len(buf), 4)
        ref = 'm.b[(1,)].c[1].x'
        self.assertIsNone(
            ComponentUID(ref, cuid_buffer=buf).find_component_on(m))
        self.assertEqual(len(buf), 4)

    def test_pickle_index(self):
        m = ConcreteModel()
        m.b = Block(Any)

        idx = "|b'foo'"
        m.b[idx].x = Var()
        cuid = ComponentUID(m.b[idx].x)
        self.assertEqual(str(cuid), 'b["|b\'foo\'"].x')
        self.assertIs(cuid.find_component_on(m), m.b[idx].x)
        tmp = ComponentUID(str(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)
        tmp = pickle.loads(pickle.dumps(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)

        idx = _Foo(1,'a')
        m.b[idx].x = Var()
        cuid = ComponentUID(m.b[idx].x)
        # Note that the pickle string for namedtuple changes between
        # Python 2, 3, and pypy, so we will just check the non-pickle
        # data part
        self.assertRegex(str(cuid), r"^b\[\|b?(['\"]).*\.\1\]\.x$")

        self.assertIs(cuid.find_component_on(m), m.b[idx].x)
        tmp = ComponentUID(str(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)
        tmp = pickle.loads(pickle.dumps(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)

        idx = datetime(1,2,3)
        m.b[idx].x = Var()
        cuid = ComponentUID(m.b[idx].x)
        # Note that the pickle string for namedtuple changes between
        # Python 2, 3, and pypy, so we will just check the non-pickle
        # data part
        self.assertRegex(str(cuid), r"^b\[\|b?(['\"]).*\.\1\]\.x$")
        self.assertIs(cuid.find_component_on(m), m.b[idx].x)
        tmp = ComponentUID(str(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)
        tmp = pickle.loads(pickle.dumps(cuid))
        self.assertIsNot(cuid, tmp)
        self.assertEqual(cuid, tmp)
        self.assertIs(tmp.find_component_on(m), m.b[idx].x)

        self.assertEqual(len(m.b), 3)

    def test_deprecated_ComponentUID_location(self):
        import pyomo.core.base.component as comp
        self.assertNotIn('ComponentUID', dir(comp))

        warning = "DEPRECATED: the 'ComponentUID' class has been moved to " \
                  "'pyomo.core.base.componentuid.ComponentUID'"
        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.core'):
            from pyomo.core.base.component import ComponentUID \
                as old_ComponentUID
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))

        self.assertIs(old_ComponentUID, ComponentUID)
        self.assertIs(old_ComponentUID, ComponentUID)

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.core'):
            self.assertIs(comp.ComponentUID, ComponentUID)
        self.assertEqual("", OUT.getvalue())

if __name__ == "__main__":
    unittest.main()
