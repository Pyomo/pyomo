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
# Unit Tests for components
#
from six import StringIO
import pyutilib.th as unittest

from pyomo.common import DeveloperError
import pyomo.core.base._pyomo
from pyomo.environ import (
    ConcreteModel, Component, Block, Var, Set, Param,
)

class TestComponent(unittest.TestCase):

    def test_construct_component_throws_exception(self):
        with self.assertRaisesRegexp(
                DeveloperError,
                "Must specify a component type for class Component"):
            Component()

    def test_getname(self):
        m = ConcreteModel()
        m.b = Block([1,2])
        m.b[2].c = Var([1,2],[3,4])

        self.assertEqual(m.getname(fully_qualified=True), "unknown")
        self.assertEqual(m.b.getname(fully_qualified=True), "b")
        self.assertEqual(m.b[1].getname(fully_qualified=True), "b[1]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
                         "b[2].c[2,4]")
        self.assertEqual(m.getname(fully_qualified=False), "unknown")
        self.assertEqual(m.b.getname(fully_qualified=False), "b")
        self.assertEqual(m.b[1].getname(fully_qualified=False), "b[1]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=False),
                         "c[2,4]")

        cache = {}
        self.assertEqual(
            m.b[2].c[2,4].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[2,4]")
        self.assertEqual(len(cache), 8)
        self.assertIn(id(m.b[2].c[2,4]), cache)
        self.assertIn(id(m.b[2].c[1,3]), cache)
        self.assertIn(id(m.b[2].c), cache)
        self.assertIn(id(m.b[2]), cache)
        self.assertIn(id(m.b[1]), cache)
        self.assertIn(id(m.b), cache)
        self.assertNotIn(id(m), cache)
        self.assertEqual(
            m.b[2].c[1,3].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[1,3]")

        m.b[2]._component = None
        self.assertEqual(m.b[2].getname(fully_qualified=True),
                         "[Unattached _BlockData]")
        # I think that getname() should do this:
        #self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
        #                 "[Unattached _BlockData].c[2,4]")
        # but it doesn't match current behavior.  I will file a PEP to
        # propose changing the behavior later and proceed to test
        # current behavior.
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=True),
                         "c[2,4]")

        self.assertEqual(m.b[2].getname(fully_qualified=False),
                         "[Unattached _BlockData]")
        self.assertEqual(m.b[2].c[2,4].getname(fully_qualified=False),
                         "c[2,4]")

        # Cached names still work...
        self.assertEqual(
            m.b[2].getname(fully_qualified=True, name_buffer=cache),
            "b[2]")
        self.assertEqual(
            m.b[2].c[1,3].getname(fully_qualified=True, name_buffer=cache),
            "b[2].c[1,3]")

    def test_component_data_pprint(self):
        m = ConcreteModel()
        m.a = Set(initialize=[1, 2, 3], ordered=True)
        m.x = Var(m.a)
        stream = StringIO()
        m.x[2].pprint(ostream=stream)
        correct_s = '{Member of x} : Size=3, Index=a\n    ' \
                    'Key : Lower : Value : Upper : Fixed : Stale : Domain\n      ' \
                    '2 :  None :  None :  None : False :  True :  Reals\n'
        self.assertEqual(correct_s, stream.getvalue())


class _TestComponentUID(unittest.TestCase):

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
        self.assertEqual(cuid._cids, (('a',tuple(),''),))

    def test_genFromComponent_nested(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        self.assertEqual(
            cuid._cids,
            (('a',(3,),'#'), ('c',tuple(),''), ('b',(1,'2'),'#$')) )

    def test_genFromComponent_indexed(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a)
        self.assertEqual(
            cuid._cids,
            (('a','**',None), ('c',tuple(),''), ('b',(1,'2'),'#$')) )

    def test_parseFromString(self):
        cuid = ComponentUID('b[1,2].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'..')) )

    def test_parseFromString_singleQuote(self):
        cuid = ComponentUID('b[1,\'2\'].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'.$')) )

    def test_parseFromString_doubleQuote(self):
        cuid = ComponentUID('b[1,\"2\"].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'.$')) )

    def test_parseFromString_typeID(self):
        cuid = ComponentUID('b[#1,$2].c.a[2]')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',(1,'2'),'#$')) )

    def test_parseFromString_wildcard_1(self):
        cuid = ComponentUID('b[**].c.a[*]')
        self.assertEqual(
            cuid._cids,
            (('a',('',),'*'), ('c',tuple(),''), ('b','**',None)) )

    def test_parseFromString_wildcard_2(self):
        cuid = ComponentUID('b[*,*].c.a[*]')
        self.assertEqual(
            cuid._cids,
            (('a',('',),'*'), ('c',tuple(),''), ('b',('',''),'**')) )

    def test_parseFromRepr(self):
        cuid = ComponentUID('b:1,2.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'..')) )

    def test_parseFromRepr_singleQuote(self):
        cuid = ComponentUID('b:1,\'2\'.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'.$')) )

    def test_parseFromRepr_doubleQuote(self):
        cuid = ComponentUID('b:1,\"2\".c.a:2')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',('1','2'),'.$')) )

    def test_parseFromRepr_typeID(self):
        cuid = ComponentUID('b:#1,$2.c.a:2')
        self.assertEqual(
            cuid._cids,
            (('a',('2',),'.'), ('c',tuple(),''), ('b',(1,'2'),'#$')) )

    def test_parseFromRepr_wildcard_1(self):
        cuid = ComponentUID('b:**.c.a:*')
        self.assertEqual(
            cuid._cids,
            (('a',('',),'*'), ('c',tuple(),''), ('b','**',None)) )

    def test_parseFromRepr_wildcard_2(self):
        cuid = ComponentUID('b:*,*.c.a:*')
        self.assertEqual(
            cuid._cids,
            (('a',('',),'*'), ('c',tuple(),''), ('b',('',''),'**')) )

    def test_find_explicit_exists(self):
        ref = self.m.b[1,'2'].c.a[3]
        cuid = ComponentUID(ref)
        self.assertTrue(cuid.find_component(self.m) is ref)

    def test_find_wildcard_exists_1(self):
        ref = self.m.b[1,'2'].c.a
        cuid = ComponentUID(ref)
        self.assertTrue(cuid.find_component(self.m) is ref)

    def test_find_wildcard_exists_2(self):
        cuid = ComponentUID('b:1,2.c.a:*')
        self.assertTrue(cuid.find_component(self.m) is self.m.b[1,'2'].c.a)

    def test_find_wildcard_exists_3(self):
        cuid = ComponentUID('b[*,*]')
        self.assertTrue(cuid.find_component(self.m) is self.m.b)

    def test_find_implicit_exists(self):
        cuid = ComponentUID('b:1,2.c.a:3')
        self.assertTrue(cuid.find_component(self.m) is self.m.b[1,'2'].c.a[3])

    def test_find_implicit_notExists_1(self):
        cuid = ComponentUID('b:1,2.c.a:4')
        self.assertTrue(cuid.find_component(self.m) is None)

    def test_find_implicit_notExists_2(self):
        cuid = ComponentUID('b:1,1.c.a:3')
        self.assertTrue(cuid.find_component(self.m) is None)

    def test_find_explicit_notExists_1(self):
        cuid = ComponentUID('b:1,2.c.a:$3')
        self.assertTrue(cuid.find_component(self.m) is None)

    def test_find_explicit_notExists_2(self):
        cuid = ComponentUID('b:$1,2.c.a:3')
        self.assertTrue(cuid.find_component(self.m) is None)

    def test_printers_1(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        s = 'b[1,2].c.a[3]'
        r = 'b:#1,$2.c.a:#3'
        self.assertEqual(repr(cuid), r)
        self.assertEqual(str(cuid), s)

    def test_printers_2(self):
        cuid = ComponentUID('b:$1,2.c.a:#3')
        s = 'b[1,2].c.a[3]'
        r = 'b:$1,2.c.a:#3'
        self.assertEqual(repr(cuid), r)
        self.assertEqual(str(cuid), s)

    def test_printers_3(self):
        cuid = ComponentUID('b:**.c.a:*')
        s = 'b[**].c.a[*]'
        r = 'b:**.c.a:*'
        self.assertEqual(repr(cuid), r)
        self.assertEqual(str(cuid), s)

    def test_printers_4(self):
        cuid = ComponentUID('b:*,*.c.a:**')
        s = 'b[*,*].c.a[**]'
        r = 'b:*,*.c.a:**'
        self.assertEqual(repr(cuid), r)
        self.assertEqual(str(cuid), s)

    def test_matches_explicit(self):
        cuid = ComponentUID(self.m.b[1,'2'].c.a[3])
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

    def test_matches_implicit(self):
        cuid = ComponentUID('b:1,2.c.a:3')
        self.assertTrue(cuid.matches(self.m.b[1,'2'].c.a[3]))
        self.assertFalse(cuid.matches(self.m.b[1,'2'].c.a['2']))

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
        cuid = ComponentUID('b:1,2.c.a:3')
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

    def test_generate_cuid_names(self):
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

        cuids = generate_cuid_names(model)
        self.assertEqual(len(cuids), 26)
        for obj in [model.x,
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
                    model.B[2].b.x]:
            assert repr(ComponentUID(obj)) == cuids[obj]
            del cuids[obj]
        self.assertEqual(len(cuids), 0)

        cuids = generate_cuid_names(model,
                                    descend_into=False)
        self.assertEqual(len(cuids), 15)
        for obj in [model.x,
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
                    model.B[2]]:
            assert repr(ComponentUID(obj)) == cuids[obj]
            del cuids[obj]
        self.assertEqual(len(cuids), 0)

        cuids = generate_cuid_names(model,
                                    ctype=Var)
        self.assertEqual(len(cuids), 21)
        for obj in [model.x,
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
            assert repr(ComponentUID(obj)) == cuids[obj]
            del cuids[obj]
        self.assertEqual(len(cuids), 0)

        cuids = generate_cuid_names(model,
                                    ctype=Var,
                                    descend_into=False)
        self.assertEqual(len(cuids), 8)
        for obj in [model.x,
                    model.y,
                    model.y[1],
                    model.y[2],
                    model.V,
                    model.V['a','b'],
                    model.V[1,'2'],
                    model.V[3,4]]:
            assert repr(ComponentUID(obj)) == cuids[obj]
            del cuids[obj]
        self.assertEqual(len(cuids), 0)

    def _slice_model(self):
        m = ConcreteModel()

        m.d1_1 = Set(initialize=[1,2,3])
        m.d1_2 = Set(initialize=['a','b','c'])
        m.d1_3 = Set(initialize=[1.1,1.2,1.3])
        m.d2 = Set(initialize=[('a',1), ('b',2)])
        m.dn = Set(initialize=[('c',3), ('d',4,5)], dimen=None)

        @m.Block()
        def b(b):

            b.b = Block()
            
            @b.Block(m.d1_1)
            def b1(b1, i):
                b1.v = Var()
                b1.v1 = Var(m.d1_3)
                b1.v2 = Var(m.d1_1, m.d1_2)
                b1.vn = Var(m.dn, m.d1_2)

            @b.Block(m.d1_1, m.d1_2)
            def b2(b2, i, j):
                b2.v = Var()
                b2.v1 = Var(m.d1_3)
                b2.v2 = Var(m.d1_1, m.d1_2)
                b2.vn = Var(m.d1_1, m.dn, m.d1_2)

            @b.Block(m.d1_3, m.d2)
            def b3(b3, i, j, k):
                b3.v = Var()
                b3.v1 = Var(m.d1_3)
                b3.v2 = Var(m.d1_1, m.d1_2)
                b3.vn = Var(m.d1_1, m.dn, m.d2)

            # Don't think I can define a dim-None Block with
            # a rule unless normalize_index.flatten is False.
            b.bn = Block(m.d1_2, m.dn, m.d2)
            # NOTE: These blocks are only defined for 'a', ('a',1)
            # in the first and last "subsets"
            b.bn['a','c',3,'a',1].v = Var()
            b.bn['a','c',3,'a',1].v1 = Var(m.d1_3)
            b.bn['a','c',3,'a',1].v2 = Var(m.d1_1, m.d1_2)
            b.bn['a','c',3,'a',1].vn = Var(m.d1_1, m.dn, m.d2)
            b.bn['a','d',4,5,'a',1].v = Var()
            b.bn['a','d',4,5,'a',1].v1 = Var(m.d1_3)
            b.bn['a','d',4,5,'a',1].v2 = Var(m.d1_1, m.d1_2)
            b.bn['a','d',4,5,'a',1].vn = Var(m.d1_1, m.dn, m.d2)

        return m

    def assertListSameComponents(self, m, cuid1, cuid2):
        self.assertTrue(cuid1.list_components(m))
        self.assertEqual(
                len(list(cuid1.list_components(m))),
                len(list(cuid2.list_components(m)))
                )
        for c1, c2 in zip(
                cuid1.list_components(m),
                cuid2.list_components(m),
                ):
            self.assertIs(c1, c2)

    def test_cuid_from_slice_1(self):
        m = self._slice_model()

        _slice = m.b[:]
        cuid_str = ComponentUID('b[*]')
        cuid = ComponentUID(_slice)
        self.assertEqual(cuid, cuid_str)

        _slice = m.b.b1[:]
        cuid_str = ComponentUID('b.b1[*]')
        cuid = ComponentUID(_slice)
        self.assertEqual(cuid, cuid_str)

        _slice = m.b.b1[...]
        cuid_str = ComponentUID('b.b1[**]')
        cuid = ComponentUID(_slice)
        self.assertEqual(cuid, cuid_str)

        _slice = m.b.b2[:,'a']
        cuid_str = ComponentUID('b.b2[*,a]')
        cuid = ComponentUID(_slice)
        self.assertEqual(str(cuid), str(cuid_str))
        # ^ A comparison between CUIDs is not appropriate here.
        # When constructed from a string, the "type char" of a
        # fixed index is set to '.', which I believe cannot be
        # reproduced by constructing with a component.
        #
        # Because asserting equality of the string representations
        # is not as strong as I'd like, I check that the two CUIDs
        # list the same components from the test model.
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[...]
        cuid_str = ComponentUID('b.b2[**]')
        cuid = ComponentUID(_slice)
        self.assertEqual(cuid, cuid_str)

        _slice = m.b.b3[1.1,:,2]
#        cuid_str = ComponentUID('b.b3[1.1,*,2]')
        cuid = ComponentUID(_slice)
#        self.assertEqual(str(cuid), str(cuid_str))
# ComponentUIDs constructed from strings do not behave well
# when decimal indices are present.
        components = [m.b.b3[1.1,'b',2]]
        self.assertTrue(cuid.list_components(m))
        self.assertEqual(
                len(list(cuid.list_components(m))),
                len(components),
                )
        for c1, c2 in zip(
                cuid.list_components(m),
                components,
                ):
            self.assertIs(c1, c2)

        _slice = m.b.b3[:,:,'b']
        cuid_str = ComponentUID('b.b3[*,*,b]')
        cuid = ComponentUID(_slice)
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        with self.assertRaises(NotImplementedError) as err:
            # This is not supported due to a limitation with CUIDs
            _slice = m.b.b3[1.1,...]
            cuid_str = 'b.b3[1.1,**]' 
            # What is the correct CUID representation for this kind of slice?
            cuid = ComponentUID(_slice)
            self.assertIn('fixed', str(err).lower())
            self.assertIn('ellipsis', str(err).lower())

        _slice = m.b.b3[...]
        cuid_str = 'b.b3[**]'
        cuid = ComponentUID(_slice)
        self.assertEqual(str(cuid), cuid_str)

        _slice = m.b.bn['a',:,:,'a',1]
        cuid_str = ComponentUID('b.bn[a,*,*,a,1]')
        cuid = ComponentUID(_slice)
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn['a','c',3,:,:]
        cuid_str = ComponentUID('b.bn[a,c,3,*,*]')
        cuid = ComponentUID(_slice)
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn[...]
        cuid_str = ComponentUID('b.bn[**]')
        cuid = ComponentUID(_slice)
        self.assertEqual(cuid, cuid_str)
        # No room for type interpretation here, so CUIDs can be compared
        # directly.

    def test_cuid_from_slice_2(self):
        m = self._slice_model()

        _slice = m.b[:].b
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b[*].b')
        self.assertEqual(cuid, cuid_str)

        _slice = m.b[:].b1[:].v
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b[*].b1[*].v')
        self.assertEqual(cuid, cuid_str)

        _slice = m.b.b2[2,:].v
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[2,*].v')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[2,:].v1[:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[2,*].v1[*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[2,:].v1[1.1]
        cuid = ComponentUID(_slice)
#        cuid_str = ComponentUID('b.b2[2,*].v1[1.1]')
#        self.assertEqual(str(cuid), str(cuid_str))
# ComponentUIDs constructed from strings don't work well
# with decimal indices.
        _slice = m.b.b2[2,:].v1[1.1]
        components = list(_slice)
        self.assertTrue(cuid.list_components(m))
        self.assertEqual(
                len(list(cuid.list_components(m))),
                len(components),
                )
        for c1, c2 in zip(
                cuid.list_components(m),
                components,
                ):
            self.assertIs(c1, c2)

        _slice = m.b.b2[2,:].vn[1,...,:,'b']
        with self.assertRaisesRegex(NotImplementedError,
                '.*Fixed.*ellipsis.*'):
            cuid = ComponentUID(_slice)

        _slice = m.b.b2[2,:].vn[...,'b']
        with self.assertRaisesRegex(NotImplementedError,
                '.*Fixed.*ellipsis.*'):
            cuid = ComponentUID(_slice)

        _slice = m.b.b2[2,:].vn[...,...]
        with self.assertRaisesRegex(NotImplementedError,
                '.*Multiple ellipses.*'):
            cuid = ComponentUID(_slice)

        _slice = m.b.b2[2,:].vn[...]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[2,*].vn[**]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[...].v2[:,'a']
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[**].v2[*,a]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b3[:,'a',:].v1
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b3[*,a,*].v1')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b3[:,'a',:].v2[1,'a']
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b3[*,a,*].v2[1,a]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b3[:,'a',:].v2[1,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b3[*,a,*].v2[1,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b3[:,'a',:].vn[1,:,:,'a',1]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b3[*,a,*].vn[1,*,*,a,1]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn['a','c',3,:,:].vn[1,:,3,'a',:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.bn[a,c,3,*,*].vn[1,*,3,a,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn[...].vn[1,:,3,'a',:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.bn[**].vn[1,*,3,a,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn[...].vn
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.bn[**].vn')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.bn[...].vn[...]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.bn[**].vn[**]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

    def test_cuid_from_slice_with_call(self):
        m = self._slice_model()

        _slice = m.b.component('b2')[:,'a'].v2[1,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[*,a].v2[1,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        # This works as find_component is not in the 
        # _call_stack of the slice.
        _slice = m.b.find_component('b2')[:,'a'].v2[1,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[*,a].v2[1,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b[:].component('b2')
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b[*].b2')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b[:].component('b2','b1')
        with self.assertRaisesRegex(NotImplementedError,
                '.*multiple arguments.*'):
            cuid = ComponentUID(_slice)

#        _slice = m.b[:].bad_call('b2')[:,'a']
#        with self.assertRaisesRegex(NotImplementedError,
#                'any method other than `component`'):
#            cuid = ComponentUID(_slice)
# Unclear how I should test this error. Would need a slice object with
# a call to attribute other than component in the call stack, but any
# call to another attribute will get iterated over immediately.
#
# Also unclear how I should test for the proper exception if set/del
# calls are (somehow) present in the call stack.

        _slice = m.b[:].component('b2', kwd=None)
        with self.assertRaisesRegex(NotImplementedError,
                '.*call that contains keywords.*'):
            cuid = ComponentUID(_slice)

        _slice = m.b.b2[:,'a'].component('vn')[:,'c',3,:,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[*,a].vn[*,c,3,*,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[1,'a'].component('vn')[:,'c',3,:,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[1,a].vn[*,c,3,*,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[...].component('vn')[:,'c',3,:,:]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[**].vn[*,c,3,*,*]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)

        _slice = m.b.b2[:,'a'].component('vn')[...]
        cuid = ComponentUID(_slice)
        cuid_str = ComponentUID('b.b2[*,a].vn[**]')
        self.assertEqual(str(cuid), str(cuid_str))
        self.assertListSameComponents(m, cuid, cuid_str)


class TestEnviron(unittest.TestCase):

    def test_components(self):
        self.assertGreaterEqual(
            set(x[0] for x in pyomo.core.base._pyomo.model_components()),
            set(['Set', 'Param', 'Var', 'Objective', 'Constraint'])
        )

    def test_sets(self):
        self.assertGreaterEqual(
            set(x[0] for x in pyomo.core.base._pyomo.predefined_sets()),
            set(['Reals', 'Integers', 'Boolean'])
        )

if __name__ == "__main__":
    unittest.main()
