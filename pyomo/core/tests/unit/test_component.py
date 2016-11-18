#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for components
#

import pyutilib.th as unittest

from pyomo.util import DeveloperError
import pyomo.core.base._pyomo
from pyomo.environ import *
from pyomo.core.base.block import generate_cuid_names

class TestComponent(unittest.TestCase):

    def test_construct_component_throws_exception(self):
        try:
            Component()
            self.fail("Expected DeveloperError")
        except DeveloperError:
            pass


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

class TestEnviron(unittest.TestCase):

    def test_components(self):
        self.assertTrue(set(x[0] for x in pyomo.core.base._pyomo.model_components()) >= set(['Set', 'Param', 'Var', 'Objective', 'Constraint']))

    def test_sets(self):
        self.assertTrue(set(x[0] for x in pyomo.core.base._pyomo.predefined_sets()) >= set(['Reals', 'Integers', 'Boolean']))

if __name__ == "__main__":
    unittest.main()
