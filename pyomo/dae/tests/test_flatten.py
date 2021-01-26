#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, Block, Var, Reference, Set, Constraint
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import (
        UnindexedComponent_set,
        normalize_index,
        )
from pyomo.dae.flatten import (
        flatten_dae_components,
        flatten_components_along_sets,
        )

class TestAssumedBehavior(unittest.TestCase):
    """
    These are some behaviors we rely on that weren't
    immediately obvious would be the case.
    """
    def test_cross(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1,2])
        m.s2 = Set(initialize=[3,4])
        m.s3 = Set(initialize=['a','b'])

        normalize_index.flatten = True

        for i in m.s1.cross():
            # A "vacuous cross product" will place set elements in tuples
            self.assertIs(type(i), tuple)

        for i in m.s1.cross(m.s2, m.s3):
            self.assertIs(type(i), tuple)
            for j in i:
                # A cross product with multiple arguments does not produce
                # nested tuples
                self.assertIsNot(type(j), tuple)

        normalize_index.flatten = False
        # This behavior is consistent regardless of the value of 
        # normalize_index.flatten

        for i in m.s1.cross():
            self.assertIs(type(i), tuple)
        for i in m.s1.cross(m.s2, m.s3):
            self.assertIs(type(i), tuple)
            for j in i:
                self.assertIsNot(type(j), tuple)

        normalize_index.flatten = True

    def test_subsets(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1,2])
        m.s2 = Set(initialize=[3,4])
        m.s3 = Set(initialize=['a','b'])

        normalize_index.flatten = True

        s12 = m.s1 * m.s2
        s12_3 = s12 * m.s3

        s123 = m.s1.cross(m.s2, m.s3)

        subsets12_3 = list(s12_3.subsets())
        subsets123 = list(s123.subsets())

        # `subsets` identifies the "base sets" regardless
        # of whether products are nested.

        self.assertEqual(len(subsets12_3), len(subsets123))
        for s_a, s_b in zip(subsets12_3, subsets123):
            self.assertIs(s_a, s_b)

        normalize_index.flatten = False

        for i, j in s12_3:
            # Make sure we have a nested product with flatten False.
            self.assertIs(type(i), tuple)

        # The behavior of subsets is unchanged
        subsets12_3 = list(s12_3.subsets())
        subsets123 = list(s123.subsets())

        self.assertEqual(len(subsets12_3), len(subsets123))
        for s_a, s_b in zip(subsets12_3, subsets123):
            self.assertIs(s_a, s_b)

        normalize_index.flatten = True


class TestCategorize(unittest.TestCase):
    def _hashRef(self, ref):
        return tuple(sorted(id(_) for _ in ref.values()))

    def test_flat_model(self):
        m = ConcreteModel()
        m.T = ContinuousSet(bounds=(0,1))
        m.x = Var()
        m.y = Var([1,2])
        m.a = Var(m.T)
        m.b = Var(m.T, [1,2])
        m.c = Var([3,4], m.T)

        regular, time = flatten_dae_components(m, m.T, Var)
        regular_id = set(id(_) for _ in regular)
        self.assertEqual(len(regular), 3)
        self.assertIn(id(m.x), regular_id)
        self.assertIn(id(m.y[1]), regular_id)
        self.assertIn(id(m.y[2]), regular_id)
        # Output for debugging
        #for v in time:
        #    v.pprint()
        #    for _ in v.values():
        #        print"     -> ", _.name
        ref_data = {
            self._hashRef(Reference(m.a[:])),
            self._hashRef(Reference(m.b[:,1])),
            self._hashRef(Reference(m.b[:,2])),
            self._hashRef(Reference(m.c[3,:])),
            self._hashRef(Reference(m.c[4,:])),
        }
        self.assertEqual(len(time), len(ref_data))
        for ref in time:
            self.assertIn(self._hashRef(ref), ref_data)

    def test_1level_model(self):
        m = ConcreteModel()
        m.T = ContinuousSet(bounds=(0,1))
        @m.Block([1,2],m.T)
        def B(b, i, t):
            b.x = Var(list(range(2*i, 2*i+2)))

        regular, time = flatten_dae_components(m, m.T, Var)
        self.assertEqual(len(regular), 0)
        # Output for debugging
        #for v in time:
        #    v.pprint()
        #    for _ in v.values():
        #        print"     -> ", _.name
        ref_data = {
            self._hashRef(Reference(m.B[1,:].x[2])),
            self._hashRef(Reference(m.B[1,:].x[3])),
            self._hashRef(Reference(m.B[2,:].x[4])),
            self._hashRef(Reference(m.B[2,:].x[5])),
        }
        self.assertEqual(len(time), len(ref_data))
        for ref in time:
            self.assertIn(self._hashRef(ref), ref_data)


    def test_2level_model(self):
        m = ConcreteModel()
        m.T = ContinuousSet(bounds=(0,1))
        @m.Block([1,2],m.T)
        def B(b, i, t):
            @b.Block(list(range(2*i, 2*i+2)))
            def bb(bb, j):
                bb.y = Var([10,11])
            b.x = Var(list(range(2*i, 2*i+2)))

        regular, time = flatten_dae_components(m, m.T, Var)
        self.assertEqual(len(regular), 0)
        # Output for debugging
        #for v in time:
        #    v.pprint()
        #    for _ in v.values():
        #        print"     -> ", _.name
        ref_data = {
            self._hashRef(Reference(m.B[1,:].x[2])),
            self._hashRef(Reference(m.B[1,:].x[3])),
            self._hashRef(Reference(m.B[2,:].x[4])),
            self._hashRef(Reference(m.B[2,:].x[5])),
            self._hashRef(Reference(m.B[1,:].bb[2].y[10])),
            self._hashRef(Reference(m.B[1,:].bb[2].y[11])),
            self._hashRef(Reference(m.B[1,:].bb[3].y[10])),
            self._hashRef(Reference(m.B[1,:].bb[3].y[11])),
            self._hashRef(Reference(m.B[2,:].bb[4].y[10])),
            self._hashRef(Reference(m.B[2,:].bb[4].y[11])),
            self._hashRef(Reference(m.B[2,:].bb[5].y[10])),
            self._hashRef(Reference(m.B[2,:].bb[5].y[11])),
        }
        self.assertEqual(len(time), len(ref_data))
        for ref in time:
            self.assertIn(self._hashRef(ref), ref_data)


    def test_2dim_set(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0,1))

        m.v = Var(m.time, [('a',1), ('b',2)])

        scalar, dae = flatten_dae_components(m, m.time, Var)
        self.assertEqual(len(scalar), 0)
        ref_data = {
                self._hashRef(Reference(m.v[:,'a',1])),
                self._hashRef(Reference(m.v[:,'b',2])),
                }
        self.assertEqual(len(dae), len(ref_data))
        for ref in dae:
            self.assertIn(self._hashRef(ref), ref_data)

    
    def test_indexed_block(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0,1))
        m.comp = Set(initialize=['a', 'b'])

        def bb_rule(bb, t):
            bb.dae_var = Var()

        def b_rule(b, c):
            b.bb = Block(m.time, rule=bb_rule)

        m.b = Block(m.comp, rule=b_rule)

        scalar, dae = flatten_dae_components(m, m.time, Var)
        self.assertEqual(len(scalar), 0)
        ref_data = {
                self._hashRef(Reference(m.b['a'].bb[:].dae_var)),
                self._hashRef(Reference(m.b['b'].bb[:].dae_var)),
                }
        self.assertEqual(len(dae), len(ref_data))
        for ref in dae:
            self.assertIn(self._hashRef(ref), ref_data)


    def test_constraint(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0,1))
        m.comp = Set(initialize=['a', 'b'])
        m.v0 = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.comp)
        
        def c0_rule(m):
            return m.v0 == 1
        m.c0 = Constraint(rule=c0_rule)

        def c1_rule(m, t):
            return m.v1[t] == 3
        m.c1 = Constraint(m.time, rule=c1_rule)

        @m.Block(m.time)
        def b(b, t):
            def c2_rule(b, j):
                return b.model().v2[t, j] == 5
            b.c2 = Constraint(m.comp, rule=c2_rule)

        scalar, dae = flatten_dae_components(m, m.time, Constraint)
        hash_scalar = {id(s) for s in scalar}
        self.assertIn(id(m.c0), hash_scalar)

        ref_data = {
                self._hashRef(Reference(m.c1[:])),
                self._hashRef(Reference(m.b[:].c2['a'])),
                self._hashRef(Reference(m.b[:].c2['b'])),
                }
        self.assertEqual(len(dae), len(ref_data))
        for ref in dae:
            self.assertIn(self._hashRef(ref), ref_data)


    def test_constraint_skip(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0,1))

        m.v = Var(m.time)
        def c_rule(m, t):
            if t == m.time.first():
                return Constraint.Skip
            return m.v[t] == 1.

        m.c = Constraint(m.time, rule=c_rule)
        scalar, dae = flatten_dae_components(m, m.time, Constraint)

        ref_data = {
                self._hashRef(Reference(m.c[:])),
                }
        self.assertEqual(len(dae), len(ref_data))
        for ref in dae:
            self.assertIn(self._hashRef(ref), ref_data)


class TestFlatten(TestCategorize):

    def _hashRef(self, ref):
        if not ref.is_indexed():
            return (id(ref),)
        else:
            return tuple(sorted(id(_) for _ in ref.values()))

    def _model1_1d_sets(self):
        # One-dimensional sets, no skipping.
        m = ConcreteModel()
        m.time = Set(initialize=[1,2,3])
        m.space = Set(initialize=[0.0, 0.5, 1.0])
        m.comp = Set(initialize=['a','b'])

        m.v0 = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.space)
        m.v3 = Var(m.time, m.space, m.comp)

        m.v_tt = Var(m.time, m.time)
        m.v_tst = Var(m.time, m.space, m.time)

        @m.Block()
        def b(b):

            @b.Block(m.time)
            def b1(b1):
                b1.v0 = Var()
                b1.v1 = Var(m.space)
                b1.v2 = Var(m.space, m.comp)

                @b1.Block(m.space)
                def b_s(b_s):
                    b_s.v0 = Var()
                    b_s.v1 = Var(m.space)
                    b_s.v2 = Var(m.space, m.comp)

            @b.Block(m.time, m.space)
            def b2(b2):
                b2.v0 = Var()
                b2.v1 = Var(m.comp)
                b2.v2 = Var(m.time, m.comp)

        return m

    def test_flatten_m1_along_time_space(self):
        m = self._model1_1d_sets()
        
        sets = ComponentSet((m.time, m.space))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 6

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                        self._hashRef(m.v0)
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 1 and sets[0] is m.time:
                ref_data = {
                        self._hashRef(Reference(m.v1)),
                        self._hashRef(Reference(m.b.b1[:].v0)),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.time and sets[1] is m.time:
                ref_data = {
                        self._hashRef(Reference(m.v_tt)),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                ref_data = {
                    self._hashRef(m.v2),
                    self._hashRef(Reference(m.b.b1[:].v1[:])),
                    self._hashRef(Reference(m.b.b2[:,:].v0)),
                    self._hashRef(Reference(m.b.b1[:].b_s[:].v0)),
                }
                ref_data.update(self._hashRef(Reference(m.v3[:,:,j])) for j in m.comp)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].v2[:,j])) for j in m.comp)
                ref_data.update(self._hashRef(Reference(m.b.b2[:,:].v1[j])) for j in m.comp)

                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.time and sets[1] is m.space
                    and sets[2] is m.time):
                ref_data = {
                    self._hashRef(m.v_tst),
                }
                ref_data.update(self._hashRef(Reference(m.b.b2[:,:].v2[:,j])) for j in m.comp)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.time and sets[1] is m.space
                    and sets[2] is m.space):
                ref_data = {
                    self._hashRef(Reference(m.b.b1[:].b_s[:].v1[:])),
                }
                ref_data.update(self._hashRef(Reference(m.b.b1[:].b_s[:].v2[:,j])) for j in m.comp),
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            else:
                raise RuntimeError()

    def test_flatten_m1_empty(self):
        m = self._model1_1d_sets()
        
        sets = ComponentSet()
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 1

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                    self._hashRef(v) for v in m.component_data_objects(Var)
                    }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_along_space(self):
        m = self._model1_1d_sets()
        
        sets = ComponentSet((m.space,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3

        T = m.time
        TC = m.time*m.comp
        TT = m.time*m.time
        TTC = m.time*m.time*m.comp
        # These products are nested, i.e. ((t,t),j). This is fine
        # as normalize_index.flatten is True.

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                    self._hashRef(m.v0),
                }
                ref_data.update(self._hashRef(m.v1[t]) for t in T)
                ref_data.update(self._hashRef(m.v_tt[t1,t2]) for t1, t2 in TT)
                ref_data.update(self._hashRef(m.b.b1[t].v0) for t in T)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.space:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v2[t,:])) for t in T)
                ref_data.update(self._hashRef(Reference(m.v3[t,:,j])) for t, j in TC)
                ref_data.update(self._hashRef(Reference(m.v_tst[t1,:,t2])) for t1, t2 in TT)
                ref_data.update(self._hashRef(Reference(m.b.b1[t].v1[:])) for t in T)
                ref_data.update(self._hashRef(Reference(m.b.b1[t].v2[:,j])) for t, j in TC)
                ref_data.update(self._hashRef(Reference(m.b.b1[t].b_s[:].v0)) for t in T)
                ref_data.update(self._hashRef(Reference(m.b.b2[t,:].v0)) for t in T)
                ref_data.update(self._hashRef(Reference(m.b.b2[t,:].v1[j])) for t, j in TC)
                ref_data.update(self._hashRef(Reference(m.b.b2[t1,:].v2[t2,j])) for t1, t2, j in TTC)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.space and sets[1] is m.space:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b.b1[t].b_s[:].v1[:])) for t in T)
                ref_data.update(self._hashRef(Reference(m.b.b1[t].b_s[:].v2[:,j])) for t,j in TC)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_along_time(self):
        m = self._model1_1d_sets()
        
        sets = ComponentSet((m.time,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        S = m.space
        SS = m.space*m.space
        SC = m.space*m.comp
        SSC = m.space*m.space*m.comp

        assert len(sets_list) == 3
        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                        self._hashRef(Reference(m.v0)),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.time:
                ref_data = {
                    # Components indexed only by time;                        
                    self._hashRef(Reference(m.v1)),
                    self._hashRef(Reference(m.b.b1[:].v0)),
                }
                # Components indexed by time and some other set(s)
                ref_data.update(self._hashRef(Reference(m.v2[:,x])) for x in S)
                ref_data.update(self._hashRef(Reference(m.v3[:,x,j])) for x, j in SC)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].v1[x])) for x in S)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].v2[x,j])) for x, j in SC)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].b_s[x].v0)) for x in S)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].b_s[x1].v1[x2])) for x1, x2 in SS)
                ref_data.update(self._hashRef(Reference(m.b.b1[:].b_s[x1].v2[x2,j])) for x1,x2,j in SSC)
                ref_data.update(self._hashRef(Reference(m.b.b2[:,x].v0)) for x in S)
                ref_data.update(self._hashRef(Reference(m.b.b2[:,x].v1[j])) for x, j in SC)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.time and sets[1] is m.time:
                ref_data = {
                    self._hashRef(Reference(m.v_tt)),
                }
                ref_data.update(self._hashRef(Reference(m.v_tst[:,x,:])) for x in S)
                ref_data.update(self._hashRef(Reference(m.b.b2[:,x].v2[:,j])) for x, j in SC)
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def _model2_nd_sets(self):
        # A more simple model, but now with some higher-dimension sets
        m = ConcreteModel()

        normalize_index.flatten = False

        m.d1 = Set(initialize=[1,2])
        m.d2 = Set(initialize=[('a',1), ('b',2)])
        m.dn = Set(initialize=[('c',3), ('d',4,5)], dimen=None)

        m.v_2n = Var(m.d2, m.dn)
        m.v_12 = Var(m.d1, m.d2)
        m.v_212 = Var(m.d2, m.d1, m.d2)
        m.v_12n = Var(m.d1, m.d2, m.dn)
        m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)

        @m.Block(m.d1, m.d2, m.dn)
        def b(b, i1, i2, i3):
            b.v0 = Var()
            b.v1 = Var(m.d1)
            b.v2 = Var(m.d2)
            b.vn = Var(m.dn)

        normalize_index.flatten = True

        return m

    def test_flatten_m2_2d(self):
        """
        This test has some issues due to incompatibility between
        slicing and `normalize_index.flatten==False`.
        """
        # TODO: If the user wants to do what this test attempts
        # and "flatten along" sets of dimension > 1 when
        # `normalize_index.flatten == False`, they will have
        # some problems. See issue #1800.
        m = self._model2_nd_sets()

        sets = ComponentSet((m.d2,))
        # need to set `flatten` to False here to properly access data,
        # since model was created with `flatten == False`.
        normalize_index.flatten = False
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        ref1 = Reference(m.v_2n[:,('c',3)])

        ref_set = ref1._index._ref # _index is a _ReferenceSet

        #next(ref_set._get_iter(ref_set._slice, ('a',1)))
        # ^ This raises a somewhat cryptic error message.
        # _ReferenceSet.__contains__ seems to be the culprit,
        # which is called in validate_index for every __getitem__.
        self.assertNotIn(('a',1), ref1._index._ref)
        # ^ This does not seem to behave as expected...
        # Reason is incompatibility with flatten==False.

        self.assertEqual(len(sets_list), len(comps_list))
        self.assertEqual(len(sets_list), 2)

        #for sets, comps in zip(sets_list, comps_list):
        #    if len(sets) == 1 and sets[0] is m.d2:
        #        ref_data = {
        #                self._hashRef(Reference(m.v_2n[:,('a',1)])),
        #                self._hashRef(Reference(m.v_2n[:,('b',2)])),
        #                }
        #        # Cannot access the data of ^these references
        #        for comp in comps:
        #            self.assertIn(self._hashRef(comp), ref_data)

        normalize_index.flatten = True

    def test_flatten_m2_1d(self):
        m = self._model2_nd_sets()

        sets = ComponentSet((m.d1,))
        # need to set `flatten` to False here to properly access data,
        # since model was created with `flatten == False`.
        normalize_index.flatten = False
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3

        D22 = m.d2*m.d2
        D2N = m.d2*m.dn
        DN2N = m.dn.cross(m.d2, m.dn)
        D2NN = m.d2.cross(m.dn, m.dn)
        D2N2 = m.d2.cross(m.dn, m.d2)

        for sets, comps in zip(sets_list, comps_list):

            if len(sets) == 1 and sets[0] is m.d1:
                ref_data = set()
                # Don't expand indices:
                ref_data.update(self._hashRef(Reference(m.v_12[:,i2])) for i2 in m.d2)
                ref_data.update(self._hashRef(Reference(m.v_212[i2a,:,i2b])) for i2a, i2b in D22)
                ref_data.update(self._hashRef(Reference(m.v_12n[:,i2,i_n])) for i2, i_n in D2N)
                ref_data.update(self._hashRef(Reference(m.v_1n2n[:,i_na,i2,i_nb])) for i_na, i2, i_nb in DN2N)
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_n].v0)) for i2, i_n in D2N)
                ref_data.update(self._hashRef(Reference(m.b[:,i2a,i_n].v2[i2b])) for i2a, i_n, i2b in D2N2)
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_na].vn[i_nb])) for i2, i_na, i_nb in D2NN)
                # Expect length to be 38
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update(self._hashRef(v) for v in m.v_2n.values())
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and sets[1] is m.d1:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_n].v1[:])) for i2, i_n in D2N)
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

        normalize_index.flatten = True

    def _model3_nd_sets_normalizeflatten(self):
        # The same as model 2, but now with `normalize_index.flatten == True`
        m = ConcreteModel()

        m.d1 = Set(initialize=[1,2])
        m.d2 = Set(initialize=[('a',1), ('b',2)])
        m.dn = Set(initialize=[('c',3), ('d',4,5)], dimen=None)

        m.v_2n = Var(m.d2, m.dn)
        m.v_12 = Var(m.d1, m.d2)
        m.v_212 = Var(m.d2, m.d1, m.d2)
        m.v_12n = Var(m.d1, m.d2, m.dn)
        m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)

        m.b = Block(m.d1, m.d2, m.dn)
        for i1 in m.d1:
            for i2 in m.d2:
                for i_n in m.dn:
                    m.b[i1,i2,i_n].v0 = Var()
                    m.b[i1,i2,i_n].v1 = Var(m.d1)
                    m.b[i1,i2,i_n].v2 = Var(m.d2)
                    m.b[i1,i2,i_n].vn = Var(m.dn)

        return m

    def test_flatten_m3_1d(self):
        m = self._model3_nd_sets_normalizeflatten()

        sets = ComponentSet((m.d1,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3

        for sets, comps in zip(sets_list, comps_list):

            if len(sets) == 1 and sets[0] is m.d1:
                ref_data = set()
                # Must iterate and slice in a manner consistent with
                # `normalize_index.flatten == True`
                ref_data.update(self._hashRef(Reference(m.v_12[:,i2]))
                      for i2 in m.d2) # 2
                ref_data.update(self._hashRef(Reference(m.v_212[i2a,:,i2b]))
                      for i2a in m.d2 for i2b in m.d2) # 4
                ref_data.update(self._hashRef(Reference(m.v_12n[:,i2,i_n]))
                      for i2 in m.d2 for i_n in m.dn) # 4
                ref_data.update(self._hashRef(Reference(m.v_1n2n[:,i_na,i2,i_nb]))
                      for i_na in m.dn for i2 in m.d2 for i_nb in m.dn) # 8
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_n].v0))
                      for i2 in m.d2 for i_n in m.dn) # 4
                ref_data.update(self._hashRef(Reference(m.b[:,i2a,i_n].v2[i2b]))
                      for i2a in m.d2 for i_n in m.dn for i2b in m.d2) # 8
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_na].vn[i_nb]))
                      for i2 in m.d2 for i_na in m.dn for i_nb in m.dn) # 8
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 38
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update(self._hashRef(v) for v in m.v_2n.values())
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and sets[1] is m.d1:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b[:,i2,i_n].v1[:]))
                      for i2 in m.d2 for i_n in m.dn)
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

    def test_flatten_m3_2d(self):
        m = self._model3_nd_sets_normalizeflatten()

        sets = ComponentSet((m.d2,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 2

        for sets, comps in zip(sets_list, comps_list):

            if len(sets) == 1 and sets[0] is m.d2:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v_2n[:,:,i_n]))
                      for i_n in m.dn) # 2
                ref_data.update(self._hashRef(Reference(m.v_12[i1,:,:]))
                      for i1 in m.d1) # 2
                ref_data.update(self._hashRef(Reference(m.v_12n[i1,:,:,i_n]))
                      for i1 in m.d1 for i_n in m.dn) # 4
                ref_data.update(self._hashRef(Reference(m.v_1n2n[i1,i_na,:,:,i_nb]))
                      for i1 in m.d1 for i_na in m.dn for i_nb in m.dn) # 8
                ref_data.update(self._hashRef(Reference(m.b[i1,:,:,i_n].v0))
                      for i1 in m.d1 for i_n in m.dn) # 4
                ref_data.update(self._hashRef(Reference(m.b[i1a,:,:,i_n].v1[i1b]))
                      for i1a in m.d1 for i_n in m.dn for i1b in m.d1) # 8
                ref_data.update(self._hashRef(Reference(m.b[i1,:,:,i_na].vn[i_nb]))
                      for i1 in m.d1 for i_na in m.dn for i_nb in m.dn) # 8
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 36
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d2 and sets[1] is m.d2:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v_212[:,:,i1,:,:]))
                      for i1 in m.d1)
                ref_data.update(self._hashRef(Reference(m.b[i1,:,:,i_n].v2[:,:]))
                      for i1 in m.d1 for i_n in m.dn)
                assert len(ref_data) == len(comps)
                assert len(ref_data) == 6
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m3_nd(self):
        m = self._model3_nd_sets_normalizeflatten()

        # Can't create a slice with multiple ellipses in the same index.
        m.del_component(m.v_1n2n)

        sets = ComponentSet((m.dn,))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 3

        for sets, comps in zip(sets_list, comps_list):

            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = set()
                ref_data.update(self._hashRef(v) for v in m.v_12.values())
                ref_data.update(self._hashRef(v) for v in m.v_212.values())
                assert len(comps) == len(ref_data)
                assert len(comps) == 12
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 1 and sets[0] is m.dn:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v_2n[i2,...]))
                      for i2 in m.d2) # 2
                ref_data.update(self._hashRef(Reference(m.v_12n[i1,i2,...]))
                      for i1 in m.d1 for i2 in m.d2) # 4
                ref_data.update(self._hashRef(Reference(m.b[i1,i2,...].v0))
                      for i1 in m.d1 for i2 in m.d2) # 4
                ref_data.update(self._hashRef(Reference(m.b[i1a,i2,...].v1[i1b]))
                      for i1a in m.d1 for i2 in m.d2 for i1b in m.d1) # 8
                ref_data.update(self._hashRef(Reference(m.b[i1,i2a,...].v2[i2b]))
                      for i1 in m.d1 for i2a in m.d2 for i2b in m.d2) # 8
                assert len(comps) == len(ref_data)
                assert len(comps) == 26
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.dn and sets[1] is m.dn:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b[i1,i2,...].vn[...]))
                      for i1 in m.d1 for i2 in m.d2) # 4
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m3_1_2(self):
        m = self._model3_nd_sets_normalizeflatten()

        sets = ComponentSet((m.d1,m.d2))
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 5

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is m.d2:
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v_2n[:,:,i_n]))
                      for i_n in m.dn)
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.d1 and sets[1] is m.d2:
                ref_data = {
                    self._hashRef(Reference(m.v_12[...])), # 1
                }
                ref_data.update(self._hashRef(Reference(m.v_12n[:,:,:,i_n]))
                      for i_n in m.dn) # 2
                ref_data.update(self._hashRef(Reference(m.v_1n2n[:,i_na,:,:,i_nb]))
                      for i_na in m.dn for i_nb in m.dn) # 4
                ref_data.update(self._hashRef(Reference(m.b[:,:,:,i_n].v0))
                      for i_n in m.dn) # 2
                ref_data.update(self._hashRef(Reference(m.b[:,:,:,i_na].vn[i_nb]))
                      for i_na in m.dn for i_nb in m.dn) # 4
                self.assertEqual(len(ref_data), len(comps))
                self.assertEqual(len(comps), 13)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.d1 and sets[1] is m.d2
                    and sets[2] is m.d1):
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b[:,:,:,i_n].v1[:]))
                      for i_n in m.dn) # 2
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.d1 and sets[1] is m.d2
                    and sets[2] is m.d2):
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.b[:,:,:,i_n].v2[:,:]))
                      for i_n in m.dn) # 2
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.d2 and sets[1] is m.d1
                    and sets[2] is m.d2):
                ref_data = {
                    self._hashRef(Reference(m.v_212[...])), # 1
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            else:
                raise RuntimeError()

    def test_specified_index_1(self):
        """
        Components indexed by flattened sets and others
        """
        m = ConcreteModel()

        m.time = Set(initialize=[1,2,3])
        m.space = Set(initialize=[2,4,6])
        m.phase = Set(initialize=['p1', 'p2'])
        m.comp = Set(initialize=['a','b'])

        phase_comp = m.comp * m.phase
        n_phase_comp = len(m.phase)*len(m.comp)

        m.v = Var(m.time, m.comp, m.space, m.phase)

        @m.Block(m.time, m.comp,  m.space, m.phase)
        def b(b, t, j, x, p):
            b.v1 = Var()

            if x != 2:
                b.v2 = Var()

        sets = (m.time, m.space)

        # No specified indices
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        # Everything is indexed by time and space
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We missed b.v2 by descending into the "first" index
                # of the block
                self.assertEqual(len(comps), 2*n_phase_comp)
                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v[:,j,:,p]))
                      for j, p in phase_comp)
                ref_data.update(self._hashRef(Reference(m.b[:,j,:,p].v1))
                      for j, p in phase_comp)
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

        # Space index specified:
        indices = ComponentMap([(m.space, 4)])
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var,
                indices=indices)

        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We descended into a block data that includes v2
                self.assertEqual(len(comps), 3*n_phase_comp)

                # Slices where we expect an attribute error somewhere,
                # due to v2 being "skipped"
                incomplete_slices = list(m.b[:,j,:,p].v2 for j, p in phase_comp)
                for ref in incomplete_slices:
                    ref.attribute_errors_generate_exceptions = False
                incomplete_refs = list(Reference(sl) for sl in incomplete_slices)

                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v[:,j,:,p]))
                      for j, p in phase_comp)
                ref_data.update(self._hashRef(Reference(m.b[:,j,:,p].v1))
                      for j, p in phase_comp)
                ref_data.update(self._hashRef(ref) for ref in incomplete_refs)
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

        # Time and space indices specified
        indices = (3, 6)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var,
                indices=indices)

        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We descended into a block data that includes v2
                self.assertEqual(len(comps), 3*n_phase_comp)

                # Slices where we expect an attribute error somewhere,
                # due to v2 being "skipped"
                incomplete_slices = list(m.b[:,j,:,p].v2 for j, p in phase_comp)
                for ref in incomplete_slices:
                    ref.attribute_errors_generate_exceptions = False
                incomplete_refs = list(Reference(sl) for sl in incomplete_slices)

                ref_data = set()
                ref_data.update(self._hashRef(Reference(m.v[:,j,:,p]))
                      for j, p in phase_comp)
                ref_data.update(self._hashRef(Reference(m.b[:,j,:,p].v1))
                      for j, p in phase_comp)
                ref_data.update(self._hashRef(ref) for ref in incomplete_refs)
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_specified_index_2(self):
        """
        Components indexed only by flattened sets
        """
        m = ConcreteModel()

        m.time = Set(initialize=[1,2,3])
        m.space = Set(initialize=[2,4,6])

        m.v = Var(m.time, m.space)

        @m.Block(m.time, m.space)
        def b(b, t, x):
            b.v1 = Var()

            if x != 2:
                b.v2 = Var()

        sets = (m.time, m.space)

        # No specified indices
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        # Everything is indexed by time and space
        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We missed b.v2 by descending into the "first" index
                # of the block
                self.assertEqual(len(comps), 2)
                ref_data = {
                        self._hashRef(Reference(m.v[...])),
                        self._hashRef(Reference(m.b[...].v1)),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

        # Space index specified:
        indices = ComponentMap([(m.space, 4)])
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var,
                indices=indices)

        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We descended into a block data that includes v2
                self.assertEqual(len(comps), 3)

                # Slices where we expect an attribute error somewhere,
                # due to v2 being "skipped"
                incomplete_slice = m.b[:,:].v2
                incomplete_slice.attribute_errors_generate_exceptions = False
                incomplete_ref = Reference(incomplete_slice)

                ref_data = {
                        self._hashRef(Reference(m.v[:,:])),
                        self._hashRef(Reference(m.b[:,:].v1)),
                        self._hashRef(incomplete_ref),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

        # Time and space indices specified
        indices = (3, 6)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var,
                indices=indices)

        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 1)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 2 and sets[0] is m.time and sets[1] is m.space:
                # We descended into a block data that includes v2
                self.assertEqual(len(comps), 3)

                # Slices where we expect an attribute error somewhere,
                # due to v2 being "skipped"
                incomplete_slice = m.b[:,:].v2
                incomplete_slice.attribute_errors_generate_exceptions = False
                incomplete_ref = Reference(incomplete_slice)

                ref_data = {
                        self._hashRef(Reference(m.v[:,:])),
                        self._hashRef(Reference(m.b[:,:].v1)),
                        self._hashRef(incomplete_ref),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def _model4_three_1d_sets(self):
        # A simple model with three sets to slice
        m = ConcreteModel()

        m.X = Set(initialize=[1,2,3])
        m.Y = Set(initialize=[1,2,3])
        m.Z = Set(initialize=[1,2,3])

        m.comp = Set(initialize=['a', 'b'])

        m.u = Var()

        m.v = Var(m.X, m.Y, m.Z, m.comp)
        m.base = Var(m.X, m.Y)

        @m.Block(m.X, m.Y, m.Z, m.comp)
        def b4(b, x, y, z, j):
            b.v = Var()

        @m.Block(m.X, m.Y)
        def b2(b, x, y):
            b.base = Var()
            b.v = Var(m.Z, m.comp)

        return m

    def test_model4_xyz(self):
        m = self._model4_three_1d_sets()

        sets = (m.X, m.Y, m.Z)
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        self.assertEqual(len(comps_list), len(sets_list))
        self.assertEqual(len(sets_list), 3)

        for sets, comps in zip(sets_list, comps_list):
            if len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                        self._hashRef(Reference(m.u)),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.X and sets[1] is m.Y:
                ref_data = {
                        self._hashRef(Reference(m.base[:,:])),
                        self._hashRef(Reference(m.b2[:,:].base)),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif (len(sets) == 3 and sets[0] is m.X and sets[1] is m.Y
                    and sets[2] is m.Z):
                ref_data = {
                        self._hashRef(Reference(m.v[:,:,:,'a'])),
                        self._hashRef(Reference(m.v[:,:,:,'b'])),
                        self._hashRef(Reference(m.b4[:,:,:,'a'].v)),
                        self._hashRef(Reference(m.b4[:,:,:,'b'].v)),
                        self._hashRef(Reference(m.b2[:,:].v[:,'a'])),
                        self._hashRef(Reference(m.b2[:,:].v[:,'b'])),
                        }
                self.assertEqual(len(ref_data), len(comps))
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()


class TestExceptional(unittest.TestCase):
    """
    These are the cases that motivate the try/excepts in the slice-checking
    part of the code.
    """
    def test_stop_iteration(self):
        """
        StopIteration is raised if we create an empty slice somewhere
        along the line. It is an open question what we should do in the
        case of an empty slice, but my preference is to omit it so we
        don't return a reference that doesn't admit any valid indices.
        """
        m = ConcreteModel()
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=['a','b','c'])

        m.v = Var(m.s1, m.s2)

        def con_rule(m, i, j):
            if j == 'a':
                # con[:, 'a'] will be an empty slice
                return Constraint.Skip
            return m.v[i, j] == 5.

        def vacuous_con_rule(m, i, j):
            # A very odd case
            return Constraint.Skip

        m.con = Constraint(m.s1, m.s2, rule=con_rule)

        with self.assertRaises(StopIteration):
            next(iter(m.con[:, 'a']))

        sets = (m.s1,)
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), len(m.s2)-1)

        m.del_component(m.con)
        m.vacuous_con = Constraint(m.s1, m.s2, rule=vacuous_con_rule)

        with self.assertRaises(StopIteration):
            next(iter(m.vacuous_con[...]))

        sets = (m.s1, m.s2)
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 0)

        m.del_component(m.vacuous_con)
        m.del_component(m.v) # No longer necessary

        # Same behavior can happen for blocks:
        
        def block_rule(b, i, j):
            b.v = Var()

        m.b = Block(m.s1, m.s2, rule=block_rule)

        for i in m.s1:
            del m.b[i, 'a']

        with self.assertRaises(StopIteration):
            next(iter(m.b[:, 'a'].v))

        sets = (m.s1,)
        ctype = Var
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 1)
        self.assertEqual(len(comps_list[0]), len(m.s2)-1)

        for idx in m.b:
            del m.b[idx]

        with self.assertRaises(StopIteration):
            next(iter(m.b[...].v))

        sets = (m.s1, m.s2)
        ctype = Var
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(comps_list), 0)

        # Have a component indexed by all of our sets that doesn't appear
        # when we flatten... seems like somewhat of a contradiction...
        subset_set = ComponentSet(m.b.index_set().subsets())
        for s in sets:
            self.assertIn(s, subset_set)
            
    def test_keyerror(self):
        """
        KeyErrors occur when we a component that we don't slice
        doesn't have data for some members of its indexing set.
        """
        m = ConcreteModel()
        m.s1 = Set(initialize=[1,2,3])
        m.s2 = Set(initialize=['a','b','c'])

        m.v = Var(m.s1, m.s2)

        def con_rule(m, i, j):
            if j == 'a':
                return Constraint.Skip
            return m.v[i, j] == 5.

        m.con = Constraint(m.s1, m.s2, rule=con_rule)

        with self.assertRaises(KeyError):
            for idx in m.con.index_set():
                temp = m.con[idx]

        sets = ()
        ctype = Constraint
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
        self.assertEqual(len(sets_list), len(comps_list))
        self.assertEqual(len(sets_list), 1)
        self.assertIs(sets_list[0][0], UnindexedComponent_set)
        self.assertEqual(len(comps_list[0]), len(list(m.con.values())))

        # NOTE: I have been unable to produce this behavior (KeyError)
        # with blocks.


if __name__ == "__main__":
    unittest.main()
