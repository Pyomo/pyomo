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
# This inport will have to change when we decide where this should go...
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import (
        UnindexedComponent_set,
        normalize_index,
        )
from pyomo.dae.flatten import (
        flatten_dae_components,
        flatten_components_along_sets,
        )

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


    # TODO: Add tests for Sets with dimen==None


class TestFlatten(TestCategorize):

    def _hashRef(self, ref):
        if not ref.is_indexed():
            return (id(ref),)
        else:
            return tuple(sorted(id(_) for _ in ref.values()))

    def _model_1(self):
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
        m = self._model_1()
        
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
                *list(self._hashRef(Reference(m.v3[:,:,j])) for j in m.comp),
                *list(self._hashRef(Reference(m.b.b1[:].v2[:,j])) for j in m.comp),
                *list(self._hashRef(Reference(m.b.b2[:,:].v1[j])) for j in m.comp),
                }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.time and sets[1] is m.space
                    and sets[2] is m.time):
                ref_data = {
                self._hashRef(m.v_tst),
                *list(self._hashRef(Reference(m.b.b2[:,:].v2[:,j])) for j in m.comp),
                }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif (len(sets) == 3 and sets[0] is m.time and sets[1] is m.space
                    and sets[2] is m.space):
                ref_data = {
                self._hashRef(Reference(m.b.b1[:].b_s[:].v1[:])),
                *list(self._hashRef(Reference(m.b.b1[:].b_s[:].v2[:,j])) for j in m.comp),
                }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            else:
                raise RuntimeError()

    def test_flatten_m1_empty(self):
        m = self._model_1()
        
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
        m = self._model_1()
        
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
                        *list(self._hashRef(m.v1[t]) for t in T),
                        *list(self._hashRef(m.v_tt[t1,t2]) for t1, t2 in TT),
                        *list(self._hashRef(m.b.b1[t].v0) for t in T),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is m.space:
                ref_data = {
        *list(self._hashRef(Reference(m.v2[t,:])) for t in T),
        *list(self._hashRef(Reference(m.v3[t,:,j])) for t, j in TC),
        *list(self._hashRef(Reference(m.v_tst[t1,:,t2])) for t1, t2 in TT),
        *list(self._hashRef(Reference(m.b.b1[t].v1[:])) for t in T),
        *list(self._hashRef(Reference(m.b.b1[t].v2[:,j])) for t, j in TC),
        *list(self._hashRef(Reference(m.b.b1[t].b_s[:].v0)) for t in T),
        *list(self._hashRef(Reference(m.b.b2[t,:].v0)) for t in T),
        *list(self._hashRef(Reference(m.b.b2[t,:].v1[j])) for t, j in TC),
        *list(self._hashRef(Reference(m.b.b2[t1,:].v2[t2,j])) for t1, t2, j in TTC),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

            elif len(sets) == 2 and sets[0] is m.space and sets[1] is m.space:
                ref_data = {
        *list(self._hashRef(Reference(m.b.b1[t].b_s[:].v1[:])) for t in T),
        *list(self._hashRef(Reference(m.b.b1[t].b_s[:].v2[:,j])) for t,j in TC),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def test_flatten_m1_along_time(self):
        m = self._model_1()
        
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
# Components indexed by time and some other set(s)
*list(self._hashRef(Reference(m.v2[:,x])) for x in S),
*list(self._hashRef(Reference(m.v3[:,x,j])) for x, j in SC),
*list(self._hashRef(Reference(m.b.b1[:].v1[x])) for x in S),
*list(self._hashRef(Reference(m.b.b1[:].v2[x,j])) for x, j in SC),
*list(self._hashRef(Reference(m.b.b1[:].b_s[x].v0)) for x in S),
*list(self._hashRef(Reference(m.b.b1[:].b_s[x1].v1[x2])) for x1, x2 in SS),
*list(self._hashRef(Reference(m.b.b1[:].b_s[x1].v2[x2,j])) for x1,x2,j in SSC),
*list(self._hashRef(Reference(m.b.b2[:,x].v0)) for x in S),
*list(self._hashRef(Reference(m.b.b2[:,x].v1[j])) for x, j in SC),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.time and sets[1] is m.time:
                ref_data = {
self._hashRef(Reference(m.v_tt)),
*list(self._hashRef(Reference(m.v_tst[:,x,:])) for x in S),
*list(self._hashRef(Reference(m.b.b2[:,x].v2[:,j])) for x, j in SC),
                        }
                assert len(comps) == len(ref_data)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            else:
                raise RuntimeError()

    def _model_2(self):
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
        m = self._model_2()

        sets = ComponentSet((m.d2,))
        # need to set `flatten` to False here to properly access data,
        # since model was created with `flatten == False`.
        normalize_index.flatten = False
        sets_list, comps_list = flatten_components_along_sets(m, sets, Var)

        ref1 = Reference(m.v_2n[:,('c',3)])

        import pdb; pdb.set_trace()
        ref1 = Reference(m.v_2n[:,('d',4,5)])

        assert len(sets_list) == len(comps_list)
        assert len(sets_list) == 2

        D1N = m.d1*m.dn
        D1NN = m.d1.cross(m.dn, m.dn)
        D1N1 = m.d1.cross(m.dn, m.d1)

        for sets, comps in zip(sets_list, comps_list):

            if len(sets) == 1 and sets[0] is m.d2:
                ref_data = {
*list(self._hashRef(Reference(m.v_2n[:,i_n])) for i_n in m.dn),
*list(self._hashRef(Reference(m.v_12[i1,:])) for i1 in m.d1),
*list(self._hashRef(Reference(m.v_12n[i1,:,i_n])) for i1,i_n in D1N),
*list(self._hashRef(Reference(m.v_1n2n[i1,i_na,:,i_nb])) for i1, i_na, i_nb in D1NN),
*list(self._hashRef(Reference(m.b[i1,:,i_n].v0)) for i1,i_n in D1N),
*list(self._hashRef(Reference(m.b[i1a,:,i_n].v1[i1b])) for i1a, i_n, i1b in D1N1),
*list(self._hashRef(Reference(m.b[i1,:,i_na].v1[i_nb])) for i1, i_na, i_nb in D1NN),
                        }
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

        normalize_index.flatten = True

    def test_flatten_m2_1d(self):
        m = self._model_2()

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
                ref_data = {
# Don't expand indices:
*list(self._hashRef(Reference(m.v_12[:,i2])) for i2 in m.d2),
*list(self._hashRef(Reference(m.v_212[i2a,:,i2b])) for i2a, i2b in D22),
*list(self._hashRef(Reference(m.v_12n[:,i2,i_n])) for i2, i_n in D2N),
*list(self._hashRef(Reference(m.v_1n2n[:,i_na,i2,i_nb])) for i_na, i2, i_nb in DN2N),
*list(self._hashRef(Reference(m.b[:,i2,i_n].v0)) for i2, i_n in D2N),
*list(self._hashRef(Reference(m.b[:,i2a,i_n].v2[i2b])) for i2a, i_n, i2b in D2N2),
*list(self._hashRef(Reference(m.b[:,i2,i_na].vn[i_nb])) for i2, i_na, i_nb in D2NN),
                        }
                # Expect length to be 38
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
                ref_data = {
                        *list(self._hashRef(v) for v in m.v_2n.values()),
                        }
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)
            elif len(sets) == 2 and sets[0] is m.d1 and sets[1] is m.d1:
                ref_data = {
                *list(self._hashRef(Reference(m.b[:,i2,i_n].v1[:])) for i2, i_n in D2N),
                        }
                assert len(ref_data) == len(comps)
                for comp in comps:
                    self.assertIn(self._hashRef(comp), ref_data)

        normalize_index.flatten = True


if __name__ == "__main__":
    #unittest.main()
    TestFlatten().test_flatten_m2_2d()

