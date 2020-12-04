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
from pyomo.dae.flatten import flatten_dae_components

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


if __name__ == "__main__":
    unittest.main()
