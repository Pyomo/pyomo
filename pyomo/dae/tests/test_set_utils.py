#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Unit Tests for pyomo.dae.set_utils
"""
import os
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.core.base import (
    Block, Constraint, ConcreteModel, Var, Set, TransformationFactory
)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
    is_explicitly_indexed_by, is_in_block_indexed_by, get_index_set_except,
    deactivate_model_at,
)

currdir = dirname(abspath(__file__)) + os.sep


def make_model():
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 10))
    m.space = ContinuousSet(bounds=(0, 5))
    m.set1 = Set(initialize=['a', 'b', 'c'])
    m.set2 = Set(initialize=['d', 'e', 'f'])
    m.fs = Block()

    m.fs.v0 = Var(m.space, initialize=1)

    @m.fs.Block()
    def b1(b):
        b.v = Var(m.time, m.space, initialize=1)
        b.dv = DerivativeVar(b.v, wrt=m.time, initialize=0)

        b.con = Constraint(m.time, m.space,
                rule=lambda b, t, x: b.dv[t, x] == 7 - b.v[t, x])
        # Inconsistent

        @b.Block(m.time)
        def b2(b, t):
            b.v = Var(initialize=2)

    @m.fs.Block(m.time, m.space)
    def b2(b, t, x):
        b.v = Var(m.set1, initialize=2)

        @b.Block(m.set1)
        def b3(b, c):
            b.v = Var(m.set2, initialize=3)

            @b.Constraint(m.set2)
            def con(b, s):
                return (5*b.v[s] ==
                        m.fs.b2[m.time.first(), m.space.first()].v[c])
                # inconsistent

    @m.fs.Constraint(m.time)
    def con1(fs, t):
        return fs.b1.v[t, m.space.last()] == 5
    # Will be inconsistent

    @m.fs.Constraint(m.space)
    def con2(fs, x):
        return fs.b1.v[m.time.first(), x] == fs.v0[x]
    # will be consistent

    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
    disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')

    return m


class TestDaeSetUtils(unittest.TestCase):
    
    # Test explicit/implicit index detection functions
    def test_indexed_by(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0, 10))
        m.space = ContinuousSet(bounds=(0, 10))
        m.set = Set(initialize=['a', 'b', 'c'])
        m.set2 = Set(initialize=[('a', 1), ('b', 2)])
        m.v = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.space)
        m.v3 = Var(m.set, m.space, m.time)
        m.v4 = Var(m.time, m.set2)
        m.v5 = Var(m.set2, m.time, m.space)

        @m.Block()
        def b(b):
            b.v = Var()
            b.v1 = Var(m.time)
            b.v2 = Var(m.time, m.space)
            b.v3 = Var(m.set, m.space, m.time)

        @m.Block(m.time)
        def b1(b):
            b.v = Var()
            b.v1 = Var(m.space)
            b.v2 = Var(m.space, m.set)

        @m.Block(m.time, m.space)
        def b2(b):
            b.v = Var()
            b.v1 = Var(m.set)

            @b.Block()
            def b(bl):
                bl.v = Var()
                bl.v1 = Var(m.set)
                bl.v2 = Var(m.time)

        @m.Block(m.set2, m.time)
        def b3(b):
            b.v = Var()
            b.v1 = Var(m.space)

            @b.Block(m.space)
            def b(bb):
                bb.v = Var(m.set)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
        disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')

        self.assertFalse(is_explicitly_indexed_by(m.v, m.time))
        self.assertTrue(is_explicitly_indexed_by(m.b.v2, m.space))
        self.assertTrue(is_explicitly_indexed_by(m.b.v3, m.time, m.space))

        self.assertFalse(is_in_block_indexed_by(m.v1, m.time))
        self.assertFalse(is_in_block_indexed_by(m.v2, m.set))
        self.assertTrue(is_in_block_indexed_by(m.b1[m.time[1]].v2, m.time))

        self.assertTrue(is_in_block_indexed_by(
            m.b2[m.time[1], m.space[1]].b.v1, m.time))
        self.assertTrue(is_in_block_indexed_by(
            m.b2[m.time[1], m.space[1]].b.v2, m.time))
        self.assertTrue(is_explicitly_indexed_by(
            m.b2[m.time[1], m.space[1]].b.v2, m.time))
        self.assertFalse(is_in_block_indexed_by(
            m.b2[m.time[1], m.space[1]].b.v1, m.set))

        self.assertFalse(is_in_block_indexed_by(
            m.b2[m.time[1], m.space[1]].b.v1, 
            m.space, stop_at=m.b2[m.time[1], m.space[1]]))

        # Explicit indexing with multi-dimensional set:
        self.assertTrue(is_explicitly_indexed_by(m.v4, m.time, m.set2))
        self.assertTrue(is_explicitly_indexed_by(m.v5, m.time, m.set2, m.space))

        # Implicit indexing with multi-dimensional set:
        self.assertTrue(is_in_block_indexed_by(
            m.b3['a', 1, m.time[1]].v, m.set2))
        self.assertTrue(is_in_block_indexed_by(
            m.b3['a', 1, m.time[1]].v, m.time))
        self.assertTrue(is_in_block_indexed_by(
            m.b3['a', 1, m.time[1]].v1[m.space[1]], m.set2))
        self.assertFalse(is_in_block_indexed_by(
            m.b3['a', 1, m.time[1]].v1[m.space[1]], m.space))
        self.assertTrue(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.set2))
        self.assertTrue(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time))
        self.assertTrue(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.space))
        self.assertFalse(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.set))
        self.assertFalse(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time,
            stop_at=m.b3['b', 2, m.time[2]]))
        self.assertFalse(is_in_block_indexed_by(
            m.b3['b', 2, m.time[2]].b[m.space[2]].v['b'], m.time,
            stop_at=m.b3))


    # Test get_index_set_except and _complete_index
    def test_get_index_set_except(self):
        '''
        Tests:
          For components indexed by 0, 1, 2, 3, 4 sets:
            get_index_set_except one, then two (if any) of those sets
            check two items that should be in set_except
            insert item(s) back into these sets via index_getter
        '''
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0, 10))
        m.space = ContinuousSet(bounds=(0, 10))
        m.set1 = Set(initialize=['a', 'b', 'c'])
        m.set2 = Set(initialize=['d', 'e', 'f'])
        m.v = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.space)
        m.v3 = Var(m.time, m.space, m.set1)
        m.v4 = Var(m.time, m.space, m.set1, m.set2)

        # Multi-dimensional set:
        m.set3 = Set(initialize=[('a', 1), ('b', 2)])
        m.v5 = Var(m.set3)
        m.v6 = Var(m.time, m.space, m.set3)
        m.v7 = Var(m.set3, m.space, m.time)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
        disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')

        # Want this to give a TypeError
        # info = get_index_set_except(m.v, m.time)

        # Indexed by one set
        info = get_index_set_except(m.v1, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(set_except == [None])
        # Variable is not indexed by anything except time
        # Test that index_getter returns only the new value given,
        # regardless of whether it was part of the set excluded (time):
        self.assertEqual(index_getter((), -1), -1)

        # Indexed by two sets
        info = get_index_set_except(m.v2, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(m.space[1] in set_except
                        and m.space.last() in set_except)
	# Here (2,) is the partial index, corresponding to space.
        # Can be provided as a scalar or tuple. 4, the time index,
        # should be inserted before (2,)
        self.assertEqual(index_getter((2,), 4), (4, 2))
        self.assertEqual(index_getter(2, 4), (4, 2))

        # Case where every set is "omitted," now for multiple sets
        info = get_index_set_except(m.v2, m.space, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(set_except == [None])
        # 5, 7 are the desired index values for space, time 
        # index_getter should put them in the right order for m.v2,
        # even if they are not valid indices for m.v2
        self.assertEqual(index_getter((), 5, 7), (7, 5))

        # Indexed by three sets
        info = get_index_set_except(m.v3, m.time)
        # In this case set_except is a product of the two non-time sets
        # indexing v3
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue((m.space[1], 'b') in set_except
                        and (m.space.last(), 'a') in set_except)
        # index_getter inserts a scalar index into an index of length 2
        self.assertEqual(index_getter((2, 'b'), 7), (7, 2, 'b'))

        info = get_index_set_except(m.v3, m.space, m.time)
        # Two sets omitted. Now set_except is just set1
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue('a' in set_except)
        # index_getter inserts the two new indices in the right order
        self.assertEqual(index_getter('b', 1.2, 1.1), (1.1, 1.2, 'b'))

        # Indexed by four sets
        info = get_index_set_except(m.v4, m.set1, m.space)
        # set_except is a product, and there are two indices to insert
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue((m.time[1], 'd') in set_except)
        self.assertEqual(index_getter((4, 'f'), 'b', 8), (4, 8, 'b', 'f'))
        
        # The intended usage of this function looks something like:
        index_set = m.v4.index_set()
        for partial_index in set_except:
            complete_index = index_getter(partial_index, 'a', m.space[2])
            self.assertTrue(complete_index in index_set)
            # Do something for every index of v4 at 'a' and space[2]

        # Indexed by a multi-dimensional set
        info = get_index_set_except(m.v5, m.set3)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertEqual(set_except, [None])
        self.assertEqual(index_getter((), ('a', 1)), ('a', 1))

        info = get_index_set_except(m.v6, m.set3, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(m.space[1] in set_except)
        self.assertEqual(index_getter(m.space[1], ('b', 2), m.time[1]),
                (m.time[1], m.space[1], 'b', 2))

        info = get_index_set_except(m.v7, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertIn(('a', 1, m.space[1]), set_except)
        self.assertEqual(index_getter(('a', 1, m.space[1]), m.time[1]),
                         ('a', 1, m.space[1], m.time[1]))

        m.v8 = Var(m.time, m.set3, m.time)
        with self.assertRaises(ValueError):
            info = get_index_set_except(m.v8, m.time)
        with self.assertRaises(ValueError):
            info = get_index_set_except(m.v8, m.space)

    def test_deactivate_model_at(self):
        m = make_model()

        deactivate_model_at(m, m.time, m.time[2])
        self.assertTrue(m.fs.con1[m.time[1]].active)
        self.assertFalse(m.fs.con1[m.time[2]].active)
        self.assertTrue(m.fs.con2[m.space[1]].active)
        self.assertFalse(m.fs.b1.con[m.time[2], m.space[1]].active)
        self.assertFalse(m.fs.b2[m.time[2], m.space.last()].active)
        self.assertTrue(m.fs.b2[m.time[2], m.space.last()].b3['a'].con['e'].active)

        deactivate_model_at(m, m.time, [m.time[1], m.time[3]])
        # disc equations at time.first()
        self.assertFalse(m.fs.con1[m.time[1]].active)
        self.assertFalse(m.fs.con1[m.time[3]].active)
        self.assertFalse(m.fs.b1.con[m.time[1], m.space[1]].active)
        self.assertFalse(m.fs.b1.con[m.time[3], m.space[1]].active)

        with self.assertRaises(KeyError):
            deactivate_model_at(m, m.time, m.time[1], allow_skip=False,
                    suppress_warnings=True)


if __name__ == "__main__":
    unittest.main()
