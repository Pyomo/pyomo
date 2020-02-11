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
Unit Tests for pyomo.dae.misc
"""
import os
from os.path import abspath, dirname

from six import StringIO

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.common.log import LoggingIntercept
from pyomo.dae import *
from pyomo.dae.set_utils import *
from pyomo.core.kernel.component_map import ComponentMap

currdir = dirname(abspath(__file__)) + os.sep


class TestDaeSetUtils(unittest.TestCase):
    
    # Test explicit/implicit index detection functions
    def test_indexed_by(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0, 10))
        m.space = ContinuousSet(bounds=(0, 10))
        m.set = Set(initialize=['a', 'b', 'c'])
        m.v = Var()
        m.v1 = Var(m.time)
        m.v2 = Var(m.time, m.space)
        m.v3 = Var(m.set, m.space, m.time)

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

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, wrt=m.time, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
        disc.apply_to(m, wrt=m.space, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')

        self.assertFalse(is_explicitly_indexed_by(m.v, m.time))
        self.assertTrue(is_explicitly_indexed_by(m.b.v2, m.space))

        self.assertFalse(is_implicitly_indexed_by(m.v1, m.time))
        self.assertFalse(is_implicitly_indexed_by(m.v2, m.set))
        self.assertTrue(is_implicitly_indexed_by(m.b1[m.time[1]].v2, m.time))

        self.assertTrue(is_implicitly_indexed_by(m.b2[m.time[1], 
            m.space[1]].b.v1, m.time))
        self.assertEqual(is_implicitly_indexed_by(m.b2[m.time[1], 
            m.space[1]].b.v2, m.time),
            is_explicitly_indexed_by(m.b2[m.time[1], 
                m.space[1]].b.v2, m.time))
        self.assertFalse(is_implicitly_indexed_by(m.b2[m.time[1], 
            m.space[1]].b.v1, m.set))

        self.assertFalse(is_implicitly_indexed_by(m.b2[m.time[1],
            m.space[1]].b.v1, m.space, stop_at=m.b2[m.time[1], m.space[1]]))


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
        self.assertEqual(index_getter([5,6,7], 3j), 3j)

        # Indexed by two sets
        info = get_index_set_except(m.v2, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(m.space[1] in set_except
                        and m.space.last() in set_except)
        self.assertEqual(index_getter(2, 4), (4, 2), index_getter((2,), 4))

        info = get_index_set_except(m.v2, m.space, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue(set_except == [None])
        self.assertEqual(index_getter((), 5, 7), (7, 5))

        # Indexed by three sets
        info = get_index_set_except(m.v3, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue((m.space[1], 'b') in set_except
                        and (m.space.last(), 'a') in set_except)
        self.assertEqual(index_getter((2, 'b'), 7), (7, 2, 'b'))

        info = get_index_set_except(m.v3, m.space, m.time)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue('a' in set_except)
        self.assertTrue(index_getter('b', 1.2, 1.1), (1.1, 1.2, 'b'))

        # Indexed by four sets
        info = get_index_set_except(m.v4, m.set1, m.space)
        set_except = info['set_except']
        index_getter = info['index_getter']
        self.assertTrue((m.time[1], 'd') in set_except)
        self.assertTrue(index_getter((4, 'f'), 'b', 8), (4, 8, 'b', 'f'))


if __name__ == "__main__":
    unittest.main()
