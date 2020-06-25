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

from pyomo.common.collections import OrderedSet

class testOrderedSet(unittest.TestCase):
    def test_constructor(self):
        a = OrderedSet()
        self.assertEqual(len(a), 0)
        self.assertEqual(list(a), [])
        self.assertEqual(str(a), 'OrderedSet()')

        ref = [1,9,'a',4,2,None]
        a = OrderedSet(ref)
        self.assertEqual(len(a), 6)
        self.assertEqual(list(a), ref)
        self.assertEqual(str(a), "OrderedSet(1, 9, 'a', 4, 2, None)")

    def test_in_add(self):
        a = OrderedSet()
        self.assertNotIn(1, a)
        self.assertNotIn(None, a)

        a.add(None)
        self.assertNotIn(1, a)
        self.assertIn(None, a)

        a.add(1)
        self.assertIn(1, a)
        self.assertIn(None, a)

        a.add(0)
        self.assertEqual(list(a), [None,1,0])

        # Adding a member alrady in the set does not change the ordering
        a.add(1)
        self.assertEqual(list(a), [None,1,0])

    def test_discard_remove_clear(self):
        a = OrderedSet([1,3,2,4])
        a.discard(3)
        self.assertEqual(list(a), [1,2,4])
        a.discard(3)
        self.assertEqual(list(a), [1,2,4])

        a.remove(2)
        self.assertEqual(list(a), [1,4])
        with self.assertRaisesRegex(KeyError,'2'):
            a.remove(2)
        
        a.clear()
        self.assertEqual(list(a), [])

    def test_pickle(self):
        ref = [1,9,'a',4,2,None]
        a = OrderedSet(ref)
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)
        self.assertIsNot(a, b)
        self.assertIsNot(a._dict, b._dict)
