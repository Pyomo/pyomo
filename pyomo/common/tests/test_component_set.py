#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle
import pyomo.common.unittest as unittest

from pyomo.common.collections._hasher import _HashKey
from pyomo.common.collections.component_set import ComponentSet, ObjectIdSet
from pyomo.environ import ConcreteModel, Var, Constraint


class ComponentSetBaseTests:

    def test_str(self):
        m = ConcreteModel()
        m.x = Var()
        cs = self.CS()
        cs.add(m.x)
        _id = id(m.x)
        cs.add(_id)
        cs.add((5, m.x))
        self.assertEqual(f"{self.CS.__name__}" f"(x, {_id}, (5, x))", str(cs))

    def test_add_remove_discard(self):
        m = ConcreteModel()
        m.x = Var()

        cs = self.CS()
        cs.add(m)
        cs.add(m.x)
        cs.add(3)
        self.assertEqual(3, len(cs))
        self.assertIn(m, cs)
        self.assertIn(m.x, cs)
        self.assertIn(3, cs)

        # Re-adding doesn't change anything
        cs.add(m)
        cs.add(m.x)
        cs.add(3)
        self.assertEqual(3, len(cs))
        self.assertIn(m, cs)
        self.assertIn(m.x, cs)
        self.assertIn(3, cs)

        cs.remove(m.x)
        self.assertEqual(2, len(cs))
        self.assertIn(m, cs)
        self.assertIn(3, cs)

        with self.assertRaisesRegex(KeyError, repr(m.x)):
            cs.remove(m.x)
        cs.discard(m.x)
        self.assertEqual(2, len(cs))
        self.assertIn(m, cs)
        self.assertIn(3, cs)

        cs.discard(m)
        self.assertEqual(1, len(cs))
        self.assertIn(3, cs)

    def test_iter(self):
        m = ConcreteModel()
        m.x = Var()

        cs = self.CS([m, m.x, 3])
        self.assertEqual([m, m.x, 3], list(cs))

    def test_eq(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c = Constraint()

        cs1 = self.CS([m, m.x, m.c])
        self.assertEqual(cs1, cs1)

        cs2 = self.CS([m, m.c])
        self.assertNotEqual(cs1, cs2)

        cs2.add(m.y)
        self.assertNotEqual(cs1, cs2)

        cs2.remove(m.y)
        cs2.add(m.x)
        self.assertEqual(cs1, cs2)

        self.assertNotEqual(cs1, {m, m.c})
        cs1.remove(m.x)
        self.assertEqual(cs1, {m, m.c})

    def test_clear(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cs1 = self.CS([m, m.x, m.c])
        cs2 = self.CS(cs1)
        self.assertEqual(cs1, cs2)
        cs1.clear()
        self.assertNotEqual(cs1, cs2)
        self.assertEqual(cs1, set())

    def test_init_update(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cs1 = self.CS([m, m.x, m.c])

        cs2 = self.CS(cs1)
        self.assertIsNot(cs1, cs2)
        self.assertIsNot(cs1._data, cs2._data)
        self.assertEqual(cs1, cs2)

        cs3 = self.CS({m, m.c})
        cs2.discard(m.x)
        self.assertEqual(cs2, cs3)

        cs3.update(cs1)
        self.assertNotEqual(cs2, cs3)
        self.assertEqual(cs1, cs3)


class TestComponentSet(ComponentSetBaseTests, unittest.TestCase):
    def setUp(self):
        self.CS = ComponentSet

    def test_pickle(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cs = self.CS([1, m.x, (1, (2, m.x)), m.c])
        m.cs = cs

        i = pickle.loads(pickle.dumps(m))
        self.assertIsNot(i, m)
        self.assertIsNot(i.cs, m.cs)
        self.assertIn(1, i.cs)
        self.assertNotIn(m.x, i.cs)
        self.assertIn(i.x, i.cs)
        self.assertNotIn((1, (2, m.x)), i.cs)
        self.assertIn((1, (2, i.x)), i.cs)
        self.assertNotIn(m.c, i.cs)
        self.assertIn(i.c, i.cs)

        _items = iter(i.cs._data.items())
        k, v = next(_items)
        self.assertEqual(k, 1)
        self.assertEqual(v, 1)
        k, v = next(_items)
        self.assertEqual(k, (_HashKey, id(i.x)))
        self.assertEqual(v, i.x)
        k, v = next(_items)
        self.assertEqual(k, (1, (2, (_HashKey, id(i.x)))))
        self.assertEqual(v, (1, (2, i.x)))
        k, v = next(_items)
        self.assertEqual(k, i.c)
        self.assertEqual(v, i.c)


class TestObjectIdSet(ComponentSetBaseTests, unittest.TestCase):
    def setUp(self):
        self.CS = ObjectIdSet

    def test_pickle(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cs = self.CS([1, m.x, (1, (2, m.x)), m.c])
        m.cs = cs

        i = pickle.loads(pickle.dumps(m))
        self.assertIsNot(i, m)
        self.assertIsNot(i.cs, m.cs)
        self.assertIn(1, i.cs)  # Note: different from ComponentMap
        self.assertNotIn(m.x, i.cs)
        self.assertIn(i.x, i.cs)
        self.assertNotIn((1, (2, m.x)), i.cs)
        self.assertNotIn((1, (2, i.x)), i.cs)  # Note: different from ComponentMap
        self.assertNotIn(m.c, i.cs)
        self.assertIn(i.c, i.cs)

        _items = iter(i.cs._data.items())
        k, v = next(_items)
        self.assertEqual(k, id(v))
        self.assertEqual(v, 1)
        k, v = next(_items)
        self.assertEqual(k, id(i.x))
        self.assertEqual(v, i.x)
        k, v = next(_items)
        self.assertEqual(k, id(v))
        self.assertEqual(v, (1, (2, i.x)))
        k, v = next(_items)
        self.assertEqual(k, id(i.c))
        self.assertEqual(v, i.c)
