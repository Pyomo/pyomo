# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pickle
import pyomo.common.unittest as unittest

from pyomo.common.collections._hasher import HashKey
from pyomo.common.collections.component_map import (
    ComponentMap,
    DefaultComponentMap,
    ObjectIdMap,
)
from pyomo.common.collections.component_set import ComponentSet
from pyomo.environ import ConcreteModel, Block, Var, Constraint, Param


class ComponentMapBaseTests:

    def test_str(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Param([1], mutable=True)
        cm = self.CM()
        cm[m.x] = m.y[1]
        _id = id(m.x)
        cm[_id] = 42
        cm[(5, m.x)] = 7
        self.assertEqual(
            f"{self.CM.__name__}" f"(x: y[1], {_id}: 42, (5, x): 7)", str(cm)
        )

    def test_get_del_item(self):
        m = ConcreteModel()
        m.x = Var()

        cm = self.CM()
        cm[m] = 10
        cm[m.x] = 20
        cm[3] = 30
        self.assertEqual(3, len(cm))
        self.assertEqual(cm[m], 10)
        self.assertEqual(cm[m.x], 20)
        self.assertEqual(cm[3], 30)

        del cm[m.x]
        self.assertEqual(2, len(cm))
        self.assertEqual(cm[m], 10)
        self.assertEqual(cm[3], 30)

        self.assertEqual(cm.get(m), 10)
        self.assertEqual(cm.get(m, 100), 10)
        self.assertEqual(cm.get(m.x), None)
        self.assertEqual(cm.get(m.x, 100), 100)
        self.assertEqual(cm.get(3), 30)
        self.assertEqual(cm.get(3, 100), 30)

        with self.assertRaisesRegex(KeyError, repr(m.x)):
            cm[m.x]

        with self.assertRaisesRegex(KeyError, repr(m.x)):
            del cm[m.x]

        self.assertEqual(2, len(cm))
        self.assertEqual(cm[m], 10)
        self.assertEqual(cm[3], 30)

    def test_iters(self):
        m = ConcreteModel()
        m.x = Var()

        cm = self.CM()
        cm[m] = 10
        cm[m.x] = 20
        cm[3] = 10

        self.assertEqual([m, m.x, 3], list(cm))

        k = cm.keys()
        self.assertEqual([m, m.x, 3], list(k))
        self.assertEqual(3, len(k))
        self.assertIn(m, k)
        self.assertIn(m.x, k)
        self.assertNotIn(4, k)

        v = cm.values()
        self.assertEqual([10, 20, 10], list(v))
        self.assertEqual(3, len(v))
        self.assertIn(10, v)
        self.assertIn(20, v)
        self.assertNotIn(30, v)

        i = cm.items()
        self.assertEqual([(m, 10), (m.x, 20), (3, 10)], list(i))
        self.assertEqual(3, len(i))
        self.assertIn((m, 10), i)
        self.assertIn((m.x, 20), i)
        self.assertIn((3, 10), i)
        self.assertNotIn((3, 30), i)
        self.assertNotIn((4, 10), i)
        self.assertNotIn('hi', i)
        self.assertNotIn(50, i)
        self.assertNotIn((1, 2, 3), i)

        # These are views... and should update to reflect the current state
        del cm[m]

        self.assertEqual([m.x, 3], list(k))
        self.assertEqual(2, len(k))
        self.assertNotIn(m, k)
        self.assertIn(m.x, k)
        self.assertNotIn(4, k)

        self.assertEqual([20, 10], list(v))
        self.assertEqual(2, len(v))
        self.assertIn(10, v)
        self.assertIn(20, v)
        self.assertNotIn(30, v)

        self.assertEqual([(m.x, 20), (3, 10)], list(i))
        self.assertEqual(2, len(i))
        self.assertNotIn((m, 10), i)
        self.assertIn((m.x, 20), i)
        self.assertIn((3, 10), i)
        self.assertNotIn((3, 30), i)
        self.assertNotIn((4, 10), i)
        self.assertNotIn('hi', i)
        self.assertNotIn(50, i)
        self.assertNotIn((1, 2, 3), i)

    def test_eq(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c = Constraint()

        cm1 = self.CM()
        cm1[m] = 10
        cm1[m.x] = 20
        cm1[m.c] = 30

        self.assertEqual(cm1, cm1)

        cm2 = self.CM()
        cm2[m] = 10
        cm2[m.c] = 30
        self.assertNotEqual(cm1, cm2)

        cm2[m.y] = 20
        self.assertNotEqual(cm1, cm2)

        del cm2[m.y]
        cm2[m.x] = 20
        self.assertEqual(cm1, cm2)

        self.assertNotEqual(cm1, {m: 10, m.c: 30})
        del cm1[m.x]
        self.assertEqual(cm1, {m: 10, m.c: 30})
        self.assertNotEqual(cm1, {m: 10, m.c: 40})

    def test_init_update(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cm1 = self.CM()
        cm1[m] = 10
        cm1[m.x] = 20
        cm1[m.c] = 30

        cm2 = self.CM(cm1)
        self.assertIsNot(cm1, cm2)
        self.assertIsNot(cm1._dict, cm2._dict)
        self.assertEqual(cm1, cm2)

        cm3 = self.CM({m: 10, m.c: 30})
        del cm2[m.x]
        self.assertEqual(cm2, cm3)

        cm3.update(cm1)
        self.assertNotEqual(cm2, cm3)
        self.assertEqual(cm1, cm3)

    def test_set_default(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cm = self.CM()
        self.assertIs(cm.setdefault(m, m.x), m.x)
        self.assertEqual(cm, {m: m.x})
        self.assertIs(cm.setdefault(m, m.c), m.x)
        self.assertEqual(cm, {m: m.x})

        cm.clear()
        self.assertEqual(cm, {})
        self.assertIs(cm.setdefault(m, m.c), m.c)
        self.assertEqual(cm, {m: m.c})


class TestComponentMap(ComponentMapBaseTests, unittest.TestCase):
    def setUp(self):
        self.CM = ComponentMap

    def test_hasher(self):
        m = self.CM()
        a = 'str'
        m[a] = 5
        self.assertTrue(m.hasher.hashable(a))
        self.assertTrue(m.hasher.hashable(str))
        self.assertEqual(m._dict, {a: (a, 5)})
        del m[a]

        m.hasher.hashable(a, False)
        m[a] = 5
        self.assertFalse(m.hasher.hashable(a))
        self.assertFalse(m.hasher.hashable(str))
        self.assertEqual(m._dict, {HashKey(a): (a, 5)})

        class TMP:
            pass

        with self.assertRaises(KeyError):
            m.hasher.hashable(TMP)

    def test_tuple(self):
        m = ConcreteModel()
        m.v = Var()
        m.c = Constraint(expr=m.v >= 0)
        m.cm = cm = self.CM()

        cm[(1, 2)] = 5
        self.assertEqual(len(cm), 1)
        self.assertIn((1, 2), cm)
        self.assertEqual(cm[1, 2], 5)

        cm[(1, 2)] = 50
        self.assertEqual(len(cm), 1)
        self.assertIn((1, 2), cm)
        self.assertEqual(cm[1, 2], 50)

        cm[(1, (2, m.v))] = 10
        self.assertEqual(len(cm), 2)
        self.assertIn((1, (2, m.v)), cm)
        self.assertEqual(cm[1, (2, m.v)], 10)

        cm[(1, (2, m.v))] = 100
        self.assertEqual(len(cm), 2)
        self.assertIn((1, (2, m.v)), cm)
        self.assertEqual(cm[1, (2, m.v)], 100)

        i = m.clone()
        self.assertIn((1, 2), i.cm)
        self.assertIn((1, (2, i.v)), i.cm)
        self.assertNotIn((1, (2, i.v)), m.cm)
        self.assertIn((1, (2, m.v)), m.cm)
        self.assertNotIn((1, (2, m.v)), i.cm)

    def test_id_int_collision(self):
        m = ConcreteModel()
        m.x = Var()
        cm = self.CM()

        cm[m.x] = 1
        cm[id(m.x)] = 2
        self.assertEqual(len(cm), 2)
        self.assertIn(m.x, cm)
        self.assertIn(id(m.x), cm)  # Note: different from ObjectIdMap
        self.assertEqual(cm[m.x], 1)

        a = (1, (m.x, 3))
        b = (1, (m.x, 3))
        self.assertNotEqual(id(a), id(b))

        cm[a] = 3
        cm[b] = 4
        self.assertEqual(len(cm), 3)  # Note: different from ObjectIdMap
        self.assertIn(a, cm)
        self.assertIn(b, cm)
        self.assertEqual(cm[a], 4)  # Note: different from ObjectIdMap
        self.assertEqual(cm[b], 4)
        self.assertIn((1, (m.x, 3)), cm)  # Note: different from ObjectIdMap

    def test_pickle(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cm = self.CM()
        cm[1] = 10
        cm[m.x] = 20
        cm[(1, (2, m.x))] = 30
        cm[m.c] = 40
        m.cm = cm

        i = pickle.loads(pickle.dumps(m))
        self.assertIsNot(i, m)
        self.assertIsNot(i.cm, m.cm)
        self.assertIn(1, i.cm)
        self.assertEqual(i.cm[1], 10)
        self.assertNotIn(m.x, i.cm)
        self.assertIn(i.x, i.cm)
        self.assertEqual(i.cm[i.x], 20)
        self.assertNotIn((1, (2, m.x)), i.cm)
        self.assertIn((1, (2, i.x)), i.cm)
        self.assertEqual(i.cm[(1, (2, i.x))], 30)
        self.assertNotIn(m.c, i.cm)
        self.assertIn(i.c, i.cm)
        self.assertEqual(i.cm[i.c], 40)

        _items = iter(i.cm._dict.items())
        k, v = next(_items)
        self.assertEqual(k, 1)
        self.assertEqual(v, (1, 10))
        k, v = next(_items)
        self.assertEqual(k, HashKey(i.x))
        self.assertEqual(v, (i.x, 20))
        self.assertEqual(k._hash, id(i.x))
        k, v = next(_items)
        self.assertEqual(k, (1, (2, HashKey(i.x))))
        self.assertEqual(v, ((1, (2, i.x)), 30))
        self.assertEqual(k[1][1]._hash, id(i.x))
        k, v = next(_items)
        self.assertEqual(k, i.c)
        self.assertEqual(v, (i.c, 40))


class TestDefaultComponentMap(ComponentMapBaseTests, unittest.TestCase):
    def setUp(self):
        self.CM = DefaultComponentMap

    def test_default_component_map(self):
        dcm = self.CM(ComponentSet)

        m = ConcreteModel()
        m.x = Var()
        m.b = Block()
        m.b.y = Var()

        self.assertEqual(len(dcm), 0)

        dcm[m.x].add(m)
        self.assertEqual(len(dcm), 1)
        self.assertIn(m.x, dcm)
        self.assertIn(m, dcm[m.x])

        dcm[m.b.y].add(m.b)
        self.assertEqual(len(dcm), 2)
        self.assertIn(m.b.y, dcm)
        self.assertNotIn(m, dcm[m.b.y])
        self.assertIn(m.b, dcm[m.b.y])

        dcm[m.b.y].add(m)
        self.assertEqual(len(dcm), 2)
        self.assertIn(m.b.y, dcm)
        self.assertIn(m, dcm[m.b.y])
        self.assertIn(m.b, dcm[m.b.y])

    def test_no_default_factory(self):
        dcm = self.CM()

        dcm['found'] = 5
        self.assertEqual(len(dcm), 1)
        self.assertIn('found', dcm)
        self.assertEqual(dcm['found'], 5)

        with self.assertRaisesRegex(KeyError, "'missing'"):
            dcm["missing"]


class TestObjectIdMap(ComponentMapBaseTests, unittest.TestCase):
    def setUp(self):
        self.CM = ObjectIdMap

    def test_str(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Param([1], mutable=True)
        cm = self.CM()
        cm[m.x] = m.y[1]
        _id = id(m.x)
        cm[_id] = 42
        _idid = id(_id)
        _tup = (5, m.x)
        cm[_tup] = 7
        self.assertEqual(
            f"{self.CM.__name__}"
            f"(x (key={_id}): y[1], {_id} (key={_idid}): 42, {_tup} (key={id(_tup)}): 7)",
            str(cm),
        )

    def test_id_int_collision(self):
        m = ConcreteModel()
        m.x = Var()
        cm = self.CM()

        cm[m.x] = 1
        cm[id(m.x)] = 2
        self.assertEqual(len(cm), 2)
        self.assertIn(m.x, cm)
        self.assertNotIn(id(m.x), cm)  # Note: different from ComponentMap
        self.assertEqual(cm[m.x], 1)

        a = (1, (m.x, 3))
        b = (1, (m.x, 3))
        self.assertNotEqual(id(a), id(b))

        cm[a] = 3
        cm[b] = 4
        self.assertEqual(len(cm), 4)  # Note: different from ComponentMap
        self.assertIn(a, cm)
        self.assertIn(b, cm)
        self.assertEqual(cm[a], 3)  # Note: different from ComponentMap
        self.assertEqual(cm[b], 4)
        self.assertNotIn((1, (m.x, 3)), cm)  # Note: different from ComponentMap

    def test_pickle(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()

        cm = self.CM()
        cm[1] = 10
        cm[m.x] = 20
        cm[(1, (2, m.x))] = 30
        cm[m.c] = 40
        m.cm = cm

        i = pickle.loads(pickle.dumps(m))
        self.assertIsNot(i, m)
        self.assertIsNot(i.cm, m.cm)
        self.assertIn(1, i.cm)
        self.assertEqual(i.cm[1], 10)
        self.assertNotIn(m.x, i.cm)
        self.assertIn(i.x, i.cm)
        self.assertEqual(i.cm[i.x], 20)
        self.assertNotIn((1, (2, m.x)), i.cm)
        self.assertNotIn((1, (2, i.x)), i.cm)  # Note: different from ComponentMap
        self.assertNotIn(m.c, i.cm)
        self.assertIn(i.c, i.cm)
        self.assertEqual(i.cm[i.c], 40)

        _items = iter(i.cm._dict.items())
        k, v = next(_items)
        self.assertEqual(k, id(1))
        self.assertEqual(v, (1, 10))
        k, v = next(_items)
        self.assertEqual(k, id(i.x))
        self.assertEqual(v, (i.x, 20))
        k, v = next(_items)
        self.assertEqual(k, id(v[0]))
        self.assertEqual(v, ((1, (2, i.x)), 30))
        k, v = next(_items)
        self.assertEqual(k, id(i.c))
        self.assertEqual(v, (i.c, 40))
