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

import pyomo.common.unittest as unittest

from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.environ import ConcreteModel, Block, Var, Constraint


class TestComponentMap(unittest.TestCase):
    def test_tuple(self):
        m = ConcreteModel()
        m.v = Var()
        m.c = Constraint(expr=m.v >= 0)
        m.cm = cm = ComponentMap()

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

    def test_hasher(self):
        m = ComponentMap()
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
        self.assertEqual(m._dict, {id(a): (a, 5)})

        class TMP:
            pass

        with self.assertRaises(KeyError):
            m.hasher.hashable(TMP)


class TestDefaultComponentMap(unittest.TestCase):
    def test_default_component_map(self):
        dcm = DefaultComponentMap(ComponentSet)

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
        dcm = DefaultComponentMap()

        dcm['found'] = 5
        self.assertEqual(len(dcm), 1)
        self.assertIn('found', dcm)
        self.assertEqual(dcm['found'], 5)

        with self.assertRaisesRegex(KeyError, "'missing'"):
            dcm["missing"]
