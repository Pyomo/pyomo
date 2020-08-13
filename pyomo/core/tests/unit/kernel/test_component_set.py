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
from pyomo.common.collections import ComponentSet
from pyomo.core.kernel.variable import (variable,
                                        variable_dict,
                                        variable_list)
from pyomo.core.kernel.constraint import (constraint,
                                          constraint_dict,
                                          constraint_list)
from pyomo.core.kernel.objective import (objective,
                                         objective_dict,
                                         objective_list)
from pyomo.core.kernel.expression import (expression,
                                          expression_dict,
                                          expression_list)
from pyomo.core.kernel.block import (block,
                                     block_dict,
                                     block_list)
from pyomo.core.kernel.suffix import suffix

import six

if six.PY3:
    from collections.abc import Set as collections_Set
    from collections.abc import MutableSet as collections_MutableSet
else:
    from collections import Set as collections_Set
    from collections import MutableSet as collections_MutableSet


class TestComponentSet(unittest.TestCase):

    _components = [variable(),
                   variable_dict(),
                   variable_list(),
                   constraint(),
                   constraint_dict(),
                   constraint_list(),
                   objective(),
                   objective_dict(),
                   objective_list(),
                   expression(),
                   expression_dict(),
                   expression_list(),
                   block(),
                   block_dict(),
                   block_list(),
                   suffix()]

    def test_pickle(self):
        c = ComponentSet()
        self.assertEqual(len(c), 0)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertIsNot(cup, c)
        self.assertEqual(len(cup), 0)

        v = variable()
        c.add(v)
        self.assertEqual(len(c), 1)
        self.assertTrue(v in c)
        cup = pickle.loads(
            pickle.dumps(c))
        vup = cup.pop()
        cup.add(vup)
        self.assertIsNot(cup, c)
        self.assertIsNot(vup, v)
        self.assertEqual(len(cup), 1)
        self.assertTrue(vup in cup)
        self.assertEqual(vup.parent, None)

        b = block()
        V = b.V = variable_list()
        b.V.append(v)
        b.c = c
        self.assertEqual(len(c), 1)
        self.assertTrue(v in c)
        self.assertIs(v.parent, b.V)
        self.assertIs(V.parent, b)
        self.assertIs(b.parent, None)
        bup = pickle.loads(
            pickle.dumps(b))
        Vup = bup.V
        vup = Vup[0]
        cup = bup.c
        self.assertIsNot(cup, c)
        self.assertIsNot(vup, v)
        self.assertIsNot(Vup, V)
        self.assertIsNot(bup, b)
        self.assertEqual(len(cup), 1)
        self.assertTrue(vup in cup)
        self.assertIs(vup.parent, Vup)
        self.assertIs(Vup.parent, bup)
        self.assertIs(bup.parent, None)

        self.assertEqual(len(c), 1)
        self.assertTrue(v in c)

    def test_init(self):
        cset = ComponentSet()
        cset = ComponentSet(self._components)
        with self.assertRaises(TypeError):
            cset = ComponentSet(*self._components)

    def test_type(self):
        cset = ComponentSet()
        self.assertTrue(isinstance(cset, collections_Set))
        self.assertTrue(isinstance(cset, collections_MutableSet))
        self.assertTrue(issubclass(type(cset), collections_Set))
        self.assertTrue(issubclass(type(cset), collections_MutableSet))

    def test_str(self):
        cset = ComponentSet()
        self.assertEqual(str(cset), "ComponentSet([])")
        cset.update(self._components)
        str(cset)

    def test_len(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        cset.update(self._components)
        self.assertEqual(len(cset), len(self._components))
        cset = ComponentSet(self._components)
        self.assertEqual(len(cset), len(self._components))
        self.assertTrue(len(self._components) > 0)

    def test_iter(self):
        cset = ComponentSet()
        self.assertEqual(list(iter(cset)), [])
        cset.update(self._components)
        ids_seen = set()
        for c in cset:
            ids_seen.add(id(c))
        self.assertEqual(ids_seen,
                         set(id(c) for c in self._components))

    def set_add(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        for i, c in enumerate(self._components):
            self.assertTrue(c not in cset)
            cset.add(c)
            self.assertTrue(c in cset)
            self.assertEqual(len(cset), i+1)
        self.assertEqual(len(cset), len(self._components))
        for c in self._components:
            self.assertTrue(c in cset)
            cset.add(c)
            self.assertTrue(c in cset)
            self.assertEqual(len(cset), len(self._components))

    def test_pop(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        with self.assertRaises(KeyError):
            cset.pop()
        v = variable()
        cset.add(v)
        self.assertTrue(v in cset)
        self.assertEqual(len(cset), 1)
        v_ = cset.pop()
        self.assertIs(v, v_)
        self.assertTrue(v not in cset)
        self.assertEqual(len(cset), 0)

    def test_update(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        cset.update(self._components)
        self.assertEqual(len(cset), len(self._components))
        for c in self._components:
            self.assertTrue(c in cset)

    def test_clear(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        cset.update(self._components)
        self.assertEqual(len(cset), len(self._components))
        cset.clear()
        self.assertEqual(len(cset), 0)

    def test_remove(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        cset.update(self._components)
        self.assertEqual(len(cset), len(self._components))
        for i, c in enumerate(self._components):
            cset.remove(c)
            self.assertEqual(len(cset), len(self._components)-(i+1))
        for c in self._components:
            self.assertTrue(c not in cset)
            with self.assertRaises(KeyError):
                cset.remove(c)

    def test_discard(self):
        cset = ComponentSet()
        self.assertEqual(len(cset), 0)
        cset.update(self._components)
        self.assertEqual(len(cset), len(self._components))
        for i, c in enumerate(self._components):
            cset.discard(c)
            self.assertEqual(len(cset), len(self._components)-(i+1))
        for c in self._components:
            self.assertTrue(c not in cset)
            cset.discard(c)

    def test_isdisjoint(self):
        cset1 = ComponentSet()
        cset2 = ComponentSet()
        self.assertTrue(cset1.isdisjoint(cset2))
        self.assertTrue(cset2.isdisjoint(cset1))
        v = variable()
        cset1.add(v)
        self.assertTrue(cset1.isdisjoint(cset2))
        self.assertTrue(cset2.isdisjoint(cset1))
        cset2.add(v)
        self.assertFalse(cset1.isdisjoint(cset2))
        self.assertFalse(cset2.isdisjoint(cset1))

    def test_misc_set_ops(self):
        v1 = variable()
        cset1 = ComponentSet([v1])
        v2 = variable()
        cset2 = ComponentSet([v2])
        cset3 = ComponentSet([v1,v2])
        empty = ComponentSet([])
        self.assertEqual(cset1 | cset2, cset3)
        self.assertEqual((cset1 | cset2) - cset3, empty)
        self.assertEqual(cset1 ^ cset2, cset3)
        self.assertEqual(cset1 ^ cset3, cset2)
        self.assertEqual(cset2 ^ cset3, cset1)
        self.assertEqual(cset1 & cset2, empty)
        self.assertEqual(cset1 & cset3, cset1)
        self.assertEqual(cset2 & cset3, cset2)

    def test_eq(self):
        cset1 = ComponentSet()
        self.assertEqual(cset1, set())
        self.assertTrue(cset1 == set())
        self.assertNotEqual(cset1, list())
        self.assertFalse(cset1 == list())
        self.assertNotEqual(cset1, tuple())
        self.assertFalse(cset1 == tuple())
        self.assertNotEqual(cset1, dict())
        self.assertFalse(cset1 == dict())

        cset1.update(self._components)
        self.assertNotEqual(cset1, set())
        self.assertFalse(cset1 == set())
        self.assertNotEqual(cset1, list())
        self.assertFalse(cset1 == list())
        self.assertNotEqual(cset1, tuple())
        self.assertFalse(cset1 == tuple())
        self.assertNotEqual(cset1, dict())
        self.assertFalse(cset1 == dict())

        self.assertTrue(cset1 == cset1)
        self.assertEqual(cset1, cset1)

        cset2 = ComponentSet(self._components)
        self.assertTrue(cset2 == cset1)
        self.assertFalse(cset2 != cset1)
        self.assertEqual(cset2, cset1)
        self.assertTrue(cset1 == cset2)
        self.assertFalse(cset1 != cset2)
        self.assertEqual(cset1, cset2)

        cset2.remove(self._components[0])
        self.assertFalse(cset2 == cset1)
        self.assertTrue(cset2 != cset1)
        self.assertNotEqual(cset2, cset1)
        self.assertFalse(cset1 == cset2)
        self.assertTrue(cset1 != cset2)
        self.assertNotEqual(cset1, cset2)


if __name__ == "__main__":
    unittest.main()
