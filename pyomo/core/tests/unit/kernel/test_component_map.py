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
from pyomo.common.collections import ComponentMap
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
    from collections.abc import Mapping as collections_Mapping
    from collections.abc import MutableMapping as collections_MutableMapping
else:
    from collections import Mapping as collections_Mapping
    from collections import MutableMapping as collections_MutableMapping


class TestComponentMap(unittest.TestCase):

    _components = [(variable(), "v"),
                   (variable_dict(), "vdict"),
                   (variable_list(), "vlist"),
                   (constraint(), "c"),
                   (constraint_dict(), "cdict"),
                   (constraint_list(), "clist"),
                   (objective(), "o"),
                   (objective_dict(), "odict"),
                   (objective_list(), "olist"),
                   (expression(), "e"),
                   (expression_dict(), "edict"),
                   (expression_list(), "elist"),
                   (block(), "b"),
                   (block_dict(), "bdict"),
                   (block_list(), "blist"),
                   (suffix(), "s")]

    def test_pickle(self):
        c = ComponentMap()
        self.assertEqual(len(c), 0)
        cup = pickle.loads(
            pickle.dumps(c))
        self.assertIsNot(cup, c)
        self.assertEqual(len(cup), 0)

        v = variable()
        c[v] = 1.0
        self.assertEqual(len(c), 1)
        self.assertEqual(c[v], 1.0)
        cup = pickle.loads(
            pickle.dumps(c))
        vup = list(cup.keys())[0]
        self.assertIsNot(cup, c)
        self.assertIsNot(vup, v)
        self.assertEqual(len(cup), 1)
        self.assertEqual(cup[vup], 1)
        self.assertEqual(vup.parent, None)

        b = block()
        V = b.V = variable_list()
        b.V.append(v)
        b.c = c
        self.assertEqual(len(c), 1)
        self.assertEqual(c[v], 1.0)
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
        self.assertEqual(cup[vup], 1)
        self.assertIs(vup.parent, Vup)
        self.assertIs(Vup.parent, bup)
        self.assertIs(bup.parent, None)

        self.assertEqual(len(c), 1)
        self.assertEqual(c[v], 1)

    def test_init(self):
        cmap = ComponentMap()
        cmap = ComponentMap(self._components)
        with self.assertRaises(TypeError):
            cmap = ComponentMap(*self._components)

    def test_type(self):
        cmap = ComponentMap()
        self.assertTrue(isinstance(cmap, collections_Mapping))
        self.assertTrue(isinstance(cmap, collections_MutableMapping))
        self.assertTrue(issubclass(type(cmap), collections_Mapping))
        self.assertTrue(issubclass(type(cmap), collections_MutableMapping))

    def test_str(self):
        cmap = ComponentMap()
        self.assertEqual(str(cmap), "ComponentMap({})")
        cmap.update(self._components)
        str(cmap)

    def test_len(self):
        cmap = ComponentMap()
        self.assertEqual(len(cmap), 0)
        cmap.update(self._components)
        self.assertEqual(len(cmap), len(self._components))
        cmap = ComponentMap(self._components)
        self.assertEqual(len(cmap), len(self._components))
        self.assertTrue(len(self._components) > 0)

    def test_getsetdelitem(self):
        cmap = ComponentMap()
        for c, val in self._components:
            self.assertTrue(c not in cmap)
        for c, val in self._components:
            cmap[c] = val
            self.assertEqual(cmap[c], val)
            self.assertEqual(cmap.get(c), val)
            del cmap[c]
            with self.assertRaises(KeyError):
                cmap[c]
            with self.assertRaises(KeyError):
                del cmap[c]
            self.assertEqual(cmap.get(c), None)

    def test_iter(self):
        cmap = ComponentMap()
        self.assertEqual(list(iter(cmap)), [])
        cmap.update(self._components)
        ids_seen = set()
        for c in cmap:
            ids_seen.add(id(c))
        self.assertEqual(ids_seen,
                         set(id(c) for c,val in self._components))

    def test_keys(self):
        cmap = ComponentMap(self._components)
        self.assertEqual(sorted(cmap.keys(), key=id),
                         sorted(list(c for c,val in self._components),
                                key=id))

    def test_values(self):
        cmap = ComponentMap(self._components)
        self.assertEqual(sorted(cmap.values()),
                         sorted(list(val for c,val in self._components)))

    def test_items(self):
        cmap = ComponentMap(self._components)
        for x in cmap.items():
            self.assertEqual(type(x), tuple)
            self.assertEqual(len(x), 2)
        self.assertEqual(sorted(cmap.items(),
                                key=lambda _x: (id(_x[0]), _x[1])),
                         sorted(self._components,
                                key=lambda _x: (id(_x[0]), _x[1])))

    def test_update(self):
        cmap = ComponentMap()
        self.assertEqual(len(cmap), 0)
        cmap.update(self._components)
        self.assertEqual(len(cmap), len(self._components))
        for c, val in self._components:
            self.assertEqual(cmap[c], val)

    def test_clear(self):
        cmap = ComponentMap()
        self.assertEqual(len(cmap), 0)
        cmap.update(self._components)
        self.assertEqual(len(cmap), len(self._components))
        cmap.clear()
        self.assertEqual(len(cmap), 0)

    def test_setdefault(self):
        cmap = ComponentMap()
        for c,_ in self._components:
            with self.assertRaises(KeyError):
                cmap[c]
            self.assertTrue(c not in cmap)
            cmap.setdefault(c, []).append(1)
            self.assertEqual(cmap[c], [1])
            del cmap[c]
            with self.assertRaises(KeyError):
                cmap[c]
            self.assertTrue(c not in cmap)
            cmap[c] = []
            cmap.setdefault(c, []).append(1)
            self.assertEqual(cmap[c], [1])

    def test_eq(self):
        cmap1 = ComponentMap()
        self.assertNotEqual(cmap1, set())
        self.assertFalse(cmap1 == set())
        self.assertNotEqual(cmap1, list())
        self.assertFalse(cmap1 == list())
        self.assertNotEqual(cmap1, tuple())
        self.assertFalse(cmap1 == tuple())
        self.assertEqual(cmap1, dict())
        self.assertTrue(cmap1 == dict())

        cmap1.update(self._components)
        self.assertNotEqual(cmap1, set())
        self.assertFalse(cmap1 == set())
        self.assertNotEqual(cmap1, list())
        self.assertFalse(cmap1 == list())
        self.assertNotEqual(cmap1, tuple())
        self.assertFalse(cmap1 == tuple())
        self.assertNotEqual(cmap1, dict())
        self.assertFalse(cmap1 == dict())

        self.assertTrue(cmap1 == cmap1)
        self.assertEqual(cmap1, cmap1)

        cmap2 = ComponentMap(self._components)
        self.assertTrue(cmap2 == cmap1)
        self.assertFalse(cmap2 != cmap1)
        self.assertEqual(cmap2, cmap1)
        self.assertTrue(cmap1 == cmap2)
        self.assertFalse(cmap1 != cmap2)
        self.assertEqual(cmap1, cmap2)

        del cmap2[self._components[0][0]]
        self.assertFalse(cmap2 == cmap1)
        self.assertTrue(cmap2 != cmap1)
        self.assertNotEqual(cmap2, cmap1)
        self.assertFalse(cmap1 == cmap2)
        self.assertTrue(cmap1 != cmap2)
        self.assertNotEqual(cmap1, cmap2)


if __name__ == "__main__":
    unittest.main()
