#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_block import (block,
                                             block_dict)

#
# There are no fully implemented test suites in this
# file. These classes are meant to be used to test
# full implementations of ComponentDict containers.
# To test a ComponentDict implementation, just import
# this class and use it as a subclass in another test suite.
#

# Note: we need to test with a pickle protocol
#       that is high enough to support __slots__
#       and weakref (bas
_pickle_test_protocol = pickle.HIGHEST_PROTOCOL

class _bad_ctype(object):
    ctype = "_this_is_definitely_not_the_ctype_being_tested"

class _TestComponentDictBase(object):

    # set by derived class
    _container_type = None
    _ctype_factory = None

    def setUp(self):
        self.model = block()

    def tearDown(self):
        self.model = None
        self._arg = None

    def test_init1(self):
        model = self.model
        model.c = self._container_type()

    def test_init2(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        with self.assertRaises(TypeError):
            model.d = \
                self._container_type(*tuple((i, self._ctype_factory())
                                   for i in index))

    def test_len1(self):
        model = self.model
        model = block()
        model.c = self._container_type()
        self.assertEqual(len(model.c), 0)

    def test_len2(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))

    def test_setitem(self):
        model = self.model
        model = block()
        model.c = self._container_type()
        index = ['a', 1, None, (1,), (1,2)]
        for i in index:
            self.assertTrue(i not in model.c)
        for cnt, i in enumerate(index, 1):
            model.c[i] = self._ctype_factory()
            self.assertEqual(len(model.c), cnt)
            self.assertTrue(i in model.c)

    # The immediately following this one was originally written when
    # implicit assignment for update was supported. This should be
    # examined more carefully before supporting it.
    # For now just test that implicit assignment raises an exception
    def test_wrong_type_init(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        with self.assertRaises(TypeError):
            model.c = self._container_type(
                (i, _bad_ctype()) for i in index)

    def test_wrong_type_update(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type()
        with self.assertRaises(TypeError):
            model.c.update((i, _bad_ctype()) for i in index)

    def test_wrong_type_setitem(self):
        model = self.model
        model.c = self._container_type()
        with self.assertRaises(TypeError):
            model.c[1] = _bad_ctype()
        model.c[1] = self._ctype_factory()
        with self.assertRaises(TypeError):
            model.c[1] = _bad_ctype()

    def test_has_parent_init(self):
        model = self.model
        model.c = self._container_type()
        model.c[1] = self._ctype_factory()
        with self.assertRaises(ValueError):
            model.d = self._container_type(model.c)
        with self.assertRaises(ValueError):
            model.d = self._container_type(dict(model.c))

    def test_has_parent_update(self):
        model = self.model
        model.c = self._container_type()
        model.c[1] = self._ctype_factory()
        model.c.update(model.c)
        self.assertEqual(len(model.c), 1)
        model.c.update(dict(model.c))
        self.assertEqual(len(model.c), 1)
        model.d = self._container_type()
        with self.assertRaises(ValueError):
            model.d.update(model.c)
        with self.assertRaises(ValueError):
            model.d.update(dict(model.c))

    def test_has_parent_setitem(self):
        model = self.model
        model.c = self._container_type()
        model.c[1] = self._ctype_factory()
        model.c[1] = model.c[1]
        with self.assertRaises(ValueError):
            model.c[2] = model.c[1]
        model.d = self._container_type()
        with self.assertRaises(ValueError):
            model.d[None] = model.c[1]

    """
    # make sure an existing Data object is NOT replaced
    # by a call to setitem but simply updated.
    def test_setitem_exists(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._arg) for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            model.c[i] = self._arg
            self.assertEqual(len(model.c), len(index))
            self.assertTrue(i in model.c)
            self.assertEqual(id(cdata), id(model.c[i]))
    """

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            model.c[i] = self._ctype_factory()
            self.assertEqual(len(model.c), len(index))
            self.assertTrue(i in model.c)
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertEqual(cdata.parent_component(), None)

    def test_delitem(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        for cnt, i in enumerate(index, 1):
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            self.assertEqual(id(cdata.parent_component()),
                             id(model.c))
            del model.c[i]
            self.assertEqual(len(model.c), len(index)-cnt)
            self.assertTrue(i not in model.c)
            self.assertEqual(cdata.parent_component(), None)

    def test_iter(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        comp_index = [i for i in model.c]
        self.assertEqual(len(comp_index), len(index))
        for idx in comp_index:
            self.assertTrue(idx in index)

    # TODO
    def Xtest_clone(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.c), id(model.c))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))

    # TODO
    def Xtest_pickle(self):
        index = ['a', 1, None, (1,), (1,2)]
        cdict = self._container_type((i, self._ctype_factory())
                                     for i in index)
        pickled_cdict = pickle.loads(
            pickle.dumps(cdict, protocol=_pickle_test_protocol))
        self.assertTrue(
            isinstance(pickled_cdict, self._container_type))
        self.assertTrue(pickled_cdict.parent is None)
        self.assertEqual(len(pickled_cdict), len(index))
        self.assertNotEqual(id(pickled_cdict), id(cdict))
        for i in index:
            self.assertNotEqual(id(pickled_cdict[i]), id(cdict[i]))
            self.assertTrue(pickled_cdict[i].parent is cdict)
            self.assertTrue(cdict[i].parent is cdict)

    def test_keys(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        model.c = self._container_type(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()),
                                key=str),
                         sorted(list(model.c.keys()), key=str))

    def test_values(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        model.c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list(id(_v)
                        for _v in raw_constraint_dict.values()),
                   key=str),
            sorted(list(id(_v)
                        for _v in model.c.values()),
                   key=str))

    def test_items(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        model.c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list((_i, id(_v))
                        for _i,_v in raw_constraint_dict.items()),
                   key=str),
            sorted(list((_i, id(_v))
                        for _i,_v in model.c.items()),
                   key=str))

    def test_update(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        model.c = self._container_type()
        model.c.update(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()),
                                key=str),
                         sorted(list(model.c.keys()), key=str))

    def test_cname(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        c = self._container_type((i, self._ctype_factory())
                                 for i in index)
        self.assertEqual(c.cname(False), None)
        self.assertEqual(c.cname(True), None)
        model.c = c
        self.assertEqual(c.cname(False), "c")
        self.assertEqual(c.cname(True), "c")
        index_to_string = {}
        index_to_string['a'] = '[a]'
        index_to_string[1] = '[1]'
        index_to_string[None] = '[None]'
        # I don't like that (1,) looks the same as 1, but oh well
        index_to_string[(1,)] = '[1]'
        index_to_string[(1,2)] = '[1,2]'
        prefix = "c"
        for i in index:
            cdata = model.c[i]
            self.assertEqual(cdata.cname(False),
                             cdata.cname(True))
            cname = prefix + index_to_string[i]
            self.assertEqual(cdata.cname(False),
                             cname)

    def test_cname(self):
        model = self.model
        components = {}
        components['a'] = self._ctype_factory()
        components[1] = self._ctype_factory()
        components[None] = self._ctype_factory()
        components[(1,)] = self._ctype_factory()
        components[(1,2)] = self._ctype_factory()
        components['(1,2)'] = self._ctype_factory()

        for key, c in components.items():
            self.assertTrue(c.parent is None)
            self.assertEqual(c.cname(False), None)
            self.assertEqual(c.cname(True), None)

        cdict = self._container_type()
        self.assertTrue(cdict.parent is None)
        self.assertEqual(cdict.cname(False), None)
        self.assertEqual(cdict.cname(True), None)
        cdict.update(components)
        for key, c in components.items():
            self.assertTrue(c.parent is cdict)
            self.assertEqual(c.cname(False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.cname(False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.cname(True, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.cname(True, convert=repr),
                             "[%s]" % (repr(key)))

        model.cdict = cdict
        self.assertTrue(cdict.parent is model)
        self.assertEqual(cdict.cname(False), "cdict")
        self.assertEqual(cdict.cname(True), "cdict")
        for key, c in components.items():
            self.assertEqual(c.cname(False, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(False, convert=repr),
                             "cdict[%s]" % (repr(key)))
            self.assertEqual(c.cname(True, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(True, convert=repr),
                             "cdict[%s]" % (repr(key)))

        b = block()
        b.model = model
        self.assertTrue(model.parent is b)
        self.assertEqual(cdict.cname(False), "cdict")
        self.assertEqual(cdict.cname(True), "model.cdict")
        for key, c in components.items():
            self.assertEqual(c.cname(False, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(False, convert=repr),
                             "cdict[%s]" % (repr(key)))
            self.assertEqual(c.cname(True, convert=str),
                             "model.cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(True, convert=repr),
                             "model.cdict[%s]" % (repr(key)))

        bdict = block_dict()
        bdict[0] = b
        self.assertTrue(b.parent is bdict)
        self.assertEqual(cdict.cname(False), "cdict")
        self.assertEqual(cdict.cname(True), "[0].model.cdict")
        for key, c in components.items():
            self.assertEqual(c.cname(False, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(False, convert=repr),
                             "cdict[%s]" % (repr(key)))
            self.assertEqual(c.cname(True, convert=str),
                             "[0].model.cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(True, convert=repr),
                             "[0].model.cdict[%s]" % (repr(key)))

        m = block()
        m.bdict = bdict
        self.assertTrue(bdict.parent is m)
        self.assertEqual(cdict.cname(False), "cdict")
        self.assertEqual(cdict.cname(True), "bdict[0].model.cdict")
        for key, c in components.items():
            self.assertEqual(c.cname(False, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(False, convert=repr),
                             "cdict[%s]" % (repr(key)))
            self.assertEqual(c.cname(True, convert=str),
                             "bdict[0].model.cdict[%s]" % (str(key)))
            self.assertEqual(c.cname(True, convert=repr),
                             "bdict[0].model.cdict[%s]" % (repr(key)))

    def test_clear(self):
        model = self.model
        model.c = self._container_type()
        model.c[1] = self._ctype_factory()
        c1 = model.c[1]
        with self.assertRaises(ValueError):
            model.c[None] = c1
        model.d = self._container_type()
        with self.assertRaises(ValueError):
            model.d[1] = c1
        model.c.clear()
        model.d[1] = c1

    def test_eq(self):
        model = self.model
        model.c = self._container_type()
        model.c[1] = self._ctype_factory()
        model.d = self._container_type()
        model.d[1] = self._ctype_factory()

        self.assertNotEqual(model.c, [])
        self.assertFalse(model.c == [])

        self.assertTrue(model.c == model.c)
        self.assertEqual(model.c, model.c)
        self.assertTrue(model.c == dict(model.c))
        self.assertEqual(model.c, dict(model.c))
        self.assertTrue(dict(model.c) == model.c)
        self.assertEqual(dict(model.c), model.c)

        self.assertFalse(model.d == model.c)
        self.assertTrue(model.d != model.c)
        self.assertNotEqual(model.d, model.c)

        self.assertFalse(model.c == model.d)
        self.assertTrue(model.c != model.d)
        self.assertNotEqual(model.c, model.d)

class _TestActiveComponentDictBase(_TestComponentDictBase):

    def test_activate(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        model.c._active = False
        for i in index:
            model.c[i]._active = False
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)
        model.c.activate()
        self.assertEqual(model.c.active, True)

    def test_activate(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        for i in index:
            self.assertEqual(model.c[i].active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)

    # Befault, assigning a new component to a dict container makes it
    # active (unless the default active state for the container
    # datatype is False, which is not the case for any currently
    # existing implementations).
    def test_active(self):
        model = self.model
        model.c = self._container_type()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c[1] = self._ctype_factory()
        self.assertEqual(model.c.active, True)

"""
class TestExpressionDict(_TestComponentDictBase,
                         unittest.TestCase):
    _container_type = expression_dict
    _ctype_factory = lambda self: expression(self.model.x**3)
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self.model.x = variable()

#
# Test components that include activate/deactivate
# functionality.
#

class TestConstraintDict(_TestActiveComponentDictBase,
                         unittest.TestCase):
    _container_type = constraint_dict
    _ctype_factory = lambda self: constraint(self.model.x >= 1)
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self.model.x = variable()

class TestObjectiveDict(_TestActiveComponentDictBase,
                        unittest.TestCase):
    _container_type = objective_dict
    _ctype_factory = lambda self: objective(self.model.x**2)
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self.model.x = variable()
"""

if __name__ == "__main__":
    unittest.main()
