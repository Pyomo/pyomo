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
                                             block_list)

#
# There are no fully implemented test suites in this
# file. These classes are meant to be used to test
# full implementations of ComponentList containers.
# To test a ComponentList implementation, just import
# this class and use it as a subclass in another test suite.
#

# Note: we need to test with a pickle protocol
#       that is high enough to support __slots__
#       and weakref (bas
_pickle_test_protocol = pickle.HIGHEST_PROTOCOL

class _bad_ctype(object):
    ctype = "_this_is_definitely_not_the_ctype_being_tested"

class _TestComponentListBase(object):

    # set by derived class
    _container_type = None
    _ctype_factory = None

    def setUp(self):
        self.model = block()

    def tearDown(self):
        self.model = None

    def test_init1(self):
        model = self.model
        model.c = self._container_type()

    def test_init2(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        with self.assertRaises(TypeError):
            model.d = self._container_type(
                *tuple(self._ctype_factory() for i in index))

    def test_len1(self):
        model = self.model
        model.c = self._container_type()
        self.assertEqual(len(model.c), 0)

    def test_len2(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(model.c), len(index))

    def test_append(self):
        model = self.model
        model.c = self._container_type()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        for i in index:
            c_new = self._ctype_factory()
            model.c.append(c_new)
            self.assertEqual(id(model.c[-1]), id(c_new))
            self.assertEqual(len(model.c), i+1)

    def test_insert(self):
        model = self.model
        model.c = self._container_type()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        for i in index:
            c_new = self._ctype_factory()
            model.c.insert(0, c_new)
            self.assertEqual(id(model.c[0]), id(c_new))
            self.assertEqual(len(model.c), i+1)

    def test_setitem(self):
        model = self.model
        model.c = self._container_type()
        index = range(5)
        for i in index:
            model.c.append(self._ctype_factory())
        for i in index:
            c_new = self._ctype_factory()
            self.assertNotEqual(id(c_new), id(model.c[i]))
            model.c[i] = c_new
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(id(c_new), id(model.c[i]))

    def test_wrong_type_init(self):
        model = self.model
        index = range(5)
        with self.assertRaises(TypeError):
            model.c = self._container_type(
                _bad_ctype() for i in index)

    def test_wrong_type_append(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        with self.assertRaises(TypeError):
            model.c.append(_bad_ctype())

    def test_wrong_type_insert(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        model.c.insert(0, self._ctype_factory())
        with self.assertRaises(TypeError):
            model.c.insert(0, _bad_ctype())

    def test_wrong_type_setitem(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        model.c[0] = self._ctype_factory()
        with self.assertRaises(TypeError):
            model.c[0] = _bad_ctype()

    def test_has_parent_init(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            model.c.append(model.c[0])
        with self.assertRaises(ValueError):
            model.d = self._container_type(model.c)

    def test_has_parent_append(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            model.c.append(model.c[0])
        d = []
        d.append(model.c[0])
        model.d = self._container_type()
        with self.assertRaises(ValueError):
            model.d.append(model.c[0])

    def test_has_parent_insert(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        model.c.insert(0, self._ctype_factory())
        with self.assertRaises(ValueError):
            model.c.insert(0, model.c[0])
        d = []
        d.insert(0, model.c[0])
        model.d = self._container_type()
        with self.assertRaises(ValueError):
            model.d.insert(0, model.c[0])

    def test_has_parent_setitem(self):
        model = self.model
        model.c = self._container_type()
        model.c.append(self._ctype_factory())
        model.c[0] = self._ctype_factory()
        model.c[0] = model.c[0]
        model.c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            model.c[0] = model.c[1]

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            cdata = model.c[i]
            self.assertEqual(id(cdata.parent),
                             id(model.c))
            model.c[i] = self._ctype_factory()
            self.assertEqual(len(model.c), len(index))
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertEqual(cdata.parent, None)

    def test_delitem(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            cdata = model.c[0]
            self.assertEqual(id(cdata.parent),
                             id(model.c))
            del model.c[0]
            self.assertEqual(len(model.c), len(index)-(i+1))
            self.assertEqual(cdata.parent, None)

    def test_iter(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(model.c), len(index))
        raw_list = model.c[:]
        self.assertEqual(type(raw_list), list)
        for c1, c2 in zip(raw_list, model.c):
            self.assertEqual(id(c1), id(c2))

    def test_reverse(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        raw_list = model.c[:]
        self.assertEqual(type(raw_list), list)
        model.c.reverse()
        raw_list.reverse()
        for c1, c2 in zip(model.c, raw_list):
            self.assertEqual(id(c1), id(c2))

    def test_remove(self):
        model = self.model
        model = block()
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = model.c[0]
            self.assertEqual(cdata in model.c, True)
            model.c.remove(cdata)
            self.assertEqual(cdata in model.c, False)

    def test_pop(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = model.c[-1]
            self.assertEqual(cdata in model.c, True)
            last = model.c.pop()
            self.assertEqual(cdata in model.c, False)
            self.assertEqual(id(cdata), id(last))

    def test_index(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = model.c[i]
            self.assertEqual(model.c.index(cdata), i)
            self.assertEqual(model.c.index(cdata, start=i), i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i+1)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i, stop=i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, stop=i)
            self.assertEqual(
                model.c.index(cdata, start=i, stop=i+1), i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i+1, stop=i+1)
            self.assertEqual(
                model.c.index(cdata, start=-len(index)+i), i)
            if i == index[-1]:
                self.assertEqual(
                    model.c.index(cdata, start=-len(index)+i+1), i)
            else:
                with self.assertRaises(ValueError):
                    self.assertEqual(
                        model.c.index(cdata, start=-len(index)+i+1),
                        i)
            if i == index[-1]:
                with self.assertRaises(ValueError):
                    self.assertEqual(
                        model.c.index(cdata, stop=-len(index)+i+1), i)
            else:
                self.assertEqual(
                    model.c.index(cdata, stop=-len(index)+i+1), i)
        tmp = self._ctype_factory()
        with self.assertRaises(ValueError):
            model.c.index(tmp)
        with self.assertRaises(ValueError):
            model.c.index(tmp, stop=len(model.c)+1)

    def test_extend(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        c_more_list = [self._ctype_factory() for i in index]
        self.assertEqual(len(model.c), len(index))
        self.assertTrue(len(c_more_list) > 0)
        for cdata in c_more_list:
            self.assertEqual(cdata.parent, None)
        model.c.extend(c_more_list)
        for cdata in c_more_list:
            self.assertEqual(id(cdata.parent),
                             id(model.c))

    def test_count(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            self.assertEqual(model.c.count(model.c[i]), 1)

    # TODO
    def Xtest_clone(self):
        model = self.model
        index = range(5)
        model.c = self._container_type(
            self._ctype_factory() for i in index)
        model_clone = model.clone()
        self.assertNotEqual(id(model_clone.c), id(model.c))
        for i in index:
            self.assertNotEqual(id(model_clone.c[i]), id(model.c[i]))

    # TODO
    def Xtest_pickle(self):
        index = range(5)
        clist = self._container_type(
            self._ctype_factory() for i in index)
        pickled_clist = pickle.loads(
            pickle.dumps(clist, protocol=_pickle_test_protocol))
        self.assertTrue(
            isinstance(pickled_clist, self._container_type))
        self.assertTrue(pickled_clist.parent is None)
        self.assertEqual(len(pickled_clist), len(index))
        self.assertNotEqual(id(pickled_clist), id(clist))
        for i in index:
            self.assertNotEqual(id(pickled_clist[i]), id(clist[i]))
            self.assertTrue(pickled_clist[i].parent is clist)
            self.assertTrue(clist[i].parent is clist)

    def test_name(self):
        model = self.model
        components = [self._ctype_factory() for i in range(5)]

        for c in components:
            self.assertTrue(c.parent is None)
            self.assertEqual(c.name(False), None)
            self.assertEqual(c.name(True), None)

        clist = self._container_type()
        self.assertTrue(clist.parent is None)
        self.assertEqual(clist.name(False), None)
        self.assertEqual(clist.name(True), None)
        clist.extend(components)
        for i, c in enumerate(components):
            self.assertTrue(c.parent is clist)
            self.assertEqual(c.name(False), "[%s]" % (i))
            self.assertEqual(c.name(True), "[%s]" % (i))

        model.clist = clist
        self.assertTrue(clist.parent is model)
        self.assertEqual(clist.name(False), "clist")
        self.assertEqual(clist.name(True), "clist")
        for i, c in enumerate(components):
            self.assertEqual(c.name(False), "clist[%s]" % (i))
            self.assertEqual(c.name(True), "clist[%s]" % (i))

        b = block()
        b.model = model
        self.assertTrue(model.parent is b)
        self.assertEqual(clist.name(False), "clist")
        self.assertEqual(clist.name(True), "model.clist")
        for i, c in enumerate(components):
            self.assertEqual(c.name(False), "clist[%s]" % (i))
            self.assertEqual(c.name(True), "model.clist[%s]" % (i))

        blist = block_list()
        blist.append(b)
        self.assertTrue(b.parent is blist)
        self.assertEqual(clist.name(False), "clist")
        self.assertEqual(clist.name(True), "[0].model.clist")
        for i, c in enumerate(components):
            self.assertEqual(c.name(False), "clist[%s]" % (i))
            self.assertEqual(c.name(True),
                             "[0].model.clist[%s]" % (i))

        m = block()
        m.blist = blist
        self.assertTrue(blist.parent is m)
        self.assertEqual(clist.name(False), "clist")
        self.assertEqual(clist.name(True), "blist[0].model.clist")
        for i, c in enumerate(components):
            self.assertEqual(c.name(False), "clist[%s]" % (i))
            self.assertEqual(c.name(True),
                             "blist[0].model.clist[%s]" % (i))

class _TestActiveComponentListBase(_TestComponentListBase):

    def test_activate(self):
        model = self.model
        index = list(range(4))
        model.c = self._container_type(self._ctype_factory()
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
        index = list(range(4))
        model.c = self._container_type(self._ctype_factory()
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
        model = block()
        model.c = self._container_type()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.append(self._ctype_factory())
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.insert(0, self._ctype_factory())
        self.assertEqual(model.c.active, True)

"""
class TestExpressionList(_TestComponentListBase,
                         unittest.TestCase):
    _container_type = expression_list
    _ctype_factory = lambda self: expression(self.model.x**3)
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self.model.x = variable()

#
# Test components that include activate/deactivate
# functionality.
#

class TestConstraintList(_TestActiveComponentListBase,
                         unittest.TestCase):
    _container_type = constraint_list
    _ctype_factory = lambda self: constraint(self.model.x >= 1)
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self.model.x = variable()

class TestObjectiveList(_TestActiveComponentListBase,
                        unittest.TestCase):
    _container_type = objective_list
    _ctype_factory = lambda self: objective(self.model.x**2)
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self.model.x = variable()
"""

if __name__ == "__main__":
    unittest.main()
