#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pickle
import collections
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponentMixin,
                                                 IComponentContainer,
                                                 _IActiveComponentContainerMixin)
from pyomo.core.base.component_block import (IBlockStorage,
                                             block,
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

    def test_init1(self):
        cdict = self._container_type()

    def test_init2(self):
        index = ['a', 1, None, (1,), (1,2)]
        self._container_type((i, self._ctype_factory())
                             for i in index)
        self._container_type(((i, self._ctype_factory())
                             for i in index))
        with self.assertRaises(TypeError):
            self._container_type(*tuple((i, self._ctype_factory())
                                        for i in index))
        with self.assertRaises(ValueError):
            self._container_type(((i, self._ctype_factory())
                                  for i in index),
                                 definately_should_not_be_a_keyword_for_this_class=True)

    def test_ordered_init(self):
        cdict = self._container_type()
        self.assertEqual(type(cdict._data), dict)
        cdict = self._container_type(ordered=False)
        self.assertEqual(type(cdict._data), dict)
        cdict = self._container_type(ordered=True)
        self.assertNotEqual(type(cdict._data), dict)
        self.assertEqual(type(cdict._data), OrderedDict)
        cdict[None] = self._ctype_factory()
        cdict[1] = self._ctype_factory()
        cdict['a'] = self._ctype_factory()
        cdict[2] = self._ctype_factory()
        cdict[3] = self._ctype_factory()
        cdict[-1] = self._ctype_factory()
        cdict['bc'] = self._ctype_factory()
        self.assertEqual(list(cdict.keys()),
                         [None,1,'a',2,3,-1,'bc'])

    def test_type(self):
        cdict = self._container_type()
        self.assertTrue(isinstance(cdict, ICategorizedObject))
        self.assertTrue(isinstance(cdict, IComponentContainer))
        self.assertFalse(isinstance(cdict, IComponent))
        self.assertTrue(isinstance(cdict, collections.Mapping))
        self.assertTrue(isinstance(cdict, collections.MutableMapping))
        self.assertTrue(issubclass(type(cdict), collections.Mapping))
        self.assertTrue(issubclass(type(cdict), collections.MutableMapping))

    def test_len1(self):
        cdict = self._container_type()
        self.assertEqual(len(cdict), 0)

    def test_len2(self):
        index = ['a', 1, None, (1,), (1,2)]
        cdict = self._container_type((i, self._ctype_factory())
                                     for i in index)
        self.assertEqual(len(cdict), len(index))

    def test_setitem(self):
        cdict = self._container_type()
        index = ['a', 1, None, (1,), (1,2)]
        for i in index:
            self.assertTrue(i not in cdict)
        for cnt, i in enumerate(index, 1):
            cdict[i] = self._ctype_factory()
            self.assertEqual(len(cdict), cnt)
            self.assertTrue(i in cdict)

    # The immediately following this one was originally written when
    # implicit assignment for update was supported. This should be
    # examined more carefully before supporting it.
    # For now just test that implicit assignment raises an exception
    def test_wrong_type_init(self):
        index = ['a', 1, None, (1,), (1,2)]
        with self.assertRaises(TypeError):
            c = self._container_type(
                (i, _bad_ctype()) for i in index)

    def test_wrong_type_update(self):
        index = ['a', 1, None, (1,), (1,2)]
        c = self._container_type()
        with self.assertRaises(TypeError):
            c.update((i, _bad_ctype()) for i in index)

    def test_wrong_type_setitem(self):
        c = self._container_type()
        with self.assertRaises(TypeError):
            c[1] = _bad_ctype()
        c[1] = self._ctype_factory()
        with self.assertRaises(TypeError):
            c[1] = _bad_ctype()

    def test_has_parent_init(self):
        c = self._container_type()
        c[1] = self._ctype_factory()
        with self.assertRaises(ValueError):
            d = self._container_type(c)
        with self.assertRaises(ValueError):
            d = self._container_type(dict(c))

    def test_has_parent_update(self):
        c = self._container_type()
        c[1] = self._ctype_factory()
        c.update(c)
        self.assertEqual(len(c), 1)
        c.update(dict(c))
        self.assertEqual(len(c), 1)
        d = self._container_type()
        with self.assertRaises(ValueError):
            d.update(c)
        with self.assertRaises(ValueError):
            d.update(dict(c))

    def test_has_parent_setitem(self):
        c = self._container_type()
        c[1] = self._ctype_factory()
        c[1] = c[1]
        with self.assertRaises(ValueError):
            c[2] = c[1]
        d = self._container_type()
        with self.assertRaises(ValueError):
            d[None] = c[1]

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        index = ['a', 1, None, (1,), (1,2)]
        c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(c), len(index))
        for i in index:
            self.assertTrue(i in c)
            cdata = c[i]
            c[i] = self._ctype_factory()
            self.assertEqual(len(c), len(index))
            self.assertTrue(i in c)
            self.assertNotEqual(id(cdata), id(c[i]))
            self.assertEqual(cdata.parent, None)

    def test_delitem(self):
        index = ['a', 1, None, (1,), (1,2)]
        c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(c), len(index))
        for cnt, i in enumerate(index, 1):
            self.assertTrue(i in c)
            cdata = c[i]
            self.assertEqual(id(cdata.parent),
                             id(c))
            del c[i]
            self.assertEqual(len(c), len(index)-cnt)
            self.assertTrue(i not in c)
            self.assertEqual(cdata.parent, None)

    def test_iter(self):
        index = ['a', 1, None, (1,), (1,2)]
        c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(c), len(index))
        comp_index = [i for i in c]
        self.assertEqual(len(comp_index), len(index))
        for idx in comp_index:
            self.assertTrue(idx in index)

    def test_pickle(self):
        index = ['a', 1, None, (1,), (1,2)]
        cdict = self._container_type((i, self._ctype_factory())
                                     for i in index)
        cdict[0] = self._container_type()
        index.append(0)
        for i in index:
            self.assertTrue(cdict[i].parent is cdict)
        pickled_cdict = pickle.loads(pickle.dumps(cdict))
        self.assertTrue(
            isinstance(pickled_cdict, self._container_type))
        self.assertTrue(pickled_cdict.parent is None)
        self.assertEqual(len(pickled_cdict), len(index))
        self.assertNotEqual(id(pickled_cdict), id(cdict))
        for i in index:
            self.assertNotEqual(id(pickled_cdict[i]), id(cdict[i]))
            self.assertTrue(pickled_cdict[i].parent is pickled_cdict)
            self.assertTrue(cdict[i].parent is cdict)

    def test_keys(self):
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()),
                                key=str),
                         sorted(list(c.keys()), key=str))

    def test_values(self):
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list(id(_v)
                        for _v in raw_constraint_dict.values()),
                   key=str),
            sorted(list(id(_v)
                        for _v in c.values()),
                   key=str))

    def test_items(self):
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list((_i, id(_v))
                        for _i,_v in raw_constraint_dict.items()),
                   key=str),
            sorted(list((_i, id(_v))
                        for _i,_v in c.items()),
                   key=str))

    def test_update(self):
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = dict((i, self._ctype_factory())
                                   for i in index)
        c = self._container_type()
        c.update(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()),
                                key=str),
                         sorted(list(c.keys()), key=str))

    def test_clear(self):
        c = self._container_type()
        c[1] = self._ctype_factory()
        c1 = c[1]
        with self.assertRaises(ValueError):
            c[None] = c1
        d = self._container_type()
        with self.assertRaises(ValueError):
            d[1] = c1
        c.clear()
        d[1] = c1

    def test_eq(self):
        cdict1 = self._container_type()
        cdict1[1] = self._ctype_factory()
        cdict2 = self._container_type()
        cdict2[1] = self._ctype_factory()

        self.assertNotEqual(cdict1, set())
        self.assertFalse(cdict1 == set())
        self.assertNotEqual(cdict1, list())
        self.assertFalse(cdict1 == list())
        self.assertNotEqual(cdict1, tuple())
        self.assertFalse(cdict1 == tuple())
        self.assertNotEqual(cdict1, dict())
        self.assertFalse(cdict1 == dict())

        self.assertTrue(cdict1 == cdict1)
        self.assertEqual(cdict1, cdict1)
        self.assertTrue(cdict1 == dict(cdict1))
        self.assertEqual(cdict1, dict(cdict1))
        self.assertTrue(dict(cdict1) == cdict1)
        self.assertEqual(dict(cdict1), cdict1)

        self.assertFalse(cdict2 == cdict1)
        self.assertTrue(cdict2 != cdict1)
        self.assertNotEqual(cdict2, cdict1)

        self.assertFalse(cdict1 == cdict2)
        self.assertTrue(cdict1 != cdict2)
        self.assertNotEqual(cdict1, cdict2)

        cdict1.clear()

        self.assertNotEqual(cdict1, set())
        self.assertFalse(cdict1 == set())
        self.assertNotEqual(cdict1, list())
        self.assertFalse(cdict1 == list())
        self.assertNotEqual(cdict1, tuple())
        self.assertFalse(cdict1 == tuple())
        self.assertEqual(cdict1, dict())
        self.assertTrue(cdict1 == dict())

    def test_child_key(self):
        cdict = self._container_type()
        c = self._ctype_factory()
        with self.assertRaises(ValueError):
            cdict.child_key(c)
        cdict[1] = c
        self.assertEqual(cdict.child_key(c), 1)

    def test_name(self):
        children = {}
        children['a'] = self._ctype_factory()
        children[1] = self._ctype_factory()
        children[None] = self._ctype_factory()
        children[(1,)] = self._ctype_factory()
        children[(1,2)] = self._ctype_factory()
        children['(1,2)'] = self._ctype_factory()
        children['x'] = self._container_type()
        children['x']['y'] = self._ctype_factory()

        for key, c in children.items():
            self.assertTrue(c.parent is None)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.local_name, None)
            self.assertEqual(c.name, None)

        cdict = self._container_type()
        self.assertTrue(cdict.parent is None)
        self.assertTrue(cdict.parent_block is None)
        self.assertTrue(cdict.root_block is None)
        self.assertEqual(cdict.local_name, None)
        self.assertEqual(cdict.name, None)
        cdict.update(children)
        names = cdict.generate_names()
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.getname(fully_qualified=False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.name, names[c])
        for c in cdict.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        model = block()
        model.cdict = cdict
        self.assertTrue(model.parent is None)
        self.assertTrue(model.parent_block is None)
        self.assertTrue(model.root_block is model)
        self.assertTrue(cdict.parent is model)
        self.assertTrue(cdict.parent_block is model)
        self.assertTrue(cdict.root_block is model)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "cdict")
        names = model.generate_names()
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is model)
            self.assertEqual(c.getname(fully_qualified=False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=str),
                             "cdict[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=repr),
                             "cdict[%s]" % (repr(key)))
            self.assertEqual(c.name, names[c])
        for c in cdict.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        b = block()
        b.model = model
        self.assertTrue(b.parent is None)
        self.assertTrue(b.parent_block is None)
        self.assertTrue(b.root_block is b)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is b)
        self.assertTrue(cdict.parent is model)
        self.assertTrue(cdict.parent_block is model)
        self.assertTrue(cdict.root_block is b)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "model.cdict")
        names = b.generate_names()
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.getname(fully_qualified=False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=str),
                             "model.cdict[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=repr),
                             "model.cdict[%s]" % (repr(key)))
            self.assertEqual(c.name, names[c])
        for c in cdict.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        bdict = block_dict()
        bdict[0] = b
        self.assertTrue(bdict.parent is None)
        self.assertTrue(bdict.parent_block is None)
        self.assertTrue(bdict.root_block is None)
        self.assertTrue(b.parent is bdict)
        self.assertTrue(b.parent_block is None)
        self.assertTrue(b.root_block is b)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is b)
        self.assertTrue(cdict.parent is model)
        self.assertTrue(cdict.parent_block is model)
        self.assertTrue(cdict.root_block is b)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "[0].model.cdict")
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.getname(fully_qualified=False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=str),
                             "[0].model.cdict[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=repr),
                             "[0].model.cdict[%s]" % (repr(key)))

        m = block()
        m.bdict = bdict
        self.assertTrue(m.parent is None)
        self.assertTrue(m.parent_block is None)
        self.assertTrue(m.root_block is m)
        self.assertTrue(bdict.parent is m)
        self.assertTrue(bdict.parent_block is m)
        self.assertTrue(bdict.root_block is m)
        self.assertTrue(b.parent is bdict)
        self.assertTrue(b.parent_block is m)
        self.assertTrue(b.root_block is m)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is m)
        self.assertTrue(cdict.parent is model)
        self.assertTrue(cdict.parent_block is model)
        self.assertTrue(cdict.root_block is m)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "bdict[0].model.cdict")
        names = m.generate_names()
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is m)
            self.assertEqual(c.getname(fully_qualified=False, convert=str),
                             "[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=False, convert=repr),
                             "[%s]" % (repr(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=str),
                             "bdict[0].model.cdict[%s]" % (str(key)))
            self.assertEqual(c.getname(fully_qualified=True, convert=repr),
                             "bdict[0].model.cdict[%s]" % (repr(key)))
            self.assertEqual(c.name, names[c])
        for c in cdict.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])
        names = m.generate_names(descend_into=False)
        self.assertEqual(len(names), len(list(m.children())))
        for c in m.children():
            self.assertEqual(c.name, names[c])

    def test_preorder_traversal(self):
        traversal = []
        cdict = self._container_type(ordered=True)
        traversal.append((None,cdict))
        cdict[0] = self._ctype_factory()
        traversal.append((0,cdict[0]))
        cdict[1] = self._container_type(ordered=True)
        traversal.append((1,cdict[1]))
        cdict[1][0] = self._ctype_factory()
        traversal.append((0,cdict[1][0]))
        cdict[2] = self._ctype_factory()
        traversal.append((2,cdict[2]))

        self.assertEqual([c.name for k,c in traversal],
                         [c.name for c in cdict.preorder_traversal()])
        self.assertEqual([id(c) for k,c in traversal],
                         [id(c) for c in cdict.preorder_traversal()])

        self.assertEqual([(k,c.name) for k,c in traversal],
                         [(k,c.name) for k,c in cdict.preorder_traversal(
                             return_key=True)])
        self.assertEqual([(k,id(c)) for k,c in traversal],
                         [(k,id(c)) for k,c in cdict.preorder_traversal(
                             return_key=True)])
        return cdict, traversal

    def test_postorder_traversal(self):
        traversal = []
        cdict = self._container_type(ordered=True)
        cdict[0] = self._ctype_factory()
        traversal.append((0,cdict[0]))
        cdict[1] = self._container_type(ordered=True)
        cdict[1][0] = self._ctype_factory()
        traversal.append((0,cdict[1][0]))
        traversal.append((1,cdict[1]))
        cdict[2] = self._ctype_factory()
        traversal.append((2,cdict[2]))
        traversal.append((None,cdict))

        self.assertEqual([c.name for k,c in traversal],
                         [c.name for c in cdict.postorder_traversal()])
        self.assertEqual([id(c) for k,c in traversal],
                         [id(c) for c in cdict.postorder_traversal()])

        self.assertEqual([(k,c.name) for k,c in traversal],
                         [(k,c.name) for k,c in cdict.postorder_traversal(
                             return_key=True)])
        self.assertEqual([(k,id(c)) for k,c in traversal],
                         [(k,id(c)) for k,c in cdict.postorder_traversal(
                             return_key=True)])
        return cdict, traversal

class _TestActiveComponentDictBase(_TestComponentDictBase):

    def test_active_type(self):
        cdict = self._container_type()
        self.assertTrue(isinstance(cdict, IComponentContainer))
        self.assertTrue(isinstance(cdict, _IActiveComponentContainerMixin))
        self.assertTrue(isinstance(cdict, ICategorizedObject))
        self.assertFalse(isinstance(cdict, IComponent))
        self.assertFalse(isinstance(cdict, _IActiveComponentMixin))

    def test_active(self):
        children = {}
        children['a'] = self._ctype_factory()
        children[1] = self._ctype_factory()
        children[None] = self._ctype_factory()
        children[(1,)] = self._ctype_factory()
        children[(1,2)] = self._ctype_factory()
        children['(1,2)'] = self._ctype_factory()

        cdict = self._container_type()
        cdict.update(children)
        with self.assertRaises(AttributeError):
            cdict.active = False
        for c in cdict.values():
            with self.assertRaises(AttributeError):
                c.active = False

        model = block()
        model.cdict = cdict
        b = block()
        b.model = model
        bdict = block_dict()
        bdict[0] = b
        bdict[None] = block()
        m = block()
        m.bdict = bdict

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)

        m.deactivate(shallow=False,
                     descend_into=True)

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)

        test_key = list(children.keys())[0]
        del cdict[test_key]
        cdict[test_key] = children[test_key]

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)

        del cdict[test_key]
        children[test_key].activate()
        self.assertEqual(children[test_key].active, True)
        cdict[test_key] = children[test_key]

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, True)
        for key, c in cdict.items():
            if key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)

        cdict.deactivate()
        m.activate(shallow=False,
                   descend_into=True)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)

        cdict.deactivate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)

        cdict.activate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)

        cdict.deactivate()
        cdict[test_key].activate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for key, c in cdict.items():
            if key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)

    def test_preorder_traversal(self):
        cdict, traversal = \
            super(_TestActiveComponentDictBase, self).\
            test_preorder_traversal()
        cdict[1].deactivate()
        self.assertEqual([c.name for k,c in traversal if c.active],
                         [c.name for c in cdict.preorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for k,c in traversal if c.active],
                         [id(c) for c in cdict.preorder_traversal(
                             active=True)])
        cdict.deactivate()
        self.assertEqual(len(list(cdict.preorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(cdict.generate_names(active=True))),
                         0)

    def test_postorder_traversal(self):
        cdict, traversal = \
            super(_TestActiveComponentDictBase, self).\
            test_postorder_traversal()
        cdict[1].deactivate()
        self.assertEqual([c.name for k,c in traversal if c.active],
                         [c.name for c in cdict.postorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for k,c in traversal if c.active],
                         [id(c) for c in cdict.postorder_traversal(
                             active=True)])
        cdict.deactivate()
        self.assertEqual(len(list(cdict.postorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(cdict.generate_names(active=True))),
                         0)

if __name__ == "__main__":
    unittest.main()
