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
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     ICategorizedObjectContainer)
from pyomo.core.kernel.homogeneous_container import \
    IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import (block,
                                     block_dict)

import six
from six import StringIO

if six.PY3:
    from collections.abc import Mapping as collections_Mapping
    from collections.abc import MutableMapping as collections_MutableMapping
else:
    from collections import Mapping as collections_Mapping
    from collections import MutableMapping as collections_MutableMapping

#
# There are no fully implemented test suites in this
# file. These classes are meant to be used to test
# full implementations of DictContainer containers.
# To test a DictContainer implementation, just import
# this class and use it as a subclass in another test suite.
#

# Note: we need to test with a pickle protocol
#       that is high enough to support __slots__
#       and weakref (bas
_pickle_test_protocol = pickle.HIGHEST_PROTOCOL


class _bad_ctype(object):
    ctype = "_this_is_definitely_not_the_ctype_being_tested"


class _TestDictContainerBase(object):

    # set by derived class
    _container_type = None
    _ctype_factory = None

    def test_overwrite_warning(self):
        c = self._container_type()
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            c[0] = self._ctype_factory()
            c[0] = c[0]
        assert out.getvalue() == ""
        with LoggingIntercept(out, 'pyomo.core'):
            c[0] = self._ctype_factory()
        assert out.getvalue() == \
            ("Implicitly replacing the entry [0] "
             "(type=%s) with a new object (type=%s). "
             "This is usually indicative of a modeling "
             "error. To avoid this warning, delete the "
             "original object from the container before "
             "assigning a new object.\n"
             % (self._ctype_factory().__class__.__name__,
                self._ctype_factory().__class__.__name__))

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        cdict = self._container_type({None: self._ctype_factory()})
        pyomo.kernel.pprint(cdict)
        b = block()
        b.cdict = cdict
        pyomo.kernel.pprint(cdict)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(cdict)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_ctype(self):
        c = self._container_type()
        ctype = self._ctype_factory().ctype
        self.assertIs(c.ctype, ctype)
        self.assertIs(type(c)._ctype, ctype)
        self.assertIs(self._container_type._ctype, ctype)

    def test_init1(self):
        cdict = self._container_type()

    def test_init2(self):
        index = ['a', 1, None, (1,), (1, 2)]
        self._container_type((i, self._ctype_factory())
                             for i in index)
        self._container_type(((i, self._ctype_factory())
                             for i in index))
        with self.assertRaises(TypeError):
            self._container_type(*tuple((i, self._ctype_factory())
                                        for i in index))
        c = self._container_type(a=self._ctype_factory(),
                                 b=self._ctype_factory())
        self.assertEqual(len(c), 2)
        self.assertTrue('a' in c)
        self.assertTrue('b' in c)

    def test_ordered_init(self):
        cdict = self._container_type()
        cdict[None] = self._ctype_factory()
        cdict[1] = self._ctype_factory()
        cdict['a'] = self._ctype_factory()
        del cdict[None]
        cdict[2] = self._ctype_factory()
        cdict[3] = self._ctype_factory()
        del cdict[3]
        cdict[None] = self._ctype_factory()
        cdict[-1] = self._ctype_factory()
        cdict['bc'] = self._ctype_factory()
        cdict[3] = self._ctype_factory()
        self.assertEqual(list(cdict.keys()),
                         [1,'a',2,None,-1,'bc',3])

    def test_type(self):
        cdict = self._container_type()
        self.assertTrue(isinstance(cdict, ICategorizedObject))
        self.assertTrue(isinstance(cdict, ICategorizedObjectContainer))
        self.assertTrue(isinstance(cdict, IHomogeneousContainer))
        self.assertTrue(isinstance(cdict, DictContainer))
        self.assertTrue(isinstance(cdict, collections_Mapping))
        self.assertTrue(isinstance(cdict, collections_MutableMapping))
        self.assertTrue(issubclass(type(cdict), collections_Mapping))
        self.assertTrue(issubclass(type(cdict), collections_MutableMapping))

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
        index = ['a', 1, None, (1,), (1, 2)]
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
        index = ['a', 1, None, (1,), (1, 2)]
        with self.assertRaises(TypeError):
            c = self._container_type(
                (i, _bad_ctype()) for i in index)

    def test_wrong_type_update(self):
        index = ['a', 1, None, (1,), (1, 2)]
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
        index = ['a', 1, None, (1,), (1, 2)]
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
        index = ['a', 1, None, (1,), (1, 2)]
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
        index = ['a', 1, None, (1,), (1, 2)]
        c = self._container_type((i, self._ctype_factory())
                              for i in index)
        self.assertEqual(len(c), len(index))
        comp_index = [i for i in c]
        self.assertEqual(len(comp_index), len(index))
        for idx in comp_index:
            self.assertTrue(idx in index)

    def test_pickle(self):
        index = ['a', 1, None, (1,), (1, 2)]
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
        index = ['a', 1, None, (1,), (1, 2)]
        raw_constraint_dict = {i:self._ctype_factory() for i in index}
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()),
                                key=str),
                         sorted(list(c.keys()), key=str))

    def test_values(self):
        index = ['a', 1, None, (1,), (1, 2)]
        raw_constraint_dict = {i:self._ctype_factory() for i in index}
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list(id(_v)
                        for _v in raw_constraint_dict.values()),
                   key=str),
            sorted(list(id(_v)
                        for _v in c.values()),
                   key=str))

    def test_items(self):
        index = ['a', 1, None, (1,), (1, 2)]
        raw_constraint_dict = {i:self._ctype_factory() for i in index}
        c = self._container_type(raw_constraint_dict)
        self.assertEqual(
            sorted(list((_i, id(_v))
                        for _i,_v in raw_constraint_dict.items()),
                   key=str),
            sorted(list((_i, id(_v))
                        for _i,_v in c.items()),
                   key=str))

    def test_update(self):
        index = ['a', 1, None, (1,), (1, 2)]
        raw_constraint_dict = {i:self._ctype_factory() for i in index}
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

    def test_child(self):
        cdict = self._container_type()
        c = self._ctype_factory()
        with self.assertRaises(KeyError):
            cdict.child(1)
        cdict[1] = c
        self.assertIs(cdict.child(1), c)

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
            self.assertEqual(c.local_name, None)
            self.assertEqual(c.name, None)

        cdict = self._container_type()
        self.assertTrue(cdict.parent is None)
        self.assertEqual(cdict.local_name, None)
        self.assertEqual(cdict.name, None)
        cdict.update(children)
        names = pmo.generate_names(cdict)
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
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
        self.assertTrue(cdict.parent is model)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "cdict")
        names = pmo.generate_names(model)
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
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
        self.assertTrue(model.parent is b)
        self.assertTrue(cdict.parent is model)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "model.cdict")
        names = pmo.generate_names(b)
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
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
        self.assertTrue(b.parent is bdict)
        self.assertTrue(model.parent is b)
        self.assertTrue(cdict.parent is model)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "[0].model.cdict")
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
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
        self.assertTrue(bdict.parent is m)
        self.assertTrue(b.parent is bdict)
        self.assertTrue(model.parent is b)
        self.assertTrue(cdict.parent is model)
        self.assertEqual(cdict.local_name, "cdict")
        self.assertEqual(cdict.name, "bdict[0].model.cdict")
        names = pmo.generate_names(m)
        for key, c in children.items():
            self.assertTrue(c.parent is cdict)
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
        names = pmo.generate_names(m)
        for c in m.children():
            self.assertEqual(c.name, names[c])

    def test_preorder_traversal(self):
        traversal = []
        cdict = self._container_type()
        traversal.append(cdict)
        cdict[0] = self._ctype_factory()
        traversal.append(cdict[0])
        cdict[1] = self._container_type()
        traversal.append(cdict[1])
        cdict[1][0] = self._ctype_factory()
        traversal.append(cdict[1][0])
        cdict[2] = self._ctype_factory()
        traversal.append(cdict[2])

        descend = lambda x: not x._is_heterogeneous_container

        self.assertEqual([c.name for c in traversal],
                         [c.name for c in pmo.preorder_traversal(
                             cdict,
                             descend=descend)])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in pmo.preorder_traversal(
                             cdict,
                             descend=descend)])
        return cdict, traversal

    def test_preorder_traversal_descend_check(self):
        traversal = []
        cdict = self._container_type()
        traversal.append(cdict)
        cdict[0] = self._ctype_factory()
        traversal.append(cdict[0])
        cdict[1] = self._container_type()
        traversal.append(cdict[1])
        cdict[1][0] = self._ctype_factory()
        traversal.append(cdict[1][0])
        cdict[2] = self._ctype_factory()
        traversal.append(cdict[2])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return False
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            descend=descend))
        self.assertEqual(len(order), 1)
        self.assertIs(order[0], cdict)
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], cdict)

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            descend=descend))
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in order])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in order])
        self.assertEqual([c.name for c in traversal
                          if c._is_container],
                         [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal
                          if c._is_container],
                         [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            descend=descend))
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in order])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in order])
        self.assertEqual([c.name for c in traversal
                          if c._is_container],
                         [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal
                          if c._is_container],
                         [id(c) for c in descend.seen])
        return cdict, traversal


class _TestActiveDictContainerBase(_TestDictContainerBase):

    def test_active_type(self):
        cdict = self._container_type()
        self.assertTrue(isinstance(cdict, ICategorizedObject))
        self.assertTrue(isinstance(cdict, ICategorizedObjectContainer))
        self.assertTrue(isinstance(cdict, IHomogeneousContainer))
        self.assertTrue(isinstance(cdict, DictContainer))
        self.assertTrue(isinstance(cdict, collections_Mapping))
        self.assertTrue(isinstance(cdict, collections_MutableMapping))

    def test_active(self):
        children = {}
        children['a'] = self._ctype_factory()
        children[1] = self._ctype_factory()
        children[None] = self._ctype_factory()
        children[(1,)] = self._ctype_factory()
        children[(1, 2)] = self._ctype_factory()
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
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())),
                         len(list(cdict.components(active=True))))

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(cdict.components())),
                            len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)

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
        self.assertNotEqual(len(list(cdict.components())),
                            len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)

        del cdict[test_key]
        children[test_key].activate()
        self.assertEqual(children[test_key].active, True)
        cdict[test_key] = children[test_key]

        self.assertEqual(m.active, False)
        self.assertEqual(bdict.active, False)
        self.assertEqual(bdict[None].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(cdict.active, False)
        cdict.activate()
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
        for c in cdict.components():
            if c.storage_key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(cdict.components())),
                            len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 1)


        cdict.deactivate()
        m.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())),
                         len(list(cdict.components(active=True))))

        cdict.deactivate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, False)
        for c in cdict.values():
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(cdict.components())),
                            len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 0)

        cdict.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, True)
        for c in cdict.values():
            self.assertEqual(c.active, True)
        for c in cdict.components():
            self.assertEqual(c.active, True)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(cdict.components())), len(cdict))
        self.assertEqual(len(list(cdict.components())),
                         len(list(cdict.components(active=True))))

        cdict.deactivate(shallow=False)
        cdict[test_key].activate()

        self.assertEqual(m.active, True)
        self.assertEqual(bdict.active, True)
        self.assertEqual(bdict[None].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(cdict.active, False)
        cdict.activate()
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
        for c in cdict.components():
            if c.storage_key == test_key:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in cdict.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(cdict.components())),
                            len(list(cdict.components(active=None))))
        self.assertEqual(len(list(cdict.components(active=True))), 1)

    def test_preorder_traversal(self):
        cdict, traversal = \
            super(_TestActiveDictContainerBase, self).\
            test_preorder_traversal()

        descend = lambda x: not x._is_heterogeneous_container

        cdict[1].deactivate()
        self.assertEqual([None, '[0]', '[2]'],
                         [c.name for c in pmo.preorder_traversal(
                             cdict,
                             active=True,
                             descend=descend)])
        self.assertEqual([id(cdict),id(cdict[0]),id(cdict[2])],
                         [id(c) for c in pmo.preorder_traversal(
                             cdict,
                             active=True,
                             descend=descend)])

        cdict[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in pmo.preorder_traversal(
                             cdict,
                             active=True,
                             descend=descend)])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in pmo.preorder_traversal(
                             cdict,
                             active=True,
                             descend=descend)])

        cdict.deactivate()
        self.assertEqual(len(list(pmo.preorder_traversal(cdict,
                                                         active=True))),
                         0)
        self.assertEqual(len(list(pmo.generate_names(cdict,
                                                     active=True))),
                         0)

    def test_preorder_traversal_descend_check(self):
        cdict, traversal = \
            super(_TestActiveDictContainerBase, self).\
            test_preorder_traversal_descend_check()

        cdict[1].deactivate()
        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        pmo.pprint(cdict)
        order = list(pmo.preorder_traversal(cdict,
                                            active=True,
                                            descend=descend))
        self.assertEqual([None, '[0]', '[2]'],
                         [c.name for c in order])
        self.assertEqual([id(cdict),id(cdict[0]),id(cdict[2])],
                         [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None, '[0]', '[2]'],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict),id(cdict[0]),id(cdict[2])],
                             [id(c) for c in descend.seen])
        else:
            self.assertEqual([None],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict)],
                             [id(c) for c in descend.seen])

        def descend(x):
            self.assertTrue(x._is_container)
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            active=None,
                                            descend=descend))
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in order])
        self.assertEqual([id(cdict),id(cdict[0]),id(cdict[1]),id(cdict[2])],
                         [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None,'[0]','[1]','[2]'],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict),id(cdict[0]),id(cdict[1]),id(cdict[2])],
                             [id(c) for c in descend.seen])
        else:
            self.assertEqual([None,'[1]'],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict),id(cdict[1])],
                             [id(c) for c in descend.seen])

        cdict[1].deactivate(shallow=False)
        def descend(x):
            descend.seen.append(x)
            return not x._is_heterogeneous_container
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            active=True,
                                            descend=descend))
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in order])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in order])
        self.assertEqual([c.name for c in traversal
                          if c.active and \
                          c._is_container],
                         [c.name for c in descend.seen])
        self.assertEqual([id(c) for c in traversal
                          if c.active and \
                          c._is_container],
                         [id(c) for c in descend.seen])

        def descend(x):
            descend.seen.append(x)
            return x.active and (not x._is_heterogeneous_container)
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            active=None,
                                            descend=descend))
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in order])
        self.assertEqual([id(cdict),id(cdict[0]),id(cdict[1]),id(cdict[2])],
                         [id(c) for c in order])
        if cdict.ctype._is_heterogeneous_container:
            self.assertEqual([None,'[0]','[1]','[2]'],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict),id(cdict[0]),id(cdict[1]),id(cdict[2])],
                             [id(c) for c in descend.seen])
        else:
            self.assertEqual([None,'[1]'],
                             [c.name for c in descend.seen])
            self.assertEqual([id(cdict),id(cdict[1])],
                             [id(c) for c in descend.seen])

        cdict.deactivate()
        def descend(x):
            descend.seen.append(x)
            return True
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            active=True,
                                            descend=descend))
        self.assertEqual(len(descend.seen), 0)
        self.assertEqual(len(list(pmo.generate_names(cdict,
                                                     active=True))),
                         0)

        def descend(x):
            descend.seen.append(x)
            return x.active
        descend.seen = []
        order = list(pmo.preorder_traversal(cdict,
                                            active=None,
                                            descend=descend))
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], cdict)


if __name__ == "__main__":
    unittest.main()
