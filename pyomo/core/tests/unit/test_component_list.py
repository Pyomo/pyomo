#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import random
import pickle
import collections

import pyutilib.th as unittest
import pyomo.environ
from pyomo.common.log import LoggingIntercept
from pyomo.core.kernel.component_interface import \
    (ICategorizedObject,
     IComponent,
     IComponentContainer,
     _ActiveObjectMixin)
from pyomo.core.kernel.component_list import (ComponentList,
                                              create_component_list)
from pyomo.core.kernel.component_block import (IBlockStorage,
                                               block,
                                               block_list)

import six
from six import StringIO

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

    def test_overwrite_warning(self):
        c = self._container_type()
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            c.append(self._ctype_factory())
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

    def test_ctype(self):
        c = self._container_type()
        ctype = self._ctype_factory().ctype
        self.assertIs(c.ctype, ctype)
        self.assertIs(type(c).ctype, ctype)
        self.assertIs(self._container_type.ctype, ctype)

    def test_init1(self):
        clist = self._container_type()

    def test_init2(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        with self.assertRaises(TypeError):
            d = self._container_type(
                *tuple(self._ctype_factory() for i in index))

    def test_type(self):
        clist = self._container_type()
        self.assertTrue(isinstance(clist, ICategorizedObject))
        self.assertTrue(isinstance(clist, IComponentContainer))
        self.assertFalse(isinstance(clist, IComponent))
        self.assertTrue(isinstance(clist, ComponentList))
        self.assertTrue(isinstance(clist, collections.Sequence))
        self.assertTrue(issubclass(type(clist), collections.Sequence))
        self.assertTrue(isinstance(clist, collections.MutableSequence))
        self.assertTrue(issubclass(type(clist), collections.MutableSequence))

    def test_len1(self):
        c = self._container_type()
        self.assertEqual(len(c), 0)

    def test_len2(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))

    def test_append(self):
        c = self._container_type()
        index = range(5)
        self.assertEqual(len(c), 0)
        for i in index:
            c_new = self._ctype_factory()
            c.append(c_new)
            self.assertEqual(id(c[-1]), id(c_new))
            self.assertEqual(len(c), i+1)

    def test_insert(self):
        c = self._container_type()
        index = range(5)
        self.assertEqual(len(c), 0)
        for i in index:
            c_new = self._ctype_factory()
            c.insert(0, c_new)
            self.assertEqual(id(c[0]), id(c_new))
            self.assertEqual(len(c), i+1)

    def test_setitem(self):
        c = self._container_type()
        index = range(5)
        for i in index:
            c.append(self._ctype_factory())
        for i in index:
            c_new = self._ctype_factory()
            self.assertNotEqual(id(c_new), id(c[i]))
            c[i] = c_new
            self.assertEqual(len(c), len(index))
            self.assertEqual(id(c_new), id(c[i]))

    def test_wrong_type_init(self):
        index = range(5)
        with self.assertRaises(TypeError):
            c = self._container_type(
                _bad_ctype() for i in index)

    def test_wrong_type_append(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        with self.assertRaises(TypeError):
            c.append(_bad_ctype())

    def test_wrong_type_insert(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        c.insert(0, self._ctype_factory())
        with self.assertRaises(TypeError):
            c.insert(0, _bad_ctype())

    def test_wrong_type_setitem(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        c[0] = self._ctype_factory()
        with self.assertRaises(TypeError):
            c[0] = _bad_ctype()

    def test_has_parent_init(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            c.append(c[0])
        with self.assertRaises(ValueError):
            d = self._container_type(c)

    def test_has_parent_append(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            c.append(c[0])
        d = []
        d.append(c[0])
        d = self._container_type()
        with self.assertRaises(ValueError):
            d.append(c[0])

    def test_has_parent_insert(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        c.insert(0, self._ctype_factory())
        with self.assertRaises(ValueError):
            c.insert(0, c[0])
        d = []
        d.insert(0, c[0])
        d = self._container_type()
        with self.assertRaises(ValueError):
            d.insert(0, c[0])

    def test_has_parent_setitem(self):
        c = self._container_type()
        c.append(self._ctype_factory())
        c[0] = self._ctype_factory()
        c[0] = c[0]
        c.append(self._ctype_factory())
        with self.assertRaises(ValueError):
            c[0] = c[1]

    def test_setitem_exists_overwrite(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))
        for i in index:
            cdata = c[i]
            self.assertEqual(id(cdata.parent),
                             id(c))
            c[i] = self._ctype_factory()
            self.assertEqual(len(c), len(index))
            self.assertNotEqual(id(cdata), id(c[i]))
            self.assertEqual(cdata.parent, None)

    def test_delitem(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))
        for i in index:
            cdata = c[0]
            self.assertEqual(id(cdata.parent),
                             id(c))
            del c[0]
            self.assertEqual(len(c), len(index)-(i+1))
            self.assertEqual(cdata.parent, None)

    def test_iter(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))
        raw_list = c[:]
        self.assertEqual(type(raw_list), list)
        for c1, c2 in zip(raw_list, c):
            self.assertEqual(id(c1), id(c2))

    def test_reverse(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        raw_list = c[:]
        self.assertEqual(type(raw_list), list)
        for c1, c2 in zip(reversed(c), reversed(raw_list)):
            self.assertEqual(id(c1), id(c2))
        c.reverse()
        raw_list.reverse()
        for c1, c2 in zip(c, raw_list):
            self.assertEqual(id(c1), id(c2))

    def test_remove(self):
        model = block()
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = c[0]
            self.assertEqual(cdata in c, True)
            c.remove(cdata)
            self.assertEqual(cdata in c, False)

    def test_pop(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = c[-1]
            self.assertEqual(cdata in c, True)
            last = c.pop()
            self.assertEqual(cdata in c, False)
            self.assertEqual(id(cdata), id(last))

    def test_index(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            cdata = c[i]
            self.assertEqual(c.index(cdata), i)
            self.assertEqual(c.index(cdata, start=i), i)
            with self.assertRaises(ValueError):
                c.index(cdata, start=i+1)
            with self.assertRaises(ValueError):
                c.index(cdata, start=i, stop=i)
            with self.assertRaises(ValueError):
                c.index(cdata, stop=i)
            self.assertEqual(
                c.index(cdata, start=i, stop=i+1), i)
            with self.assertRaises(ValueError):
                c.index(cdata, start=i+1, stop=i+1)
            self.assertEqual(
                c.index(cdata, start=-len(index)+i), i)
            if i == index[-1]:
                self.assertEqual(
                    c.index(cdata, start=-len(index)+i+1), i)
            else:
                with self.assertRaises(ValueError):
                    self.assertEqual(
                        c.index(cdata, start=-len(index)+i+1),
                        i)
            if i == index[-1]:
                with self.assertRaises(ValueError):
                    self.assertEqual(
                        c.index(cdata, stop=-len(index)+i+1), i)
            else:
                self.assertEqual(
                    c.index(cdata, stop=-len(index)+i+1), i)
        tmp = self._ctype_factory()
        with self.assertRaises(ValueError):
            c.index(tmp)
        with self.assertRaises(ValueError):
            c.index(tmp, stop=len(c)+1)

    def test_extend(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        c_more_list = [self._ctype_factory() for i in index]
        self.assertEqual(len(c), len(index))
        self.assertTrue(len(c_more_list) > 0)
        for cdata in c_more_list:
            self.assertEqual(cdata.parent, None)
        c.extend(c_more_list)
        for cdata in c_more_list:
            self.assertEqual(id(cdata.parent),
                             id(c))

    def test_count(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            self.assertEqual(c.count(c[i]), 1)

    def test_pickle(self):
        index = range(5)
        clist = self._container_type(
            self._ctype_factory() for i in index)
        clist.append(self._container_type())
        index = list(index)
        index = index + [len(index)]
        for i in index:
            self.assertTrue(clist[i].parent is clist)
        pickled_clist = pickle.loads(
            pickle.dumps(clist, protocol=_pickle_test_protocol))
        self.assertTrue(
            isinstance(pickled_clist, self._container_type))
        self.assertTrue(pickled_clist.parent is None)
        self.assertEqual(len(pickled_clist), len(index))
        self.assertNotEqual(id(pickled_clist), id(clist))
        for i in index:
            self.assertNotEqual(id(pickled_clist[i]), id(clist[i]))
            self.assertTrue(pickled_clist[i].parent is pickled_clist)
            self.assertTrue(clist[i].parent is clist)

    def test_eq(self):
        clist1 = self._container_type()
        clist1.append(self._ctype_factory())
        clist2 = self._container_type()
        clist2.append(self._ctype_factory())

        self.assertNotEqual(clist1, set())
        self.assertFalse(clist1 == set())
        self.assertNotEqual(clist1, list())
        self.assertFalse(clist1 == list())
        self.assertNotEqual(clist1, tuple())
        self.assertFalse(clist1 == tuple())
        self.assertNotEqual(clist1, dict())
        self.assertFalse(clist1 == dict())

        self.assertTrue(clist1 == clist1)
        self.assertEqual(clist1, clist1)
        self.assertTrue(clist1 == list(clist1))
        self.assertEqual(clist1, list(clist1))
        self.assertTrue(clist1 == tuple(clist1))
        self.assertEqual(clist1, tuple(clist1))
        self.assertTrue(list(clist1) == clist1)
        self.assertEqual(list(clist1), clist1)
        self.assertTrue(tuple(clist1) == clist1)
        self.assertEqual(tuple(clist1), clist1)

        self.assertFalse(clist2 == clist1)
        self.assertTrue(clist2 != clist1)
        self.assertNotEqual(clist2, clist1)

        self.assertFalse(clist1 == clist2)
        self.assertTrue(clist1 != clist2)
        self.assertNotEqual(clist1, clist2)

        clist1 = self._container_type()

        self.assertEqual(clist1, set())
        self.assertTrue(clist1 == set())
        self.assertEqual(clist1, list())
        self.assertTrue(clist1 == list())
        self.assertEqual(clist1, tuple())
        self.assertTrue(clist1 == tuple())
        self.assertNotEqual(clist1, dict())
        self.assertFalse(clist1 == dict())

    def test_child(self):
        clist = self._container_type()
        c = self._ctype_factory()
        clist.append(c)
        with self.assertRaises(KeyError):
            clist.child(1)
        self.assertIs(clist.child(0), c)

    def test_name(self):
        children = [self._ctype_factory() for i in range(5)]
        children.append(self._container_type())
        children[-1].append(self._ctype_factory())

        for c in children:
            self.assertTrue(c.parent is None)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.local_name, None)
            self.assertEqual(c.name, None)

        clist = self._container_type()
        self.assertTrue(clist.parent is None)
        self.assertTrue(clist.parent_block is None)
        self.assertTrue(clist.root_block is None)
        self.assertEqual(clist.local_name, None)
        self.assertEqual(clist.name, None)
        clist.extend(children)
        names = clist.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is clist)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in clist.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        model = block()
        model.clist = clist
        self.assertTrue(model.parent is None)
        self.assertTrue(model.parent_block is None)
        self.assertTrue(model.root_block is model)
        self.assertTrue(clist.parent is model)
        self.assertTrue(clist.parent_block is model)
        self.assertTrue(clist.root_block is model)
        self.assertEqual(clist.local_name, "clist")
        self.assertEqual(clist.name, "clist")
        names = model.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is clist)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is model)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "clist[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in clist.components():
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
        self.assertTrue(clist.parent is model)
        self.assertTrue(clist.parent_block is model)
        self.assertTrue(clist.root_block is b)
        self.assertEqual(clist.local_name, "clist")
        self.assertEqual(clist.name, "model.clist")
        names = b.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is clist)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "model.clist[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in clist.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        blist = block_list()
        blist.append(b)
        self.assertTrue(blist.parent is None)
        self.assertTrue(blist.parent_block is None)
        self.assertTrue(blist.root_block is None)
        self.assertTrue(b.parent is blist)
        self.assertTrue(b.parent_block is None)
        self.assertTrue(b.root_block is b)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is b)
        self.assertTrue(clist.parent is model)
        self.assertTrue(clist.parent_block is model)
        self.assertTrue(clist.root_block is b)
        self.assertEqual(clist.local_name, "clist")
        self.assertEqual(clist.name, "[0].model.clist")
        for i, c in enumerate(children):
            self.assertTrue(c.parent is clist)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name,
                             "[0].model.clist[%s]" % (i))

        m = block()
        m.blist = blist
        self.assertTrue(m.parent is None)
        self.assertTrue(m.parent_block is None)
        self.assertTrue(m.root_block is m)
        self.assertTrue(blist.parent is m)
        self.assertTrue(blist.parent_block is m)
        self.assertTrue(blist.root_block is m)
        self.assertTrue(b.parent is blist)
        self.assertTrue(b.parent_block is m)
        self.assertTrue(b.root_block is m)
        self.assertTrue(model.parent is b)
        self.assertTrue(model.parent_block is b)
        self.assertTrue(model.root_block is m)
        self.assertTrue(clist.parent is model)
        self.assertTrue(clist.parent_block is model)
        self.assertTrue(clist.root_block is m)
        self.assertEqual(clist.local_name, "clist")
        self.assertEqual(clist.name, "blist[0].model.clist")
        names = m.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is clist)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is m)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name,
                             "blist[0].model.clist[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in clist.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])
        names = m.generate_names(descend_into=False)
        self.assertEqual(len(names), len(list(m.children())))
        for c in m.children():
            self.assertEqual(c.name, names[c])

    def test_components(self):
        clist = self._container_type()
        self.assertEqual(list(clist.components()), [])

        clistflattened = []
        clistflattened.append(self._ctype_factory())
        clist.append(clistflattened[-1])
        clistflattened.append(self._ctype_factory())
        clist.append(clistflattened[-1])
        self.assertEqual(list(id(_c) for _c in clist.components()),
                         list(id(_c) for _c in clistflattened))

        csublist = self._container_type()
        self.assertEqual(list(csublist.components()), [])

        csublistflattened = []
        csublistflattened.append(self._ctype_factory())
        clistflattened.append(csublistflattened[-1])
        csublist.append(csublistflattened[-1])

        csublistflattened.append(self._ctype_factory())
        clistflattened.append(csublistflattened[-1])
        csublist.append(csublistflattened[-1])

        csublistflattened.append(self._ctype_factory())
        clistflattened.append(csublistflattened[-1])
        csublist.append(csublistflattened[-1])

        self.assertEqual(list(id(_c) for _c in csublist.components()),
                         list(id(_c) for _c in csublistflattened))
        self.assertEqual(len(set(id(_c) for _c in csublist.components())),
                         len(list(id(_c) for _c in csublist.components())))
        self.assertEqual(len(set(id(_c) for _c in csublist.components())),
                         3)

        clist.append(csublist)
        self.assertEqual(list(id(_c) for _c in clist.components()),
                         list(id(_c) for _c in clistflattened))
        self.assertEqual(len(set(id(_c) for _c in clist.components())),
                         len(list(id(_c) for _c in clist.components())))
        self.assertEqual(len(set(id(_c) for _c in clist.components())),
                         5)

    def test_preorder_traversal(self):
        traversal = []
        clist = self._container_type()
        traversal.append(clist)
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])
        clist.append(self._container_type())
        traversal.append(clist[-1])
        clist[1].append(self._ctype_factory())
        traversal.append(clist[1][-1])
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])

        self.assertEqual([c.name for c in traversal],
                         [c.name for c in clist.preorder_traversal()])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in clist.preorder_traversal()])

        return clist, traversal

    def test_preorder_visit(self):
        traversal = []
        clist = self._container_type()
        traversal.append(clist)
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])
        clist.append(self._container_type())
        traversal.append(clist[-1])
        clist[1].append(self._ctype_factory())
        traversal.append(clist[1][-1])
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])

        def visit(x):
            visit.traversal.append(x)
            return False
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual(len(visit.traversal), 1)
        self.assertIs(visit.traversal[0], clist)

        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in visit.traversal])
        return clist, traversal

    def test_postorder_traversal(self):
        traversal = []
        clist = self._container_type()
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])
        clist.append(self._container_type())
        clist[1].append(self._ctype_factory())
        traversal.append(clist[1][-1])
        traversal.append(clist[-1])
        clist.append(self._ctype_factory())
        traversal.append(clist[-1])
        traversal.append(clist)

        self.assertEqual([c.name for c in traversal],
                         [c.name for c in clist.postorder_traversal()])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in clist.postorder_traversal()])

        return clist, traversal

    def test_create_component_list(self):
        clist1 = self._container_type(
            self._ctype_factory()
            for i in range(5))
        self.assertEqual(len(clist1), 5)
        for obj in clist1:
            self.assertIs(obj.parent, clist1)
        objects = iter(clist1)
        def type_(x, y=None):
            self.assertEqual(x, 1)
            self.assertEqual(y, 'a')
        type_ = lambda x, y=None: six.next(objects)
        type_.ctype = clist1.ctype
        # this will result in clist1 and clist2
        # being "equal" in that they both store the
        # same objects, except that clist2 has stolen
        # ownership of the objects from clist1 (all of the
        # .parent weakrefs have been changed)
        clist2 = create_component_list(self._container_type,
                                       type_,
                                       5, 1, y='a')
        self.assertEqual(len(clist2), 5)
        self.assertEqual(clist1, clist2)
        self.assertIsNot(clist1, clist2)
        for obj in clist1:
            self.assertIs(obj.parent, clist2)
        for obj in clist2:
            self.assertIs(obj.parent, clist2)

class _TestActiveComponentListBase(_TestComponentListBase):

    def test_active_type(self):
        clist = self._container_type()
        self.assertTrue(isinstance(clist, IComponentContainer))
        self.assertTrue(isinstance(clist, ICategorizedObject))
        self.assertTrue(isinstance(clist, _ActiveObjectMixin))
        self.assertFalse(isinstance(clist, IComponent))

    def test_active(self):
        index = list(range(4))
        clist = self._container_type(self._ctype_factory()
                                     for i in index)
        with self.assertRaises(AttributeError):
            clist.active = False
        for c in clist:
            with self.assertRaises(AttributeError):
                c.active = False

        model = block()
        model.clist = clist
        b = block()
        b.model = model
        blist = block_list()
        blist.append(b)
        blist.append(block())
        m = block()
        m.blist = blist

        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(clist.active, True)
        for c in clist:
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())),
                         len(list(clist.components(active=True))))

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 0)

        test_c = clist[0]
        clist.remove(test_c)
        clist.append(test_c)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 0)

        clist.remove(test_c)
        test_c.activate()
        self.assertEqual(test_c.active, True)
        self.assertEqual(clist.active, False)
        clist.append(test_c)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for c in clist:
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components():
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 1)

        m.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(clist.active, True)
        for c in clist:
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())),
                         len(list(clist.components(active=True))))

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for c in clist:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 0)

        clist[len(clist)-1] = self._ctype_factory()

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            if i == len(clist)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(clist.components()):
            if i == len(clist)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 1)

        clist.activate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())),
                         len(list(clist.components(active=True))))

        clist.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 0)

        clist[-1].activate()

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, False)
        clist.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            if i == len(clist)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(clist.components()):
            if i == len(clist)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(clist.components())),
                            len(list(clist.components(active=True))))
        self.assertEqual(len(list(clist.components(active=True))), 1)

        clist.deactivate(shallow=False)
        clist.activate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(clist.active, True)
        for i, c in enumerate(clist):
            self.assertEqual(c.active, True)
        for c in clist.components():
            self.assertEqual(c.active, True)
        for c in clist.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(clist.components())), len(clist))
        self.assertEqual(len(list(clist.components())),
                         len(list(clist.components(active=True))))

    def test_preorder_traversal(self):
        clist, traversal = \
            super(_TestActiveComponentListBase, self).\
            test_preorder_traversal()

        clist[1].deactivate()
        self.assertEqual([None,'[0]','[2]'],
                         [c.name for c in clist.preorder_traversal(
                             active=True)])
        self.assertEqual([id(clist),id(clist[0]),id(clist[2])],
                         [id(c) for c in clist.preorder_traversal(
                             active=True)])

        clist[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in clist.preorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in clist.preorder_traversal(
                             active=True)])

        clist.deactivate()
        self.assertEqual(len(list(clist.preorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(clist.generate_names(active=True))),
                         0)

    def test_preorder_visit(self):
        clist, traversal = \
            super(_TestActiveComponentListBase, self).\
            test_preorder_visit()

        clist[1].deactivate()
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        clist.preorder_visit(visit, active=True)
        self.assertEqual([None,'[0]','[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(clist),id(clist[0]),id(clist[2])],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(clist),id(clist[0]),id(clist[1]),id(clist[2])],
                         [id(c) for c in visit.traversal])

        clist[1].deactivate(shallow=False)
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        clist.preorder_visit(visit, active=True)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(clist),id(clist[0]),id(clist[1]),id(clist[2])],
                         [id(c) for c in visit.traversal])

        clist.deactivate()
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        clist.preorder_visit(visit, active=True)
        self.assertEqual(len(visit.traversal), 0)
        self.assertEqual(len(list(clist.generate_names(active=True))),
                         0)

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        clist.preorder_visit(visit)
        self.assertEqual(len(visit.traversal), 1)
        self.assertIs(visit.traversal[0], clist)

    def test_postorder_traversal(self):
        clist, traversal = \
            super(_TestActiveComponentListBase, self).\
            test_postorder_traversal()

        clist[1].deactivate()
        self.assertEqual(['[0]','[2]',None],
                         [c.name for c in clist.postorder_traversal(
                             active=True)])
        self.assertEqual([id(clist[0]),id(clist[2]),id(clist)],
                         [id(c) for c in clist.postorder_traversal(
                             active=True)])

        clist[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in clist.postorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in clist.postorder_traversal(
                             active=True)])

        clist.deactivate()
        self.assertEqual(len(list(clist.postorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(clist.generate_names(active=True))),
                         0)

if __name__ == "__main__":
    unittest.main()
