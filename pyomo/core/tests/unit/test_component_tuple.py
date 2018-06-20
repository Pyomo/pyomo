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
from pyomo.core.kernel.component_interface import \
    (ICategorizedObject,
     IComponent,
     IComponentContainer,
     _ActiveObjectMixin)
from pyomo.core.kernel.component_tuple import (ComponentTuple,
                                               create_component_tuple)
from pyomo.core.kernel.component_block import (IBlockStorage,
                                               block,
                                               block_list)

import six

#
# There are no fully implemented test suites in this
# file. These classes are meant to be used to test
# full implementations of ComponentTuple containers.
# To test a ComponentTuple implementation, just import
# this class and use it as a subclass in another test suite.
#

# Note: we need to test with a pickle protocol
#       that is high enough to support __slots__
#       and weakref (bas
_pickle_test_protocol = pickle.HIGHEST_PROTOCOL

class _bad_ctype(object):
    ctype = "_this_is_definitely_not_the_ctype_being_tested"

class _TestComponentTupleBase(object):

    # set by derived class
    _container_type = None
    _ctype_factory = None

    def test_ctype(self):
        c = self._container_type()
        ctype = self._ctype_factory().ctype
        self.assertIs(c.ctype, ctype)
        self.assertIs(type(c).ctype, ctype)
        self.assertIs(self._container_type.ctype, ctype)

    def test_init1(self):
        ctuple = self._container_type()

    def test_init2(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        with self.assertRaises(TypeError):
            d = self._container_type(
                *tuple(self._ctype_factory() for i in index))

    def test_type(self):
        ctuple = self._container_type()
        self.assertTrue(isinstance(ctuple, ICategorizedObject))
        self.assertTrue(isinstance(ctuple, IComponentContainer))
        self.assertFalse(isinstance(ctuple, IComponent))
        self.assertTrue(isinstance(ctuple, ComponentTuple))
        self.assertTrue(isinstance(ctuple, collections.Sequence))
        self.assertTrue(issubclass(type(ctuple), collections.Sequence))

    def test_len1(self):
        c = self._container_type()
        self.assertEqual(len(c), 0)

    def test_len2(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))


    def test_wrong_type_init(self):
        index = range(5)
        with self.assertRaises(TypeError):
            c = self._container_type(
                _bad_ctype() for i in index)

    def test_has_parent_init(self):
        ctuple = self._container_type([self._ctype_factory()])
        with self.assertRaises(ValueError):
            self._container_type([ctuple[0]])

    def test_iter(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        self.assertEqual(len(c), len(index))
        raw_tuple = c[:]
        self.assertEqual(type(raw_tuple), tuple)
        for c1, c2 in zip(raw_tuple, c):
            self.assertEqual(id(c1), id(c2))

    def test_reverse(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        raw_tuple = c[:]
        self.assertEqual(type(raw_tuple), tuple)
        for c1, c2 in zip(reversed(c), reversed(raw_tuple)):
            self.assertEqual(id(c1), id(c2))

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

    def test_count(self):
        index = range(5)
        c = self._container_type(
            self._ctype_factory() for i in index)
        for i in index:
            self.assertEqual(c.count(c[i]), 1)

    def test_pickle(self):
        index = range(5)
        ctuple = self._container_type(
            [self._ctype_factory() for i in index] + \
            [self._container_type()])
        index = list(index)
        index = index + [len(index)]
        for i in index:
            self.assertTrue(ctuple[i].parent is ctuple)
        pickled_ctuple = pickle.loads(
            pickle.dumps(ctuple, protocol=_pickle_test_protocol))
        self.assertTrue(
            isinstance(pickled_ctuple, self._container_type))
        self.assertTrue(pickled_ctuple.parent is None)
        self.assertEqual(len(pickled_ctuple), len(index))
        self.assertNotEqual(id(pickled_ctuple), id(ctuple))
        for i in index:
            self.assertNotEqual(id(pickled_ctuple[i]), id(ctuple[i]))
            self.assertTrue(pickled_ctuple[i].parent is pickled_ctuple)
            self.assertTrue(ctuple[i].parent is ctuple)

    def test_eq(self):
        ctuple1 = self._container_type(
            [self._ctype_factory()])
        ctuple2 = self._container_type(
            [self._ctype_factory()])

        self.assertNotEqual(ctuple1, set())
        self.assertFalse(ctuple1 == set())
        self.assertNotEqual(ctuple1, list())
        self.assertFalse(ctuple1 == list())
        self.assertNotEqual(ctuple1, tuple())
        self.assertFalse(ctuple1 == tuple())
        self.assertNotEqual(ctuple1, dict())
        self.assertFalse(ctuple1 == dict())

        self.assertTrue(ctuple1 == ctuple1)
        self.assertEqual(ctuple1, ctuple1)
        self.assertTrue(ctuple1 == list(ctuple1))
        self.assertEqual(ctuple1, list(ctuple1))
        self.assertTrue(ctuple1 == tuple(ctuple1))
        self.assertEqual(ctuple1, tuple(ctuple1))
        self.assertTrue(list(ctuple1) == ctuple1)
        self.assertEqual(list(ctuple1), ctuple1)
        self.assertTrue(tuple(ctuple1) == ctuple1)
        self.assertEqual(tuple(ctuple1), ctuple1)

        self.assertFalse(ctuple2 == ctuple1)
        self.assertTrue(ctuple2 != ctuple1)
        self.assertNotEqual(ctuple2, ctuple1)

        self.assertFalse(ctuple1 == ctuple2)
        self.assertTrue(ctuple1 != ctuple2)
        self.assertNotEqual(ctuple1, ctuple2)

        ctuple1 = self._container_type()

        self.assertEqual(ctuple1, set())
        self.assertTrue(ctuple1 == set())
        self.assertEqual(ctuple1, list())
        self.assertTrue(ctuple1 == list())
        self.assertEqual(ctuple1, tuple())
        self.assertTrue(ctuple1 == tuple())
        self.assertNotEqual(ctuple1, dict())
        self.assertFalse(ctuple1 == dict())

    def test_child(self):
        ctuple = self._container_type()
        with self.assertRaises(KeyError):
            ctuple.child(0)
        c = self._ctype_factory()
        ctuple = self._container_type([c])
        self.assertIs(ctuple.child(0), c)

    def test_name(self):
        children = [self._ctype_factory() for i in range(5)]
        children.append(self._container_type(
            [self._ctype_factory()]))

        for c in children:
            self.assertTrue(c.parent is None)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.local_name, None)
            self.assertEqual(c.name, None)

        ctuple = self._container_type(children)
        self.assertTrue(ctuple.parent is None)
        self.assertTrue(ctuple.parent_block is None)
        self.assertTrue(ctuple.root_block is None)
        self.assertEqual(ctuple.local_name, None)
        self.assertEqual(ctuple.name, None)
        names = ctuple.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is ctuple)
            self.assertTrue(c.parent_block is None)
            if isinstance(c, IBlockStorage):
                self.assertTrue(c.root_block is c)
            else:
                self.assertTrue(c.root_block is None)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in ctuple.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])

        model = block()
        model.ctuple = ctuple
        self.assertTrue(model.parent is None)
        self.assertTrue(model.parent_block is None)
        self.assertTrue(model.root_block is model)
        self.assertTrue(ctuple.parent is model)
        self.assertTrue(ctuple.parent_block is model)
        self.assertTrue(ctuple.root_block is model)
        self.assertEqual(ctuple.local_name, "ctuple")
        self.assertEqual(ctuple.name, "ctuple")
        names = model.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is ctuple)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is model)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "ctuple[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in ctuple.components():
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
        self.assertTrue(ctuple.parent is model)
        self.assertTrue(ctuple.parent_block is model)
        self.assertTrue(ctuple.root_block is b)
        self.assertEqual(ctuple.local_name, "ctuple")
        self.assertEqual(ctuple.name, "model.ctuple")
        names = b.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is ctuple)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name, "model.ctuple[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in ctuple.components():
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
        self.assertTrue(ctuple.parent is model)
        self.assertTrue(ctuple.parent_block is model)
        self.assertTrue(ctuple.root_block is b)
        self.assertEqual(ctuple.local_name, "ctuple")
        self.assertEqual(ctuple.name, "[0].model.ctuple")
        for i, c in enumerate(children):
            self.assertTrue(c.parent is ctuple)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is b)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name,
                             "[0].model.ctuple[%s]" % (i))

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
        self.assertTrue(ctuple.parent is model)
        self.assertTrue(ctuple.parent_block is model)
        self.assertTrue(ctuple.root_block is m)
        self.assertEqual(ctuple.local_name, "ctuple")
        self.assertEqual(ctuple.name, "blist[0].model.ctuple")
        names = m.generate_names()
        for i, c in enumerate(children):
            self.assertTrue(c.parent is ctuple)
            self.assertTrue(c.parent_block is model)
            self.assertTrue(c.root_block is m)
            self.assertEqual(c.local_name, "[%s]" % (i))
            self.assertEqual(c.name,
                             "blist[0].model.ctuple[%s]" % (i))
            self.assertEqual(c.name, names[c])
        for c in ctuple.components():
            self.assertNotEqual(c.name, None)
            self.assertEqual(c.name, names[c])
        names = m.generate_names(descend_into=False)
        self.assertEqual(len(names), len(list(m.children())))
        for c in m.children():
            self.assertEqual(c.name, names[c])

    def test_components(self):
        ctuple = self._container_type()
        self.assertEqual(list(ctuple.components()), [])

        ctupleflattened = []
        ctupleflattened.append(self._ctype_factory())
        ctupleflattened.append(self._ctype_factory())

        csubtupleflattened = []
        csubtupleflattened.append(self._ctype_factory())
        ctupleflattened.append(csubtupleflattened[-1])

        csubtupleflattened.append(self._ctype_factory())
        ctupleflattened.append(csubtupleflattened[-1])

        csubtupleflattened.append(self._ctype_factory())
        ctupleflattened.append(csubtupleflattened[-1])

        csubtuple = self._container_type(csubtupleflattened)
        self.assertEqual(list(id(_c) for _c in csubtuple.components()),
                         list(id(_c) for _c in csubtupleflattened))
        self.assertEqual(len(set(id(_c) for _c in csubtuple.components())),
                         len(list(id(_c) for _c in csubtuple.components())))
        self.assertEqual(len(set(id(_c) for _c in csubtuple.components())),
                         3)

        ctuple = self._container_type([ctupleflattened[0],
                                       ctupleflattened[1],
                                       csubtuple])
        self.assertEqual(list(id(_c) for _c in ctuple.components()),
                         list(id(_c) for _c in ctupleflattened))
        self.assertEqual(len(set(id(_c) for _c in ctuple.components())),
                         len(list(id(_c) for _c in ctuple.components())))
        self.assertEqual(len(set(id(_c) for _c in ctuple.components())),
                         5)

    def test_preorder_traversal(self):

        csubtuple = self._container_type(
            [self._ctype_factory()])
        ctuple = self._container_type(
            [self._ctype_factory(),
             csubtuple,
             self._ctype_factory()])

        traversal = []
        traversal.append(ctuple)
        traversal.append(ctuple[0])
        traversal.append(ctuple[1])
        traversal.append(ctuple[1][0])
        traversal.append(ctuple[2])

        self.assertEqual([c.name for c in traversal],
                         [c.name for c in ctuple.preorder_traversal()])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in ctuple.preorder_traversal()])

        return ctuple, traversal

    def test_preorder_visit(self):

        csubtuple = self._container_type(
            [self._ctype_factory()])
        ctuple = self._container_type(
            [self._ctype_factory(),
             csubtuple,
             self._ctype_factory()])

        traversal = []
        traversal.append(ctuple)
        traversal.append(ctuple[0])
        traversal.append(ctuple[1])
        traversal.append(ctuple[1][0])
        traversal.append(ctuple[2])

        def visit(x):
            visit.traversal.append(x)
            return False
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual(len(visit.traversal), 1)
        self.assertIs(visit.traversal[0], ctuple)

        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual([c.name for c in traversal],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in visit.traversal])
        return ctuple, traversal

    def test_postorder_traversal(self):

        csubtuple = self._container_type(
            [self._ctype_factory()])
        ctuple = self._container_type(
            [self._ctype_factory(),
             csubtuple,
             self._ctype_factory()])

        traversal = []
        traversal.append(ctuple[0])
        traversal.append(ctuple[1][0])
        traversal.append(ctuple[1])
        traversal.append(ctuple[2])
        traversal.append(ctuple)

        self.assertEqual([c.name for c in traversal],
                         [c.name for c in ctuple.postorder_traversal()])
        self.assertEqual([id(c) for c in traversal],
                         [id(c) for c in ctuple.postorder_traversal()])

        return ctuple, traversal

    def test_create_component_tuple(self):
        ctuple1 = self._container_type(
            self._ctype_factory()
            for i in range(5))
        self.assertEqual(len(ctuple1), 5)
        for obj in ctuple1:
            self.assertIs(obj.parent, ctuple1)
        objects = iter(ctuple1)
        def type_(x, y=None):
            self.assertEqual(x, 1)
            self.assertEqual(y, 'a')
        type_ = lambda x, y=None: six.next(objects)
        type_.ctype = ctuple1.ctype
        # this will result in ctuple1 and ctuple2
        # being "equal" in that they both store the
        # same objects, except that ctuple2 has stolen
        # ownership of the objects from ctuple1 (all of the
        # .parent weakrefs have been changed)
        ctuple2 = create_component_tuple(self._container_type,
                                       type_,
                                       5, 1, y='a')
        self.assertEqual(len(ctuple2), 5)
        self.assertEqual(ctuple1, ctuple2)
        self.assertIsNot(ctuple1, ctuple2)
        for obj in ctuple1:
            self.assertIs(obj.parent, ctuple2)
        for obj in ctuple2:
            self.assertIs(obj.parent, ctuple2)

class _TestActiveComponentTupleBase(_TestComponentTupleBase):

    def test_active_type(self):
        ctuple = self._container_type()
        self.assertTrue(isinstance(ctuple, IComponentContainer))
        self.assertTrue(isinstance(ctuple, ICategorizedObject))
        self.assertTrue(isinstance(ctuple, _ActiveObjectMixin))
        self.assertFalse(isinstance(ctuple, IComponent))

    def test_active(self):
        index = list(range(4))
        ctuple = self._container_type(self._ctype_factory()
                                      for i in index)
        with self.assertRaises(AttributeError):
            ctuple.active = False
        for c in ctuple:
            with self.assertRaises(AttributeError):
                c.active = False

        model = block()
        model.ctuple = ctuple
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
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())),
                         len(list(ctuple.components(active=True))))

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())),
                            len(list(ctuple.components(active=True))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)

        test_c = ctuple[0]
        test_c.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        ctuple.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components():
            if c is test_c:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(ctuple.components())),
                            len(list(ctuple.components(active=True))))
        self.assertEqual(len(list(ctuple.components(active=True))), 1)

        m.activate(shallow=False)

        self.assertEqual(m.active, True)
        self.assertEqual(blist.active, True)
        self.assertEqual(blist[1].active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(model.active, True)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())),
                         len(list(ctuple.components(active=True))))

        m.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())),
                            len(list(ctuple.components(active=True))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)

        ctuple.activate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())),
                         len(list(ctuple.components(active=True))))

        ctuple.deactivate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, False)
        self.assertNotEqual(len(list(ctuple.components())),
                            len(list(ctuple.components(active=True))))
        self.assertEqual(len(list(ctuple.components(active=True))), 0)

        ctuple[-1].activate()

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, False)
        ctuple.activate()
        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            if i == len(ctuple)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for i, c in enumerate(ctuple.components()):
            if i == len(ctuple)-1:
                self.assertEqual(c.active, True)
            else:
                self.assertEqual(c.active, False)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertNotEqual(len(list(ctuple.components())),
                            len(list(ctuple.components(active=True))))
        self.assertEqual(len(list(ctuple.components(active=True))), 1)

        ctuple.deactivate(shallow=False)
        ctuple.activate(shallow=False)

        self.assertEqual(m.active, False)
        self.assertEqual(blist.active, False)
        self.assertEqual(blist[1].active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(model.active, False)
        self.assertEqual(ctuple.active, True)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.active, True)
        for c in ctuple.components():
            self.assertEqual(c.active, True)
        for c in ctuple.components(active=True):
            self.assertEqual(c.active, True)
        self.assertEqual(len(list(ctuple.components())), len(ctuple))
        self.assertEqual(len(list(ctuple.components())),
                         len(list(ctuple.components(active=True))))

    def test_preorder_traversal(self):
        ctuple, traversal = \
            super(_TestActiveComponentTupleBase, self).\
            test_preorder_traversal()

        ctuple[1].deactivate()
        self.assertEqual([None, '[0]', '[2]'],
                         [c.name for c in ctuple.preorder_traversal(
                             active=True)])
        self.assertEqual([id(ctuple),id(ctuple[0]),id(ctuple[2])],
                         [id(c) for c in ctuple.preorder_traversal(
                             active=True)])

        ctuple[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in ctuple.preorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in ctuple.preorder_traversal(
                             active=True)])

        ctuple.deactivate()
        self.assertEqual(len(list(ctuple.preorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(ctuple.generate_names(active=True))),
                         0)

    def test_preorder_visit(self):
        ctuple, traversal = \
            super(_TestActiveComponentTupleBase, self).\
            test_preorder_visit()

        ctuple[1].deactivate()
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit, active=True)
        self.assertEqual([None, '[0]', '[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(ctuple),id(ctuple[0]),id(ctuple[2])],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(ctuple),id(ctuple[0]),id(ctuple[1]),id(ctuple[2])],
                         [id(c) for c in visit.traversal])

        ctuple[1].deactivate(shallow=False)
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit, active=True)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in visit.traversal])

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual([None,'[0]','[1]','[2]'],
                         [c.name for c in visit.traversal])
        self.assertEqual([id(ctuple),id(ctuple[0]),id(ctuple[1]),id(ctuple[2])],
                         [id(c) for c in visit.traversal])

        ctuple.deactivate()
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit, active=True)
        self.assertEqual(len(visit.traversal), 0)
        self.assertEqual(len(list(ctuple.generate_names(active=True))),
                         0)

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual(len(visit.traversal), 1)
        self.assertIs(visit.traversal[0], ctuple)

        ctuple.deactivate(shallow=False)
        def visit(x):
            visit.traversal.append(x)
            return True
        visit.traversal = []
        ctuple.preorder_visit(visit, active=True)
        self.assertEqual(len(visit.traversal), 0)
        self.assertEqual(len(list(ctuple.generate_names(active=True))),
                         0)

        def visit(x):
            visit.traversal.append(x)
            return x.active
        visit.traversal = []
        ctuple.preorder_visit(visit)
        self.assertEqual(len(visit.traversal), 1)
        self.assertIs(visit.traversal[0], ctuple)

    def test_postorder_traversal(self):
        ctuple, traversal = \
            super(_TestActiveComponentTupleBase, self).\
            test_postorder_traversal()

        ctuple[1].deactivate()
        self.assertEqual(['[0]', '[2]', None],
                         [c.name for c in ctuple.postorder_traversal(
                             active=True)])
        self.assertEqual([id(ctuple[0]),id(ctuple[2]),id(ctuple)],
                         [id(c) for c in ctuple.postorder_traversal(
                             active=True)])

        ctuple[1].deactivate(shallow=False)
        self.assertEqual([c.name for c in traversal if c.active],
                         [c.name for c in ctuple.postorder_traversal(
                             active=True)])
        self.assertEqual([id(c) for c in traversal if c.active],
                         [id(c) for c in ctuple.postorder_traversal(
                             active=True)])

        ctuple.deactivate()
        self.assertEqual(len(list(ctuple.postorder_traversal(active=True))),
                         0)
        self.assertEqual(len(list(ctuple.generate_names(active=True))),
                         0)

if __name__ == "__main__":
    unittest.main()
