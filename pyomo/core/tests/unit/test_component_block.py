import tempfile
import os
import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 _IActiveComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_map import ComponentMap
from pyomo.core.base.component_constraint import constraint
from pyomo.core.base.component_variable import (IVariable,
                                                variable,
                                                variable_dict,
                                                variable_list)
from pyomo.core.base.component_block import (IBlockStorage,
                                             block,
                                             block_dict,
                                             block_list,
                                             StaticBlock)
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var


def _path_to_object_exists(obj, descendent):
    if descendent is obj:
        return True
    else:
        parent = descendent.parent
        if parent is None:
            return False
        else:
            return _path_to_object_exists(obj, parent)

def _active_path_to_object_exists(obj, descendent):
    if descendent is obj:
        return True
    else:
        parent = descendent.parent
        if parent is None:
            return False
        else:
            if getattr(descendent, "active", True):
                return _active_path_to_object_exists(obj, parent)
            else:
                return False

class _Test_block_base(object):

    _children = None
    _child_key = None
    _components_no_descend = None
    _components = None
    _blocks_no_descend = None
    _blocks = None
    _block = None

    def test_pickle(self):
        b = pickle.loads(
            pickle.dumps(self._block))
        self.assertEqual(len(list(b.preorder_traversal())),
                         len(self._names)+1)
        self.assertEqual(len(list(b.postorder_traversal())),
                         len(self._names)+1)
        names = b.generate_names()
        self.assertEqual(sorted(names.values()),
                         sorted(self._names.values()))

    def test_child_key(self):
        for child in self._child_key:
            parent = child.parent
            self.assertTrue(parent is not None)
            self.assertTrue(id(child) in set(
                id(_c) for _c in self._children[parent]))
            self.assertEqual(self._child_key[child],
                             parent.child_key(child))

    def test_child_key_no_entry(self):
        v = variable()
        with self.assertRaises(ValueError):
            self._block.child_key(v)

    def test_children(self):
        for obj in self._children:
            self.assertTrue(isinstance(obj, IComponentContainer))
            if isinstance(obj, IBlockStorage):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children()),
                    sorted(str(child)
                           for child in self._children[obj]))
                self.assertEqual(
                    set(id(child) for child in obj.children()),
                    set(id(child) for child in self._children[obj]))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children()))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=Block)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is Block))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=Block)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is Block))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(ctype=Block,
                                                                      return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children(ctype=Block)
                         if child.ctype is Block))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=Var)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is Var))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=Var)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is Var))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(ctype=Var,
                                                                      return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children(ctype=Var)
                         if child.ctype is Var))
            elif isinstance(obj, IComponentContainer):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                self.assertEqual(
                    set(id(child) for child in obj.children()),
                    set(id(child) for child in self._children[obj]))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children()))
            else:
                self.assertEqual(len(self._children[obj]), 0)

    def test_components_no_descend_active_None(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.components(descend_into=False):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][Block]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=None,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][Block]))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][Var]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=None,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][Var]))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=None,
                                      descend_into=False)),
                sorted(str(_c)
                       for ctype in
                       self._components_no_descend[obj]
                       for _c in
                       self._components_no_descend[obj][ctype]))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=None,
                                   descend_into=False)),
                set(id(_c) for ctype in
                    self._components_no_descend[obj]
                    for _c in
                    self._components_no_descend[obj][ctype]))

    def test_components_no_descend_active_True(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][Block]
                       if _b.active))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=True,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][Block]
                    if _b.active))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][Var]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=True,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][Var]))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=True,
                                      descend_into=False)),
                sorted(str(_c)
                       for ctype in
                       self._components_no_descend[obj]
                       for _c in
                       self._components_no_descend[obj][ctype]
                       if getattr(_c, "active", True)))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=True,
                                   descend_into=False)),
                set(id(_c) for ctype in
                    self._components_no_descend[obj]
                    for _c in
                    self._components_no_descend[obj][ctype]
                    if getattr(_c, "active", True)))

    def test_components_active_None(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.components(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][Block]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=None,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][Block]))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][Var]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=None,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][Var]))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=None,
                                      descend_into=True)),
                sorted(str(_c)
                       for ctype in
                       self._components[obj]
                       for _c in
                       self._components[obj][ctype]))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=None,
                                   descend_into=True)),
                set(id(_c) for ctype in
                    self._components[obj]
                    for _c in
                    self._components[obj][ctype]))

    def test_components_active_True(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][Block]
                       if _b.active))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=True,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][Block]
                    if _b.active))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][Var]
                       if _active_path_to_object_exists(obj, _v)))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=True,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][Var]
                    if _active_path_to_object_exists(obj, _v)))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=True,
                                      descend_into=True)),
                sorted(str(_c)
                       for ctype in
                       self._components[obj]
                       for _c in
                       self._components[obj][ctype]
                       if _active_path_to_object_exists(obj, _c)))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=True,
                                   descend_into=True)),
                set(id(_c) for ctype in
                    self._components[obj]
                    for _c in
                    self._components[obj][ctype]
                    if _active_path_to_object_exists(obj, _c)))

    def test_blocks_no_descend_active_None(self):
        for obj in self._blocks_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.blocks(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=None,
                                  descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._blocks_no_descend[obj]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=None,
                               descend_into=False)),
                set(id(_b) for _b in
                    self._blocks_no_descend[obj]))

    def test_blocks_no_descend_active_True(self):
        for obj in self._blocks_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=True,
                                  descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._blocks_no_descend[obj]
                       if _b.active))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=True,
                               descend_into=False)),
                set(id(_b) for _b in
                    self._blocks_no_descend[obj]
                    if _b.active))

    def test_blocks_active_None(self):
        for obj in self._blocks:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.blocks(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=None,
                                  descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._blocks[obj]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=None,
                               descend_into=True)),
                set(id(_b) for _b in
                    self._blocks[obj]))

    def test_blocks_active_True(self):
        for obj in self._blocks:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=True,
                                  descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._blocks[obj]
                       if _b.active))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=True,
                               descend_into=True)),
                set(id(_b) for _b in
                    self._blocks[obj]
                    if _b.active))

class Test_block(_Test_block_base, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = cls._block = block()
        model.v_1 = variable()
        model.vdict_1 = variable_dict()
        model.vdict_1[None] = variable()
        model.vlist_1 = variable_list()
        model.vlist_1.append(variable())
        model.vlist_1.append(variable())
        model.b_1 = block()
        model.b_1.v_2 = variable()
        model.b_1.b_2 = block()
        model.b_1.b_2.b_3 = block()
        model.b_1.b_2.v_3 = variable()
        model.b_1.b_2.vlist_3 = variable_list()
        model.b_1.b_2.vlist_3.append(variable())
        model.b_1.b_2.deactivate()
        model.bdict_1 = block_dict()
        model.blist_1 = block_list()
        model.blist_1.append(block())
        model.blist_1[0].v_2 = variable()
        model.blist_1[0].b_2 = block()

        #
        # Manually encode the correct output
        # for tests in the base testing class
        #

        cls._names = ComponentMap()
        cls._names[model.v_1] = "v_1"
        cls._names[model.vdict_1] = "vdict_1"
        cls._names[model.vdict_1[None]] = "vdict_1[None]"
        cls._names[model.vlist_1] = "vlist_1"
        cls._names[model.vlist_1[0]] = "vlist_1[0]"
        cls._names[model.vlist_1[1]] = "vlist_1[1]"
        cls._names[model.b_1] = "b_1"
        cls._names[model.b_1.v_2] = "b_1.v_2"
        cls._names[model.b_1.b_2] = "b_1.b_2"
        cls._names[model.b_1.b_2.b_3] = "b_1.b_2.b_3"
        cls._names[model.b_1.b_2.v_3] = "b_1.b_2.v_3"
        cls._names[model.b_1.b_2.vlist_3] = "b_1.b_2.vlist_3"
        cls._names[model.b_1.b_2.vlist_3[0]] = "b_1.b_2.vlist_3[0]"
        cls._names[model.bdict_1] = "bdict_1"
        cls._names[model.blist_1] = "blist_1"
        cls._names[model.blist_1[0]] = "blist_1[0]"
        cls._names[model.blist_1[0].v_2] = "blist_1[0].v_2"
        cls._names[model.blist_1[0].b_2] = "blist_1[0].b_2"

        cls._children = ComponentMap()
        cls._children[model] = [model.v_1,
                                model.vdict_1,
                                model.vlist_1,
                                model.b_1,
                                model.bdict_1,
                                model.blist_1]
        cls._children[model.vdict_1] = [model.vdict_1[None]]
        cls._children[model.vlist_1] = [model.vlist_1[0],
                                        model.vlist_1[1]]
        cls._children[model.b_1] = [model.b_1.v_2,
                                    model.b_1.b_2]
        cls._children[model.b_1.b_2] = [model.b_1.b_2.v_3,
                                        model.b_1.b_2.vlist_3,
                                        model.b_1.b_2.b_3]
        cls._children[model.b_1.b_2.b_3] = []
        cls._children[model.b_1.b_2.vlist_3] = \
            [model.b_1.b_2.vlist_3[0]]
        cls._children[model.bdict_1] = []
        cls._children[model.blist_1] = [model.blist_1[0]]
        cls._children[model.blist_1[0]] = [model.blist_1[0].v_2,
                                           model.blist_1[0].b_2]

        cls._child_key = ComponentMap()
        cls._child_key[model.v_1] = "v_1"
        cls._child_key[model.vdict_1] = "vdict_1"
        cls._child_key[model.vlist_1] = "vlist_1"
        cls._child_key[model.b_1] = "b_1"
        cls._child_key[model.bdict_1] = "bdict_1"
        cls._child_key[model.blist_1] = "blist_1"
        cls._child_key[model.vdict_1[None]] = None
        cls._child_key[model.vlist_1[0]] = 0
        cls._child_key[model.vlist_1[1]] = 1
        cls._child_key[model.b_1.v_2] = "v_2"
        cls._child_key[model.b_1.b_2] = "b_2"
        cls._child_key[model.b_1.b_2.b_3] = "b_3"
        cls._child_key[model.b_1.b_2.v_3] = "v_3"
        cls._child_key[model.b_1.b_2.vlist_3] = "vlist_3"
        cls._child_key[model.b_1.b_2.vlist_3[0]] = 0
        cls._child_key[model.blist_1[0]] = 0
        cls._child_key[model.blist_1[0].v_2] = "v_2"
        cls._child_key[model.blist_1[0].b_2] = "b_2"

        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][Var] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1]]
        cls._components_no_descend[model][Block] = \
            [model.b_1,
             model.blist_1[0]]
        cls._components_no_descend[model.b_1] = {}
        cls._components_no_descend[model.b_1][Var] = \
            [model.b_1.v_2]
        cls._components_no_descend[model.b_1][Block] = \
            [model.b_1.b_2]
        cls._components_no_descend[model.b_1.b_2] = {}
        cls._components_no_descend[model.b_1.b_2][Var] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components_no_descend[model.b_1.b_2][Block] = \
            [model.b_1.b_2.b_3]
        cls._components_no_descend[model.b_1.b_2.b_3] = {}
        cls._components_no_descend[model.b_1.b_2.b_3][Var] = []
        cls._components_no_descend[model.b_1.b_2.b_3][Block] = []
        cls._components_no_descend[model.blist_1[0]] = {}
        cls._components_no_descend[model.blist_1[0]][Var] = \
            [model.blist_1[0].v_2]
        cls._components_no_descend[model.blist_1[0]][Block] = \
            [model.blist_1[0].b_2]
        cls._components_no_descend[model.blist_1[0].b_2] = {}
        cls._components_no_descend[model.blist_1[0].b_2][Var] = []
        cls._components_no_descend[model.blist_1[0].b_2][Block] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][Var] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1],
             model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0],
             model.blist_1[0].v_2]
        cls._components[model][Block] = \
            [model.b_1,
             model.blist_1[0],
             model.b_1.b_2,
             model.b_1.b_2.b_3,
             model.blist_1[0].b_2]
        cls._components[model.b_1] = {}
        cls._components[model.b_1][Var] = \
            [model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1][Block] = \
            [model.b_1.b_2,
             model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2] = {}
        cls._components[model.b_1.b_2][Var] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1.b_2][Block] = \
            [model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2.b_3] = {}
        cls._components[model.b_1.b_2.b_3][Var] = []
        cls._components[model.b_1.b_2.b_3][Block] = []
        cls._components[model.blist_1[0]] = {}
        cls._components[model.blist_1[0]][Var] = \
            [model.blist_1[0].v_2]
        cls._components[model.blist_1[0]][Block] = \
            [model.blist_1[0].b_2]
        cls._components[model.blist_1[0].b_2] = {}
        cls._components[model.blist_1[0].b_2][Var] = []
        cls._components[model.blist_1[0].b_2][Block] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][Block]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][Block]

    def test_init(self):
        b = block()
        self.assertTrue(b.parent is None)
        self.assertEqual(b.ctype, Block)

    def test_type(self):
        b = block()
        self.assertTrue(isinstance(b, ICategorizedObject))
        self.assertTrue(isinstance(b, IActiveObject))
        self.assertTrue(isinstance(b, IComponentContainer))
        self.assertTrue(isinstance(b, _IActiveComponentContainer))
        self.assertTrue(isinstance(b, IBlockStorage))

    def test_overwrite(self):
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = variable()
        self.assertTrue(v.parent is None)

        # the same component can overwrite itself
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = v
        self.assertTrue(v.parent is b)

        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = constraint()
        self.assertTrue(c.parent is None)

        # the same component can overwrite itself
        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = c
        self.assertTrue(c.parent is b)

        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = constraint()
        self.assertTrue(v.parent is None)

        b = block()
        c = b.c = variable()
        self.assertIs(c.parent, b)
        b.c = variable()
        self.assertTrue(c.parent is None)

    def test_already_has_parent(self):
        b1 = block()
        v = b1.v = variable()
        b2 = block()
        with self.assertRaises(ValueError):
            b2.v = v
        self.assertTrue(v.parent is b1)
        del b1.v
        b2.v = v
        self.assertTrue(v.parent is b2)

    def test_collect_ctypes(self):
        b = block()
        self.assertEqual(b.collect_ctypes(),
                         set())
        self.assertEqual(b.collect_ctypes(active=True),
                         set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(),
                         set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var, Constraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(),
                         set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))
        B = block()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([Block, Var]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([Var]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([Var]))
        del b.y

        # a block DOES check its own .active flag apparently
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))

        del b.x
        self.assertEqual(b.collect_ctypes(), set())

class _MyBlock(StaticBlock):
    __slots__ = ("b", "v", "n")
    def __init__(self):
        self.b = block()
        self.b.v = variable()
        self.b.blist = block_list()
        self.b.blist.append(block())
        self.v = variable()
        self.n = 2.0
        super(_MyBlock, self).__init__()

class Test_StaticBlock(_Test_block_base, unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls._myblock_type = _MyBlock
        model = cls._block = _MyBlock()

        #
        # Manually encode the correct output
        # for tests in the base testing class
        #

        cls._names = ComponentMap()
        cls._names[model.b] = "b"
        cls._names[model.b.v] = "b.v"
        cls._names[model.b.blist] = "b.blist"
        cls._names[model.b.blist[0]] = "b.blist[0]"
        cls._names[model.v] = "v"

        cls._children = ComponentMap()
        cls._children[model] = [model.b,
                                model.v]
        cls._children[model.b] = [model.b.v,
                                  model.b.blist]
        cls._children[model.b.blist] = [model.b.blist[0]]
        cls._children[model.b.blist[0]] = []

        cls._child_key = ComponentMap()
        cls._child_key[model.b] = "b"
        cls._child_key[model.b.v] = "v"
        cls._child_key[model.b.blist] = "blist"
        cls._child_key[model.b.blist[0]] = 0
        cls._child_key[model.v] = "v"

        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][Var] = [model.v]
        cls._components_no_descend[model][Block] = [model.b]
        cls._components_no_descend[model.b] = {}
        cls._components_no_descend[model.b][Var] = [model.b.v]
        cls._components_no_descend[model.b][Block] = [model.b.blist[0]]
        cls._components_no_descend[model.b.blist[0]] = {}
        cls._components_no_descend[model.b.blist[0]][Block] = []
        cls._components_no_descend[model.b.blist[0]][Var] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][Var] = [model.v, model.b.v]
        cls._components[model][Block] = [model.b, model.b.blist[0]]
        cls._components[model.b] = {}
        cls._components[model.b][Var] = [model.b.v]
        cls._components[model.b][Block] = [model.b.blist[0]]
        cls._components[model.b.blist[0]] = {}
        cls._components[model.b.blist[0]][Block] = []
        cls._components[model.b.blist[0]][Var] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][Block]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][Block]

    def test_collect_ctypes(self):
        self.assertEqual(self._block.collect_ctypes(),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True,
                                                    descend_into=False),
                         set([Block, Var]))

        self._block.b.deactivate()
        self.assertEqual(self._block.collect_ctypes(),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True),
                         set([Var]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True,
                                                    descend_into=False),
                         set([Var]))
        self._block.b.activate()

class Test_block_dict(_TestActiveComponentDictBase,
                      unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: block()

class Test_block_list(_TestActiveComponentListBase,
                      unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: block()

if __name__ == "__main__":
    unittest.main()
