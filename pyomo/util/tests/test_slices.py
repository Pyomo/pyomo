#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest

from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
        get_index_if_present,
        list_from_possible_scalar,
        tuple_from_possible_scalar,
        get_component_call_stack,
        slice_component_along_sets,
        replace_indices,
        get_location_set_map,
#        get_locations_of_sets,
#        get_sets_of_locations,
        )

def model():
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[1,2,3])
    m.space = dae.ContinuousSet(initialize=[0,2])
    m.comp = pyo.Set(initialize=['a','b'])
    m.d_2 = pyo.Set(initialize=[('a',1),('b',2)])
    m.d_none = pyo.Set(initialize=[('c',1,10), ('d',3)], dimen=None)

    @m.Block()
    def b(b):

        @b.Block(m.time)
        def bb1(bb1, t):
            bb1.v0 = pyo.Var(initialize=1.)

        @b.Block(m.time, m.space)
        def bb2(bb2, t, x):
            bb2.v1 = pyo.Var(initialize=1.)

class TestGetComponentCallStack(unittest.TestCase):

    get_attribute = IndexedComponent_slice.get_attribute
    get_item = IndexedComponent_slice.get_item

    def assertSameStack(self, stack1, stack2):
        for (call1, arg1), (call2, arg2) in zip(stack1, stack2):
            self.assertIs(call1, call2)
            self.assertEqual(arg1, arg2)

    def assertCorrectStack(self, comp, pred_stack, context=None):
        act_stack = get_component_call_stack(comp, context=None)
        self.assertSameStack(pred_stack, act_stack)

    def model(self):

        m = pyo.ConcreteModel()
        m.s1 = pyo.Set(initialize=[1,2,3])
        m.s2 = pyo.Set(initialize=[('a',1),('b',2)])

        m.v0 = pyo.Var()
        m.v1 = pyo.Var(m.s1)
        m.v2 = pyo.Var(m.s2)

        @m.Block()
        def b(b):

            @b.Block(m.s1)
            def b1(b):
                b.v0 = pyo.Var()
                b.v1 = pyo.Var(m.s1)

                @b.Block()
                def b(b):
                    b.v0 = pyo.Var()
                    b.v2 = pyo.Var(m.s2)

                    @b.Block(m.s1, m.s2)
                    def b2(b,i1,i2,i3=None):
                        # Optional i3 is to allow for the case
                        # when normalize_index.flatten is False
                        b.v0 = pyo.Var()
                        b.v1 = pyo.Var(m.s1)

                        @b.Block()
                        def b(b):
                            b.v0 = pyo.Var()
                            b.v2 = pyo.Var(m.s2)
        return m

    def test_no_context(self):
        m = self.model()
        
        comp = m.v1[1]
        pred_stack = [
                (self.get_item, 1),
                (self.get_attribute, 'v1'), 
                ]
        self.assertCorrectStack(comp, pred_stack)

        comp = m.v1
        pred_stack = [
                (self.get_attribute, 'v1'),
                ]
        self.assertCorrectStack(comp, pred_stack)

        comp = m.b.b1[1].b.b2
        pred_stack = [
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 1),
                (self.get_attribute, 'b1'),
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack)

        comp = m.b.b1[1].b.b2[1,'a',1]
        pred_stack = [
                (self.get_item, (1,'a',1)),
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 1),
                (self.get_attribute, 'b1'),
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack)

        normalize_index.flatten = False
        comp = m.b.b1[1].b.b2[1,('a',1)]
        # NOTE: This fails without the parenthesis around ('a',1)
        # as comp.index() is None. The actual stack just starts one
        # layer higher...
        # It is confusing that comp can even be generated in this case;
        # shouldn't (1,'a',1) be an invalid index?
        pred_stack = [
                (self.get_item, (1,('a',1))),
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 1),
                (self.get_attribute, 'b1'),
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack)
        normalize_index.flatten = True

    def test_from_block(self):
        m = self.model()

        comp = m.v0
        pred_stack = [
                (self.get_attribute, 'v0'),
                ]
        self.assertCorrectStack(comp, pred_stack, context=m)

        comp = m.b.b1[2].b.b2[1,'a',1]
        pred_stack = [
                (self.get_item, (1,'a',1)),
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 2),
                (self.get_attribute, 'b1'),
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack, context=m)

        comp = m.b.b1[2].b.b2[1,'a',1].b.v2['b',2]
        pred_stack = [
                (self.get_item, ('b',2)),
                (self.get_attribute, 'v2'),
                (self.get_attribute, 'b'),
                (self.get_item, (1,'a',1)),
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 2),
                ]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)

        comp = m.b.b1[2].b.b2
        pred_stack = [
                (self.get_attribute, 'b2'),
                (self.get_attribute, 'b'),
                (self.get_item, 2),
                ]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)

        comp = m.b.b1[2]
        pred_stack = [
                (self.get_item, 2),
                ]
        self.assertCorrectStack(comp, pred_stack, context=m.b.b1)

        comp = m.b.b1
        act_stack = get_component_call_stack(comp, context=m.b.b1)
        self.assertEqual(len(act_stack), 0)

    def test_from_blockdata(self):

        m = self.model()

        context = m.b.b1[3].b.b2[2,'b',2]
        comp = m.b.b1[3].b.b2[2,'b',2].b
        pred_stack = [
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack, context=context)

        comp = m.b.b1[3].b.b2[2,'b',2].b.v2['a',1]
        pred_stack = [
                (self.get_item, ('a',1)),
                (self.get_attribute, 'v2'),
                (self.get_attribute, 'b'),
                ]
        self.assertCorrectStack(comp, pred_stack, context=context)

        context = m.b.b1[3]
        comp = m.b.b1[3]
        act_stack = get_component_call_stack(comp, context=context)
        self.assertEqual(len(act_stack), 0)

class TestGetLocationSetMap(unittest.TestCase):

    def model(self):

        m = pyo.ConcreteModel()

        m.time = pyo.Set(initialize=[1,2,3])
        m.space = dae.ContinuousSet(initialize=[0,2])
        m.comp = pyo.Set(initialize=['a','b'])
        m.d_2 = pyo.Set(initialize=[('a',1),('b',2)])
        m.d_none = pyo.Set(initialize=[('c',1,10), ('d',3)], dimen=None)

        m.b0 = pyo.Block()
        m.b1 = pyo.Block(m.time)
        m.b2 = pyo.Block(m.time, m.space)
        m.b0d2 = pyo.Block(m.d_2)
        m.b1d2 = pyo.Block(m.time, m.d_2)
        m.b2d2 = pyo.Block(m.time, m.d_2, m.space)
        m.b3d2 = pyo.Block(m.time, m.d_2, m.space, m.time)

        m.b0dn = pyo.Block(m.d_none)
        m.b1dn = pyo.Block(m.time, m.d_none)
        m.b1dnd2 = pyo.Block(m.time, m.d_none, m.d_2)
        m.dnd2b1 = pyo.Block(m.d_none, m.d_2, m.time)
        m.b2dn = pyo.Block(m.time, m.space, m.d_none)
        m.b2dnd2 = pyo.Block(m.time, m.d_none, m.d_2, m.space)
        m.b3dn = pyo.Block(m.time, m.d_2, m.d_none, m.space, m.d_2)

        m.dn2 = pyo.Block(m.time, m.d_none, m.d_2, m.d_none, m.d_2)

        return m

    def assertSameMap(self, pred, act):
        self.assertEqual(len(pred), len(act))
        for k, v in pred.items():
            self.assertIn(k, act)
            self.assertIs(pred[k], act[k])
        
    def test_one_to_one(self):
        m = self.model()

        index_set = m.b0.index_set()
        index = None
        pred_map = {}
        location_set_map = get_location_set_map(index, index_set)
        self.assertEqual(pred_map, location_set_map)

        index_set = m.b1.index_set()
        index = 1
        pred_map = {
                0: m.time,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b2.index_set()
        index = (1,2)
        pred_map = {
                0: m.time,
                1: m.space,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

    def test_multi_dim(self):
        m = self.model()
        index_set = m.b0d2.index_set()
        index = ('a',1)
        pred_map = {
                0: m.d_2,
                1: m.d_2,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        normalize_index.flatten = False
        # A single multi-dimensional index will behave the same when
        # normalize_index.flatten is False. This is probably unexpected
        # Should we just fail in this case?
        # Note that the index (('a',1),) will not work.
        index_set = m.b0d2.index_set()
        index = ('a',1)
        pred_map = {
                0: m.d_2,
                1: m.d_2,
                }
        with self.assertRaises(RuntimeError) as cm:
            location_set_map = get_location_set_map(index, index_set)
        self.assertIn('normalize_index.flatten', str(cm.exception))
        normalize_index.flatten = True

        index_set = m.b1d2.index_set()
        index = (1, 'a', 1)
        pred_map = {
                0: m.time,
                1: m.d_2,
                2: m.d_2,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b2d2.index_set()
        index = (1, 'a', 1, 2)
        pred_map = {
                0: m.time,
                1: m.d_2,
                2: m.d_2,
                3: m.space,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b3d2.index_set()
        index = (1, 'a', 1, 2, 1)
        pred_map = {
                0: m.time,
                1: m.d_2,
                2: m.d_2,
                3: m.space,
                4: m.time,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

    def test_dimen_none(self):
        m = self.model()

        index_set = m.b0dn.index_set()
        index = ('c',1,10)
        pred_map = {
                0: m.d_none,
                1: m.d_none,
                2: m.d_none,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b1dn.index_set()
        index = (1, 'c', 1, 10)
        pred_map = {
                0: m.time,
                1: m.d_none,
                2: m.d_none,
                3: m.d_none,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b1dnd2.index_set()
        index = (1, 'c', 1, 10, 'a', 1)
        pred_map = {
                0: m.time,
                1: m.d_none,
                2: m.d_none,
                3: m.d_none,
                4: m.d_2,
                5: m.d_2,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b2dn.index_set()
        index = (1, 0, 'd', 3)
        pred_map = {
                0: m.time,
                1: m.space,
                2: m.d_none,
                3: m.d_none,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b2dnd2.index_set()
        index = (1, 'c', 1, 10, 'b', 2, 0)
        pred_map = {
                0: m.time,
                1: m.d_none,
                2: m.d_none,
                3: m.d_none,
                4: m.d_2,
                5: m.d_2,
                6: m.space,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.dnd2b1.index_set()
        index = ('d', 3, 'b', 2, 1)
        pred_map = {
                0: m.d_none,
                1: m.d_none,
                2: m.d_2,
                3: m.d_2,
                4: m.time,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.b3dn.index_set()
        index = (1, 'a', 1, 'd', 3, 0, 'b', 2)
        pred_map = {
                0: m.time,
                1: m.d_2,
                2: m.d_2,
                3: m.d_none,
                4: m.d_none,
                5: m.space,
                6: m.d_2,
                7: m.d_2,
                }
        location_set_map = get_location_set_map(index, index_set)
        self.assertSameMap(pred_map, location_set_map)

        index_set = m.dn2.index_set()
        index = (1, 'c', 1, 10, 'b', 2, 'd', 3, 'a', 1)
        with self.assertRaises(RuntimeError) as cm:
            location_set_map = get_location_set_map(index, index_set)
        self.assertIn('multiple sets of dimen==None', str(cm.exception))

if __name__ == '__main__':
    unittest.main()
