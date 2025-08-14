#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for Elements of a Block
#

from io import StringIO
import logging
import os
import pickle
import sys
import types
import json

from copy import deepcopy
from os.path import abspath, dirname, join

currdir = dirname(abspath(__file__))

import pyomo.common.unittest as unittest

from pyomo.environ import (
    AbstractModel,
    ConcreteModel,
    Var,
    Set,
    Param,
    Block,
    Suffix,
    Constraint,
    Component,
    Objective,
    Expression,
    Reference,
    SOSConstraint,
    SortComponents,
    NonNegativeIntegers,
    TraversalStrategy,
    RangeSet,
    SolverFactory,
    value,
    sum_product,
    ComponentUID,
    Any,
)
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
    ScalarBlock,
    SubclassOf,
    BlockData,
    declare_custom_block,
)
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers

from pyomo.gdp import Disjunct

solvers = check_available_solvers('glpk')


class DerivedBlock(ScalarBlock):
    def __init__(self, *args, **kwargs):
        """Constructor"""
        kwargs['ctype'] = DerivedBlock
        super(DerivedBlock, self).__init__(*args, **kwargs)

    def foo(self):
        pass


DerivedBlock._Block_reserved_words = set(dir(DerivedBlock()))


@declare_custom_block("FooBlock", rule="build")
class FooBlockData(BlockData):
    def build(self, *args, capex, opex):
        self.x = Var(list(args))
        self.y = Var()

        self.capex = capex
        self.opex = opex


class TestGenerators(unittest.TestCase):
    def generate_model(self):
        #
        # ** DO NOT modify the model below without updating the
        # component_lists and component_data_lists with the correct
        # declaration orders
        #

        model = ConcreteModel()
        model.q = Set(initialize=[1, 2])
        model.Q = Set(model.q, initialize=[1, 2])
        model.qq = NonNegativeIntegers * model.q
        model.x = Var(initialize=-1)
        model.X = Var(model.q, initialize=-1)
        model.e = Expression(initialize=-1)
        model.E = Expression(model.q, initialize=-1)
        model.p = Param(mutable=True, initialize=-1)
        model.P = Param(model.q, mutable=True, initialize=-1)
        model.o = Objective(expr=-1)
        model.O = Objective(model.q, rule=lambda model, i: -1)
        model.c = Constraint(expr=model.x >= -1)
        model.C = Constraint(model.q, rule=lambda model, i: model.X[i] >= -1)
        model.sos = SOSConstraint(var=model.X, index=model.q, sos=1)
        model.SOS = SOSConstraint(model.q, var=model.X, index=model.Q, sos=1)
        model.s = Suffix()

        model.b = Block()
        model.b.q = Set(initialize=[1, 2])
        model.b.Q = Set(model.b.q, initialize=[1, 2])
        model.b.x = Var(initialize=0)
        model.b.X = Var(model.b.q, initialize=0)
        model.b.e = Expression(initialize=0)
        model.b.E = Expression(model.b.q, initialize=0)
        model.b.p = Param(mutable=True, initialize=0)
        model.b.P = Param(model.b.q, mutable=True, initialize=0)
        model.b.o = Objective(expr=0)
        model.b.O = Objective(model.b.q, rule=lambda b, i: 0)
        model.b.c = Constraint(expr=model.b.x >= 0)
        model.b.C = Constraint(model.b.q, rule=lambda b, i: b.X[i] >= 0)
        model.b.sos = SOSConstraint(var=model.b.X, index=model.b.q, sos=1)
        model.b.SOS = SOSConstraint(model.b.q, var=model.b.X, index=model.b.Q, sos=1)
        model.b.s = Suffix()
        model.b.component_lists = {}
        model.b.component_data_lists = {}
        model.b.component_lists[Set] = [model.b.q, model.b.Q]
        model.b.component_data_lists[Set] = [model.b.q, model.b.Q[1], model.b.Q[2]]
        model.b.component_lists[Var] = [model.b.x, model.b.X]
        model.b.component_data_lists[Var] = [model.b.x, model.b.X[1], model.b.X[2]]
        model.b.component_lists[Expression] = [model.b.e, model.b.E]
        model.b.component_data_lists[Expression] = [
            model.b.e,
            model.b.E[1],
            model.b.E[2],
        ]
        model.b.component_lists[Param] = [model.b.p, model.b.P]
        model.b.component_data_lists[Param] = [model.b.p, model.b.P[1], model.b.P[2]]
        model.b.component_lists[Objective] = [model.b.o, model.b.O]
        model.b.component_data_lists[Objective] = [
            model.b.o,
            model.b.O[1],
            model.b.O[2],
        ]
        model.b.component_lists[Constraint] = [model.b.c, model.b.C]
        model.b.component_data_lists[Constraint] = [
            model.b.c,
            model.b.C[1],
            model.b.C[2],
        ]
        model.b.component_lists[SOSConstraint] = [model.b.sos, model.b.SOS]
        model.b.component_data_lists[SOSConstraint] = [
            model.b.sos,
            model.b.SOS[1],
            model.b.SOS[2],
        ]
        model.b.component_lists[Suffix] = [model.b.s]
        model.b.component_data_lists[Suffix] = [model.b.s]
        model.b.component_lists[Block] = []
        model.b.component_data_lists[Block] = []

        def B_rule(block, i):
            block.q = Set(initialize=[1, 2])
            block.Q = Set(block.q, initialize=[1, 2])
            block.x = Var(initialize=i)
            block.X = Var(block.q, initialize=i)
            block.e = Expression(initialize=i)
            block.E = Expression(block.q, initialize=i)
            block.p = Param(mutable=True, initialize=i)
            block.P = Param(block.q, mutable=True, initialize=i)
            block.o = Objective(expr=i)
            block.O = Objective(block.q, rule=lambda b, i: i)
            block.c = Constraint(expr=block.x >= i)
            block.C = Constraint(block.q, rule=lambda b, i: b.X[i] >= i)
            block.sos = SOSConstraint(var=block.X, index=block.q, sos=1)
            block.SOS = SOSConstraint(block.q, var=block.X, index=block.Q, sos=1)
            block.s = Suffix()
            block.component_lists = {}
            block.component_data_lists = {}
            block.component_lists[Set] = [block.q, block.Q]
            block.component_data_lists[Set] = [block.q, block.Q[1], block.Q[2]]
            block.component_lists[Var] = [block.x, block.X]
            block.component_data_lists[Var] = [block.x, block.X[1], block.X[2]]
            block.component_lists[Expression] = [block.e, block.E]
            block.component_data_lists[Expression] = [block.e, block.E[1], block.E[2]]
            block.component_lists[Param] = [block.p, block.P]
            block.component_data_lists[Param] = [block.p, block.P[1], block.P[2]]
            block.component_lists[Objective] = [block.o, block.O]
            block.component_data_lists[Objective] = [block.o, block.O[1], block.O[2]]
            block.component_lists[Constraint] = [block.c, block.C]
            block.component_data_lists[Constraint] = [block.c, block.C[1], block.C[2]]
            block.component_lists[SOSConstraint] = [block.sos, block.SOS]
            block.component_data_lists[SOSConstraint] = [
                block.sos,
                block.SOS[1],
                block.SOS[2],
            ]
            block.component_lists[Suffix] = [block.s]
            block.component_data_lists[Suffix] = [block.s]
            block.component_lists[Block] = []
            block.component_data_lists[Block] = []

        model.B = Block(model.q, rule=B_rule)

        model.component_lists = {}
        model.component_data_lists = {}
        model.component_lists[Set] = [model.q, model.Q, model.qq]
        model.component_data_lists[Set] = [model.q, model.Q[1], model.Q[2], model.qq]
        model.component_lists[Var] = [model.x, model.X]
        model.component_data_lists[Var] = [model.x, model.X[1], model.X[2]]
        model.component_lists[Expression] = [model.e, model.E]
        model.component_data_lists[Expression] = [model.e, model.E[1], model.E[2]]
        model.component_lists[Param] = [model.p, model.P]
        model.component_data_lists[Param] = [model.p, model.P[1], model.P[2]]
        model.component_lists[Objective] = [model.o, model.O]
        model.component_data_lists[Objective] = [model.o, model.O[1], model.O[2]]
        model.component_lists[Constraint] = [model.c, model.C]
        model.component_data_lists[Constraint] = [model.c, model.C[1], model.C[2]]
        model.component_lists[SOSConstraint] = [model.sos, model.SOS]
        model.component_data_lists[SOSConstraint] = [
            model.sos,
            model.SOS[1],
            model.SOS[2],
        ]
        model.component_lists[Suffix] = [model.s]
        model.component_data_lists[Suffix] = [model.s]
        model.component_lists[Block] = [model.b, model.B]
        model.component_data_lists[Block] = [model.b, model.B[1], model.B[2]]

        return model

    def generator_runner(self, ctype):
        model = self.generate_model()

        for block in model.block_data_objects(sort=SortComponents.indices):
            # Non-nested components(active=True)
            generator = None
            try:
                generator = list(
                    block.component_objects(ctype, active=True, descend_into=False)
                )
            except:
                if issubclass(ctype, Component):
                    print("component_objects(active=True) failed with ctype %s" % ctype)
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail(
                        "component_objects(active=True) should have failed with ctype %s"
                        % ctype
                    )
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    [comp.name for comp in generator],
                    [comp.name for comp in block.component_lists[ctype]],
                )
                self.assertEqual(
                    [id(comp) for comp in generator],
                    [id(comp) for comp in block.component_lists[ctype]],
                )

            # Non-nested components
            generator = None
            try:
                generator = list(block.component_objects(ctype, descend_into=False))
            except:
                if issubclass(ctype, Component):
                    print("components failed with ctype %s" % ctype)
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail("components should have failed with ctype %s" % ctype)
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    [comp.name for comp in generator],
                    [comp.name for comp in block.component_lists[ctype]],
                )
                self.assertEqual(
                    [id(comp) for comp in generator],
                    [id(comp) for comp in block.component_lists[ctype]],
                )

            # Non-nested component_data_objects, active=True, sort_by_keys=False
            generator = None
            try:
                generator = list(
                    block.component_data_iterindex(
                        ctype, active=True, sort=False, descend_into=False
                    )
                )
            except:
                if issubclass(ctype, Component):
                    print(
                        "component_data_objects(active=True, sort_by_keys=False) failed with ctype %s"
                        % ctype
                    )
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail(
                        "component_data_objects(active=True, sort_by_keys=False) should have failed with ctype %s"
                        % ctype
                    )
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    [comp.name for name, comp in generator],
                    [comp.name for comp in block.component_data_lists[ctype]],
                )
                self.assertEqual(
                    [id(comp) for name, comp in generator],
                    [id(comp) for comp in block.component_data_lists[ctype]],
                )

            # Non-nested component_data_objects, active=True, sort=True
            generator = None
            try:
                generator = list(
                    block.component_data_iterindex(
                        ctype, active=True, sort=True, descend_into=False
                    )
                )
            except:
                if issubclass(ctype, Component):
                    print(
                        "component_data_objects(active=True, sort=True) failed with ctype %s"
                        % ctype
                    )
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail(
                        "component_data_objects(active=True, sort=True) should have failed with ctype %s"
                        % ctype
                    )
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    sorted([comp.name for name, comp in generator]),
                    sorted([comp.name for comp in block.component_data_lists[ctype]]),
                )
                self.assertEqual(
                    sorted([id(comp) for name, comp in generator]),
                    sorted([id(comp) for comp in block.component_data_lists[ctype]]),
                )

            # Non-nested components_data, sort_by_keys=True
            generator = None
            try:
                generator = list(
                    block.component_data_iterindex(
                        ctype, sort=False, descend_into=False
                    )
                )
            except:
                if issubclass(ctype, Component):
                    print(
                        "components_data(sort_by_keys=True) failed with ctype %s"
                        % ctype
                    )
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail(
                        "components_data(sort_by_keys=True) should have failed with ctype %s"
                        % ctype
                    )
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    [comp.name for name, comp in generator],
                    [comp.name for comp in block.component_data_lists[ctype]],
                )
                self.assertEqual(
                    [id(comp) for name, comp in generator],
                    [id(comp) for comp in block.component_data_lists[ctype]],
                )

            # Non-nested components_data, sort_by_keys=False
            generator = None
            try:
                generator = list(
                    block.component_data_iterindex(ctype, sort=True, descend_into=False)
                )
            except:
                if issubclass(ctype, Component):
                    print(
                        "components_data(sort_by_keys=False) failed with ctype %s"
                        % ctype
                    )
                    raise
            else:
                if not issubclass(ctype, Component):
                    self.fail(
                        "components_data(sort_by_keys=False) should have failed with ctype %s"
                        % ctype
                    )
                # This first check is less safe but it gives a cleaner
                # failure message. I leave comparison of ids in the
                # second assertEqual to make sure the tests are working
                # as expected
                self.assertEqual(
                    sorted([comp.name for name, comp in generator]),
                    sorted([comp.name for comp in block.component_data_lists[ctype]]),
                )
                self.assertEqual(
                    sorted([id(comp) for name, comp in generator]),
                    sorted([id(comp) for comp in block.component_data_lists[ctype]]),
                )

    def test_Objective(self):
        self.generator_runner(Objective)

    def test_Expression(self):
        self.generator_runner(Expression)

    def test_Suffix(self):
        self.generator_runner(Suffix)

    def test_Constraint(self):
        self.generator_runner(Constraint)

    def test_Param(self):
        self.generator_runner(Param)

    def test_Var(self):
        self.generator_runner(Var)

    def test_Set(self):
        self.generator_runner(Set)

    def test_SOSConstraint(self):
        self.generator_runner(SOSConstraint)

    def test_Block(self):
        self.generator_runner(Block)

        model = self.generate_model()

        # sorted all_blocks
        self.assertEqual(
            [
                id(comp)
                for comp in model.block_data_objects(sort=SortComponents.deterministic)
            ],
            [id(comp) for comp in [model] + model.component_data_lists[Block]],
        )

        # unsorted all_blocks
        self.assertEqual(
            sorted([id(comp) for comp in model.block_data_objects(sort=False)]),
            sorted([id(comp) for comp in [model] + model.component_data_lists[Block]]),
        )

    def test_mixed_index_type(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, '1', 3.5, 4])
        m.x = Var(m.I)
        v = list(m.component_data_objects(Var, sort=True))
        self.assertEqual(len(v), 4)
        for a, b in zip([m.x[1], m.x[3.5], m.x[4], m.x['1']], v):
            self.assertIs(a, b)


class HierarchicalModel(object):
    def __init__(self):
        m = self.model = ConcreteModel()
        m.a1_IDX = Set(initialize=[5, 4], ordered=True)
        m.a3_IDX = Set(initialize=[6, 7], ordered=True)

        m.c = Block()

        def x(b, i):
            pass

        def a(b, i):
            if i == 1:
                b.d = Block()
                b.c = Block(b.model().a1_IDX, rule=x)
            elif i == 3:
                b.e = Block()
                b.f = Block(b.model().a3_IDX, rule=x)

        m.a = Block([1, 2, 3], rule=a)
        m.b = Block()

        self.PrefixDFS = [
            'unknown',
            'c',
            'a[1]',
            'a[1].d',
            'a[1].c[5]',
            'a[1].c[4]',
            'a[2]',
            'a[3]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'b',
        ]

        self.PrefixDFS_sortIdx = [
            'unknown',
            'c',
            'a[1]',
            'a[1].d',
            'a[1].c[4]',
            'a[1].c[5]',
            'a[2]',
            'a[3]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'b',
        ]
        self.PrefixDFS_sortName = [
            'unknown',
            'a[1]',
            'a[1].c[5]',
            'a[1].c[4]',
            'a[1].d',
            'a[2]',
            'a[3]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'b',
            'c',
        ]
        self.PrefixDFS_sort = [
            'unknown',
            'a[1]',
            'a[1].c[4]',
            'a[1].c[5]',
            'a[1].d',
            'a[2]',
            'a[3]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'b',
            'c',
        ]

        self.PostfixDFS = [
            'c',
            'a[1].d',
            'a[1].c[5]',
            'a[1].c[4]',
            'a[1]',
            'a[2]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'a[3]',
            'b',
            'unknown',
        ]

        self.PostfixDFS_sortIdx = [
            'c',
            'a[1].d',
            'a[1].c[4]',
            'a[1].c[5]',
            'a[1]',
            'a[2]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'a[3]',
            'b',
            'unknown',
        ]
        self.PostfixDFS_sortName = [
            'a[1].c[5]',
            'a[1].c[4]',
            'a[1].d',
            'a[1]',
            'a[2]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'a[3]',
            'b',
            'c',
            'unknown',
        ]
        self.PostfixDFS_sort = [
            'a[1].c[4]',
            'a[1].c[5]',
            'a[1].d',
            'a[1]',
            'a[2]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
            'a[3]',
            'b',
            'c',
            'unknown',
        ]

        self.BFS = [
            'unknown',
            'c',
            'a[1]',
            'a[2]',
            'a[3]',
            'b',
            'a[1].d',
            'a[1].c[5]',
            'a[1].c[4]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
        ]

        self.BFS_sortIdx = [
            'unknown',
            'c',
            'a[1]',
            'a[2]',
            'a[3]',
            'b',
            'a[1].d',
            'a[1].c[4]',
            'a[1].c[5]',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
        ]
        self.BFS_sortName = [
            'unknown',
            'a[1]',
            'a[2]',
            'a[3]',
            'b',
            'c',
            'a[1].c[5]',
            'a[1].c[4]',
            'a[1].d',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
        ]
        self.BFS_sort = [
            'unknown',
            'a[1]',
            'a[2]',
            'a[3]',
            'b',
            'c',
            'a[1].c[4]',
            'a[1].c[5]',
            'a[1].d',
            'a[3].e',
            'a[3].f[6]',
            'a[3].f[7]',
        ]


class MixedHierarchicalModel(object):
    def __init__(self):
        m = self.model = ConcreteModel()
        m.a = Block()
        m.a.c = DerivedBlock()
        m.b = DerivedBlock()
        m.b.d = DerivedBlock()
        m.b.e = Block()
        m.b.e.f = DerivedBlock()
        m.b.e.f.g = Block()

        self.PrefixDFS_block = ['unknown', 'a']
        self.PostfixDFS_block = ['a', 'unknown']
        self.BFS_block = ['unknown', 'a']

        self.PrefixDFS_both = [
            'unknown',
            'a',
            'a.c',
            'b',
            'b.d',
            'b.e',
            'b.e.f',
            'b.e.f.g',
        ]
        self.PostfixDFS_both = [
            'a.c',
            'a',
            'b.d',
            'b.e.f.g',
            'b.e.f',
            'b.e',
            'b',
            'unknown',
        ]
        self.BFS_both = ['unknown', 'a', 'b', 'a.c', 'b.d', 'b.e', 'b.e.f', 'b.e.f.g']

        #
        # References for component_objects tests (note: the model
        # doesn't appear)
        #
        self.PrefixDFS_block_subclass = ['a', 'b.e', 'b.e.f.g']
        self.PostfixDFS_block_subclass = ['b.e.f.g', 'b.e', 'a']
        self.BFS_block_subclass = ['a', 'b.e', 'b.e.f.g']


class TestBlock(unittest.TestCase):
    def setUp(self):
        #
        # Create block
        #
        self.block = Block()
        self.block.construct()

    def tearDown(self):
        self.block = None
        if os.path.exists("unknown.lp"):
            os.unlink("unknown.lp")
        TempfileManager.clear_tempfiles()

    def test_collect_ctypes(self):
        b = Block(concrete=True)
        self.assertEqual(b.collect_ctypes(), set())
        self.assertEqual(b.collect_ctypes(active=True), set())
        b.x = Var()
        self.assertEqual(b.collect_ctypes(), set([Var]))
        self.assertEqual(b.collect_ctypes(active=True), set([Var]))
        b.y = Constraint(expr=b.x >= 1)
        self.assertEqual(b.collect_ctypes(), set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True), set([Var, Constraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(), set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True), set([Var]))
        B = Block()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False), set([Block]))
        self.assertEqual(
            B.collect_ctypes(descend_into=False, active=True), set([Block])
        )
        self.assertEqual(B.collect_ctypes(), set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True), set([Block, Var]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False), set([Block]))
        self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([]))
        self.assertEqual(B.collect_ctypes(), set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True), set([]))
        del b.y

        # a block DOES check its own .active flag
        self.assertEqual(b.collect_ctypes(), set([Var]))
        self.assertEqual(b.collect_ctypes(active=True), set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(), set([Var]))
        self.assertEqual(b.collect_ctypes(active=True), set([Var]))

        del b.x
        self.assertEqual(b.collect_ctypes(), set())
        b.x = Var()
        self.assertEqual(b.collect_ctypes(), set([Var]))
        b.reclassify_component_type(b.x, Constraint)
        self.assertEqual(b.collect_ctypes(), set([Constraint]))
        del b.x
        self.assertEqual(b.collect_ctypes(), set())

    def test_clear_attribute(self):
        """Coverage of the _clear_attribute method"""
        obj = Set()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

        obj = Var()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

        obj = Param()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

        obj = Objective()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

        obj = Constraint()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

        obj = Set()
        self.block.A = obj
        self.assertEqual(self.block.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, self.block.A)

    def test_set_attr(self):
        p = Param(mutable=True)
        self.block.x = p
        self.block.x = 5
        self.assertEqual(value(self.block.x), 5)
        self.assertEqual(value(p), 5)
        self.block.x = 6
        self.assertEqual(value(self.block.x), 6)
        self.assertEqual(value(p), 6)
        self.block.x = None
        self.assertEqual(self.block.x._value, None)

        ### creation of a circular reference
        b = Block(concrete=True)
        b.c = Block()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot assign the top-level block as a subblock "
            r"of one of its children \(c\): creates a circular hierarchy",
        ):
            b.c.d = b

    def test_set_value(self):
        b = Block(concrete=True)
        with self.assertRaisesRegex(
            RuntimeError, "Block components do not support assignment or set_value"
        ):
            b.set_value(None)

        b.b = Block()
        with self.assertRaisesRegex(
            RuntimeError, "Block components do not support assignment or set_value"
        ):
            b.b = 5

    def test_clear(self):
        class DerivedBlock(ScalarBlock):
            _Block_reserved_words = None

        DerivedBlock._Block_reserved_words = (
            set(['a', 'b', 'c']) | BlockData._Block_reserved_words
        )

        m = ConcreteModel()
        m.clear()
        self.assertEqual(m._ctypes, {})
        self.assertEqual(m._decl, {})
        self.assertEqual(m._decl_order, [])

        m.w = 5
        m.x = Var()
        m.y = Param()
        m.z = Var()
        m.clear()
        self.assertFalse(hasattr(m, 'w'))
        self.assertEqual(m._ctypes, {})
        self.assertEqual(m._decl, {})
        self.assertEqual(m._decl_order, [])

        m.b = DerivedBlock()
        with m.b._declare_reserved_components():
            m.b.a = a = Param()
            m.b.x = Var()
            m.b.b = b = Var()
            m.b.y = Var()
            m.b.z = Param()
            m.b.c = c = Param()
        m.b.clear()
        self.assertEqual(m.b._ctypes, {Var: [1, 1, 1], Param: [0, 2, 2]})
        self.assertEqual(m.b._decl, {'a': 0, 'b': 1, 'c': 2})
        self.assertEqual(len(m.b._decl_order), 3)
        self.assertIs(m.b._decl_order[0][0], a)
        self.assertIs(m.b._decl_order[1][0], b)
        self.assertIs(m.b._decl_order[2][0], c)
        self.assertEqual(m.b._decl_order[0][1], 2)
        self.assertEqual(m.b._decl_order[1][1], None)
        self.assertEqual(m.b._decl_order[2][1], None)

    def test_transfer_attributes_from(self):
        b = Block(concrete=True)
        b.x = Var()
        b.y = Var()
        c = Block(concrete=True)
        c.z = Param(initialize=5)
        c.x = c_x = Param(initialize=5)
        c.y = c_y = 5

        b.clear()
        b.transfer_attributes_from(c)
        self.assertEqual(list(b.component_map()), ['z', 'x'])
        self.assertEqual(list(c.component_map()), [])
        self.assertIs(b.x, c_x)
        self.assertIs(b.y, c_y)

        class DerivedBlock(ScalarBlock):
            _Block_reserved_words = set()

            def __init__(self, *args, **kwds):
                super(DerivedBlock, self).__init__(*args, **kwds)
                with self._declare_reserved_components():
                    self.x = Var()
                    self.y = Var()

        DerivedBlock._Block_reserved_words = set(dir(DerivedBlock()))

        b = DerivedBlock(concrete=True)
        b_x = b.x
        b_y = b.y
        c = Block(concrete=True)
        c.z = Param(initialize=5)
        c.x = c_x = Param(initialize=5)
        c.y = c_y = 5

        b.clear()
        b.transfer_attributes_from(c)
        self.assertEqual(list(b.component_map()), ['y', 'z', 'x'])
        self.assertEqual(list(c.component_map()), [])
        self.assertIs(b.x, c_x)
        self.assertIsNot(b.y, c_y)
        self.assertIs(b.y, b_y)
        self.assertEqual(value(b.y), value(c_y))

        ### assignment of dict
        b = DerivedBlock(concrete=True)
        b_x = b.x
        b_y = b.y
        c = {'z': Param(initialize=5), 'x': Param(initialize=5), 'y': 5}

        b.clear()
        b.transfer_attributes_from(c)
        self.assertEqual(list(b.component_map()), ['y', 'z', 'x'])
        self.assertEqual(sorted(list(c.keys())), ['x', 'y', 'z'])
        self.assertIs(b.x, c['x'])
        self.assertIsNot(b.y, c['y'])
        self.assertIs(b.y, b_y)
        self.assertEqual(value(b.y), value(c_y))

        ### assignment of self
        b = Block(concrete=True)
        b.x = b_x = Var()
        b.y = b_y = Var()
        b.transfer_attributes_from(b)

        self.assertEqual(list(b.component_map()), ['x', 'y'])
        self.assertIs(b.x, b_x)
        self.assertIs(b.y, b_y)

        ### creation of a circular reference
        b = Block(concrete=True)
        b.c = Block()
        b.c.d = Block()
        b.c.d.e = Block()
        with self.assertRaisesRegex(
            ValueError,
            r'BlockData.transfer_attributes_from\(\): '
            r'Cannot set a sub-block \(c.d.e\) to a parent block \(c\):',
        ):
            b.c.d.e.transfer_attributes_from(b.c)

        ### bad data type
        b = Block(concrete=True)
        with self.assertRaisesRegex(
            ValueError,
            r'BlockData.transfer_attributes_from\(\): expected a Block '
            'or dict; received str',
        ):
            b.transfer_attributes_from('foo')

    def test_iterate_hierarchy_defaults(self):
        self.assertIs(TraversalStrategy.BFS, TraversalStrategy.BreadthFirstSearch)

        self.assertIs(TraversalStrategy.DFS, TraversalStrategy.PrefixDepthFirstSearch)
        self.assertIs(TraversalStrategy.DFS, TraversalStrategy.PrefixDFS)
        self.assertIs(
            TraversalStrategy.DFS, TraversalStrategy.ParentFirstDepthFirstSearch
        )

        self.assertIs(
            TraversalStrategy.PostfixDepthFirstSearch, TraversalStrategy.PostfixDFS
        )
        self.assertIs(
            TraversalStrategy.PostfixDepthFirstSearch,
            TraversalStrategy.ParentLastDepthFirstSearch,
        )

        HM = HierarchicalModel()
        m = HM.model
        result = [x.name for x in m.block_data_objects()]
        self.assertEqual(HM.PrefixDFS, result)

    def test_iterate_hierarchy_PrefixDFS(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch
            )
        ]
        self.assertEqual(HM.PrefixDFS, result)

    def test_iterate_hierarchy_PrefixDFS_sortIndex(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                sort=SortComponents.indices,
            )
        ]
        self.assertEqual(HM.PrefixDFS_sortIdx, result)

    def test_iterate_hierarchy_PrefixDFS_sortName(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                sort=SortComponents.alphaOrder,
            )
        ]
        self.assertEqual(HM.PrefixDFS_sortName, result)

    def test_iterate_hierarchy_PrefixDFS_sort(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch, sort=True
            )
        ]
        self.assertEqual(HM.PrefixDFS_sort, result)

    def test_iterate_hierarchy_PostfixDFS(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch
            )
        ]
        self.assertEqual(HM.PostfixDFS, result)

    def test_iterate_hierarchy_PostfixDFS_sortIndex(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                sort=SortComponents.indices,
            )
        ]
        self.assertEqual(HM.PostfixDFS_sortIdx, result)

    def test_iterate_hierarchy_PostfixDFS_sortName(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                sort=SortComponents.alphaOrder,
            )
        ]
        self.assertEqual(HM.PostfixDFS_sortName, result)

    def test_iterate_hierarchy_PostfixDFS_sort(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch, sort=True
            )
        ]
        self.assertEqual(HM.PostfixDFS_sort, result)

    def test_iterate_hierarchy_BFS(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BreadthFirstSearch
            )
        ]
        self.assertEqual(HM.BFS, result)

    def test_iterate_hierarchy_BFS_sortIndex(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BreadthFirstSearch,
                sort=SortComponents.indices,
            )
        ]
        self.assertEqual(HM.BFS_sortIdx, result)

    def test_iterate_hierarchy_BFS_sortName(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BreadthFirstSearch,
                sort=SortComponents.alphaOrder,
            )
        ]
        self.assertEqual(HM.BFS_sortName, result)

    def test_iterate_hierarchy_BFS_sort(self):
        HM = HierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BreadthFirstSearch, sort=True
            )
        ]
        self.assertEqual(HM.BFS_sort, result)

    def test_iterate_mixed_hierarchy_PrefixDFS_block(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                descend_into=Block,
            )
        ]
        self.assertEqual(HM.PrefixDFS_block, result)

    def test_iterate_mixed_hierarchy_PrefixDFS_both(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                descend_into=(Block, DerivedBlock),
            )
        ]
        self.assertEqual(HM.PrefixDFS_both, result)

    def test_iterate_mixed_hierarchy_PrefixDFS_SubclassOf(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                descend_into=SubclassOf(Block),
            )
        ]
        self.assertEqual(HM.PrefixDFS_both, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                descend_into=SubclassOf(Block),
            )
        ]
        self.assertEqual(HM.PrefixDFS_block_subclass, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.PrefixDepthFirstSearch,
                descend_into=SubclassOf(Var, Block),
            )
        ]
        self.assertEqual(HM.PrefixDFS_block_subclass, result)

    def test_iterate_mixed_hierarchy_PostfixDFS_block(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                descend_into=Block,
            )
        ]
        self.assertEqual(HM.PostfixDFS_block, result)

    def test_iterate_mixed_hierarchy_PostfixDFS_both(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                descend_into=(Block, DerivedBlock),
            )
        ]
        self.assertEqual(HM.PostfixDFS_both, result)

    def test_iterate_mixed_hierarchy_PostfixDFS_SubclassOf(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                descend_into=SubclassOf(Block),
            )
        ]
        self.assertEqual(HM.PostfixDFS_both, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                descend_into=SubclassOf(Block),
            )
        ]
        self.assertEqual(HM.PostfixDFS_block_subclass, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.PostfixDepthFirstSearch,
                descend_into=SubclassOf(Var, Block),
            )
        ]
        self.assertEqual(HM.PostfixDFS_block_subclass, result)

    def test_iterate_mixed_hierarchy_BFS_block(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BFS, descend_into=Block
            )
        ]
        self.assertEqual(HM.BFS_block, result)

    def test_iterate_mixed_hierarchy_BFS_both(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BFS, descend_into=(Block, DerivedBlock)
            )
        ]
        self.assertEqual(HM.BFS_both, result)

    def test_iterate_mixed_hierarchy_BFS_SubclassOf(self):
        HM = MixedHierarchicalModel()
        m = HM.model
        result = [
            x.name
            for x in m.block_data_objects(
                descent_order=TraversalStrategy.BFS, descend_into=SubclassOf(Block)
            )
        ]
        self.assertEqual(HM.BFS_both, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.BFS,
                descend_into=SubclassOf(Block),
            )
        ]
        self.assertEqual(HM.BFS_block_subclass, result)
        result = [
            x.name
            for x in m.component_objects(
                ctype=Block,
                descent_order=TraversalStrategy.BFS,
                descend_into=SubclassOf(Var, Block),
            )
        ]
        self.assertEqual(HM.BFS_block_subclass, result)

    def test_add_remove_component_byname(self):
        m = Block()
        self.assertFalse(m.contains_component(Var))
        self.assertFalse(m.component_map(Var))

        m.x = x = Var()
        self.assertTrue(m.contains_component(Var))
        self.assertTrue(m.component_map(Var))
        self.assertTrue('x' in m.__dict__)
        self.assertIs(m.component('x'), x)

        m.del_component('x')
        self.assertFalse(m.contains_component(Var))
        self.assertFalse(m.component_map(Var))
        self.assertFalse('x' in m.__dict__)
        self.assertIs(m.component('x'), None)

    def test_add_remove_component_byref(self):
        m = Block()
        self.assertFalse(m.contains_component(Var))
        self.assertFalse(m.component_map(Var))

        m.x = x = Var()
        self.assertTrue(m.contains_component(Var))
        self.assertTrue(m.component_map(Var))
        self.assertTrue('x' in m.__dict__)
        self.assertIs(m.component('x'), x)

        m.del_component(m.x)
        self.assertFalse(m.contains_component(Var))
        self.assertFalse(m.component_map(Var))
        self.assertFalse('x' in m.__dict__)
        self.assertIs(m.component('x'), None)

    def test_add_del_component(self):
        m = Block()
        self.assertFalse(m.contains_component(Var))
        m.x = x = Var()
        self.assertTrue(m.contains_component(Var))
        self.assertTrue('x' in m.__dict__)
        self.assertIs(m.component('x'), x)
        del m.x
        self.assertFalse(m.contains_component(Var))
        self.assertFalse('x' in m.__dict__)
        self.assertIs(m.component('x'), None)

    def test_del_component_data(self):
        m = ConcreteModel()
        self.assertFalse(m.contains_component(Var))
        x = m.x = Var([1, 2, 3])
        self.assertTrue(m.contains_component(Var))
        self.assertIs(m.component('x'), x)
        del m.x[1]
        self.assertTrue(m.contains_component(Var))
        self.assertTrue('x' in m.__dict__)
        self.assertEqual(len(m.x), 2)
        self.assertIn(m.x[2], ComponentSet(m.x.values()))
        self.assertIn(m.x[3], ComponentSet(m.x.values()))

        # This fails:
        with self.assertRaisesRegex(
            ValueError,
            r"Argument 'x\[2\]' to del_component is a ComponentData object. "
            r"Please use the Python 'del' function to delete members of "
            r"indexed Pyomo components. The del_component function can "
            r"only be used to delete IndexedComponents and "
            r"ScalarComponents.",
        ):
            m.del_component(m.x[2])

        # But we can use del
        del m.x[2]
        self.assertTrue(m.contains_component(Var))
        self.assertTrue('x' in m.__dict__)
        self.assertEqual(len(m.x), 1)
        self.assertIn(m.x[3], ComponentSet(m.x.values()))

    def test_reclassify_component(self):
        m = Block()
        m.a = Var()
        m.b = Var()
        m.c = Param()

        self.assertEqual(len(m.component_map(Var)), 2)
        self.assertEqual(len(m.component_map(Param)), 1)
        self.assertEqual(['a', 'b'], list(m.component_map(Var)))
        self.assertEqual(['c'], list(m.component_map(Param)))

        # Test removing from the end of a list and appending to the beginning
        # of a list
        m.reclassify_component_type(m.b, Param)
        self.assertEqual(len(m.component_map(Var)), 1)
        self.assertEqual(len(m.component_map(Param)), 2)
        self.assertEqual(['a'], list(m.component_map(Var)))
        self.assertEqual(['b', 'c'], list(m.component_map(Param)))

        # Test removing from the beginning of a list and appending to
        # the end of a list
        m.reclassify_component_type(m.b, Var)
        self.assertEqual(len(m.component_map(Var)), 2)
        self.assertEqual(len(m.component_map(Param)), 1)
        self.assertEqual(['a', 'b'], list(m.component_map(Var)))
        self.assertEqual(['c'], list(m.component_map(Param)))

        # Test removing the last element of a list and creating a new list
        m.reclassify_component_type(m.c, Var)
        self.assertEqual(len(m.component_map(Var)), 3)
        self.assertEqual(len(m.component_map(Param)), 0)
        self.assertTrue(m.contains_component(Var))
        self.assertFalse(m.contains_component(Param))
        self.assertFalse(m.contains_component(Constraint))
        self.assertEqual(['a', 'b', 'c'], list(m.component_map(Var)))
        self.assertEqual([], list(m.component_map(Param)))

        # Test removing the last element of a list and creating a new list
        m.reclassify_component_type(m.c, Param)
        self.assertEqual(len(m.component_map(Var)), 2)
        self.assertEqual(len(m.component_map(Param)), 1)
        self.assertEqual(len(m.component_map(Constraint)), 0)
        self.assertTrue(m.contains_component(Var))
        self.assertTrue(m.contains_component(Param))
        self.assertFalse(m.contains_component(Constraint))
        self.assertEqual(['a', 'b'], list(m.component_map(Var)))
        self.assertEqual(['c'], list(m.component_map(Param)))

        # Test removing the first element of a list and creating a new list
        m.reclassify_component_type(m.a, Constraint)
        self.assertEqual(len(m.component_map(Var)), 1)
        self.assertEqual(len(m.component_map(Param)), 1)
        self.assertEqual(len(m.component_map(Constraint)), 1)
        self.assertTrue(m.contains_component(Var))
        self.assertTrue(m.contains_component(Param))
        self.assertTrue(m.contains_component(Constraint))
        self.assertEqual(['b'], list(m.component_map(Var)))
        self.assertEqual(['c'], list(m.component_map(Param)))
        self.assertEqual(['a'], list(m.component_map(Constraint)))

        # Test removing the last element of a list and inserting it into
        # the middle of new list
        m.reclassify_component_type(m.a, Param)
        m.reclassify_component_type(m.b, Param)
        self.assertEqual(len(m.component_map(Var)), 0)
        self.assertEqual(len(m.component_map(Param)), 3)
        self.assertEqual(len(m.component_map(Constraint)), 0)
        self.assertFalse(m.contains_component(Var))
        self.assertTrue(m.contains_component(Param))
        self.assertFalse(m.contains_component(Constraint))
        self.assertEqual([], list(m.component_map(Var)))
        self.assertEqual(['a', 'b', 'c'], list(m.component_map(Param)))
        self.assertEqual([], list(m.component_map(Constraint)))

        # Test idnoring decl order
        m.reclassify_component_type('b', Var, preserve_declaration_order=False)
        m.reclassify_component_type('c', Var, preserve_declaration_order=False)
        m.reclassify_component_type('a', Var, preserve_declaration_order=False)
        self.assertEqual(len(m.component_map(Var)), 3)
        self.assertEqual(len(m.component_map(Param)), 0)
        self.assertEqual(len(m.component_map(Constraint)), 0)
        self.assertTrue(m.contains_component(Var))
        self.assertFalse(m.contains_component(Param))
        self.assertFalse(m.contains_component(Constraint))
        self.assertEqual(['b', 'c', 'a'], list(m.component_map(Var)))
        self.assertEqual([], list(m.component_map(Param)))
        self.assertEqual([], list(m.component_map(Constraint)))

    def test_replace_attribute_with_component(self):
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            self.block.x = 5
            self.block.x = Var()
        self.assertIn('Reassigning the non-component attribute', OUTPUT.getvalue())

    def test_replace_component_with_component(self):
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            self.block.x = Var()
            self.block.x = Var()
        self.assertIn('Implicitly replacing the Component attribute', OUTPUT.getvalue())

    def test_pseudomap_len(self):
        m = Block()
        m.a = Constraint()
        m.b = Constraint()  # active=False
        m.c = Constraint()
        m.z = Objective()  # active=False
        m.x = Objective()
        m.v = Objective()
        m.y = Objective()
        m.w = Objective()  # active=False

        m.b.deactivate()
        m.z.deactivate()
        m.w.deactivate()

        self.assertEqual(len(m.component_map()), 8)
        self.assertEqual(len(m.component_map(active=True)), 5)
        self.assertEqual(len(m.component_map(active=False)), 3)

        self.assertEqual(len(m.component_map(Constraint)), 3)
        self.assertEqual(len(m.component_map(Constraint, active=True)), 2)
        self.assertEqual(len(m.component_map(Constraint, active=False)), 1)

        self.assertEqual(len(m.component_map(Objective)), 5)
        self.assertEqual(len(m.component_map(Objective, active=True)), 3)
        self.assertEqual(len(m.component_map(Objective, active=False)), 2)

    def test_pseudomap_contains(self):
        m = Block()
        m.a = Constraint()
        m.b = Constraint()  # active=False
        m.c = Constraint()
        m.s = Set()
        m.t = Suffix()
        m.z = Objective()  # active=False
        m.x = Objective()
        m.b.deactivate()
        m.z.deactivate()

        pm = m.component_map()
        self.assertTrue('a' in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' in pm)
        self.assertTrue('t' in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' in pm)

        pm = m.component_map(active=True)
        self.assertTrue('a' in pm)
        self.assertTrue('b' not in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' in pm)
        self.assertTrue('t' in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map(active=False)
        self.assertTrue('a' not in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' not in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' in pm)

        pm = m.component_map(Constraint)
        self.assertTrue('a' in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map(Constraint, active=True)
        self.assertTrue('a' in pm)
        self.assertTrue('b' not in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map(Constraint, active=False)
        self.assertTrue('a' not in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' not in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map([Constraint, Objective])
        self.assertTrue('a' in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' in pm)

        pm = m.component_map([Constraint, Objective], active=True)
        self.assertTrue('a' in pm)
        self.assertTrue('b' not in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map([Constraint, Objective], active=False)
        self.assertTrue('a' not in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' not in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' in pm)

        # You should be able to pass in a set as well as a list
        pm = m.component_map(set([Constraint, Objective]))
        self.assertTrue('a' in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' in pm)

        pm = m.component_map(set([Constraint, Objective]), active=True)
        self.assertTrue('a' in pm)
        self.assertTrue('b' not in pm)
        self.assertTrue('c' in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' in pm)
        self.assertTrue('z' not in pm)

        pm = m.component_map(set([Constraint, Objective]), active=False)
        self.assertTrue('a' not in pm)
        self.assertTrue('b' in pm)
        self.assertTrue('c' not in pm)
        self.assertTrue('d' not in pm)
        self.assertTrue('s' not in pm)
        self.assertTrue('t' not in pm)
        self.assertTrue('x' not in pm)
        self.assertTrue('z' in pm)

    def test_pseudomap_getitem(self):
        m = Block()
        m.a = a = Constraint()
        m.b = b = Constraint()  # active=False
        m.c = c = Constraint()
        m.s = s = Set()
        m.t = t = Suffix()
        m.z = z = Objective()  # active=False
        m.x = x = Objective()
        m.b.deactivate()
        m.z.deactivate()

        def assertWorks(self, key, pm):
            self.assertIs(pm[key.local_name], key)

        def assertFails(self, key, pm):
            if not isinstance(key, str):
                key = key.local_name
            self.assertRaises(KeyError, pm.__getitem__, key)

        pm = m.component_map()
        assertWorks(self, a, pm)
        assertWorks(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertWorks(self, s, pm)
        assertWorks(self, t, pm)
        assertWorks(self, x, pm)
        assertWorks(self, z, pm)

        pm = m.component_map(active=True)
        assertWorks(self, a, pm)
        assertFails(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertWorks(self, s, pm)
        assertWorks(self, t, pm)
        assertWorks(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map(active=False)
        assertFails(self, a, pm)
        assertWorks(self, b, pm)
        assertFails(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertWorks(self, z, pm)

        pm = m.component_map(Constraint)
        assertWorks(self, a, pm)
        assertWorks(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map(Constraint, active=True)
        assertWorks(self, a, pm)
        assertFails(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map(Constraint, active=False)
        assertFails(self, a, pm)
        assertWorks(self, b, pm)
        assertFails(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map([Constraint, Objective])
        assertWorks(self, a, pm)
        assertWorks(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertWorks(self, x, pm)
        assertWorks(self, z, pm)

        pm = m.component_map([Constraint, Objective], active=True)
        assertWorks(self, a, pm)
        assertFails(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertWorks(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map([Constraint, Objective], active=False)
        assertFails(self, a, pm)
        assertWorks(self, b, pm)
        assertFails(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertWorks(self, z, pm)

        pm = m.component_map(set([Constraint, Objective]))
        assertWorks(self, a, pm)
        assertWorks(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertWorks(self, x, pm)
        assertWorks(self, z, pm)

        pm = m.component_map(set([Constraint, Objective]), active=True)
        assertWorks(self, a, pm)
        assertFails(self, b, pm)
        assertWorks(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertWorks(self, x, pm)
        assertFails(self, z, pm)

        pm = m.component_map(set([Constraint, Objective]), active=False)
        assertFails(self, a, pm)
        assertWorks(self, b, pm)
        assertFails(self, c, pm)
        assertFails(self, 'd', pm)
        assertFails(self, s, pm)
        assertFails(self, t, pm)
        assertFails(self, x, pm)
        assertWorks(self, z, pm)

    def test_pseudomap_getitem_exceptionString(self):
        def tester(pm, _str):
            try:
                pm['a']
                self.fail("Expected PseudoMap to raise a KeyError")
            except KeyError:
                err = sys.exc_info()[1].args[0]
                self.assertEqual(_str, err)

        m = Block(name='foo')
        tester(m.component_map(), "component 'a' not found in block foo")
        tester(
            m.component_map(active=True), "active component 'a' not found in block foo"
        )
        tester(
            m.component_map(active=False),
            "inactive component 'a' not found in block foo",
        )

        tester(m.component_map(Var), "Var component 'a' not found in block foo")
        tester(
            m.component_map(Var, active=True),
            "active Var component 'a' not found in block foo",
        )
        tester(
            m.component_map(Var, active=False),
            "inactive Var component 'a' not found in block foo",
        )

        tester(
            m.component_map(SubclassOf(Var)),
            "SubclassOf(Var) component 'a' not found in block foo",
        )
        tester(
            m.component_map(SubclassOf(Var), active=True),
            "active SubclassOf(Var) component 'a' not found in block foo",
        )
        tester(
            m.component_map(SubclassOf(Var), active=False),
            "inactive SubclassOf(Var) component 'a' not found in block foo",
        )

        tester(
            m.component_map(SubclassOf(Var, Block)),
            "SubclassOf(Var,Block) component 'a' not found in block foo",
        )
        tester(
            m.component_map(SubclassOf(Var, Block), active=True),
            "active SubclassOf(Var,Block) component 'a' not found in block foo",
        )
        tester(
            m.component_map(SubclassOf(Var, Block), active=False),
            "inactive SubclassOf(Var,Block) component 'a' not found in block foo",
        )

        tester(
            m.component_map([Var, Param]),
            "Param or Var component 'a' not found in block foo",
        )
        tester(
            m.component_map(set([Var, Param]), active=True),
            "active Param or Var component 'a' not found in block foo",
        )
        tester(
            m.component_map(set([Var, Param]), active=False),
            "inactive Param or Var component 'a' not found in block foo",
        )

        tester(
            m.component_map(set([Set, Var, Param])),
            "Param, Set or Var component 'a' not found in block foo",
        )
        tester(
            m.component_map(set([Set, Var, Param]), active=True),
            "active Param, Set or Var component 'a' not found in block foo",
        )
        tester(
            m.component_map(set([Set, Var, Param]), active=False),
            "inactive Param, Set or Var component 'a' not found in block foo",
        )

    def test_pseudomap_iteration(self):
        m = Block()
        m.a = Constraint()
        m.z = Objective()  # active=False
        m.x = Objective()
        m.v = Objective()
        m.b = Constraint()  # active=False
        m.t = Block()  # active=False
        m.s = Block()
        m.c = Constraint()
        m.y = Objective()
        m.w = Objective()  # active=False

        m.b.deactivate()
        m.z.deactivate()
        m.w.deactivate()
        m.t.deactivate()

        self.assertEqual(
            ['a', 'z', 'x', 'v', 'b', 't', 's', 'c', 'y', 'w'], list(m.component_map())
        )

        self.assertEqual(
            ['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'],
            list(m.component_map(set([Constraint, Objective]))),
        )

        # test that the order of ctypes in the argument does not affect
        # the order in the resulting list
        self.assertEqual(
            ['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'],
            list(m.component_map([Constraint, Objective])),
        )

        self.assertEqual(
            ['a', 'z', 'x', 'v', 'b', 'c', 'y', 'w'],
            list(m.component_map([Objective, Constraint])),
        )

        self.assertEqual(['a', 'b', 'c'], list(m.component_map(Constraint)))

        self.assertEqual(
            ['z', 'x', 'v', 'y', 'w'], list(m.component_map(set([Objective])))
        )

        self.assertEqual(
            ['a', 'x', 'v', 's', 'c', 'y'], list(m.component_map(active=True))
        )

        self.assertEqual(
            ['a', 'x', 'v', 'c', 'y'],
            list(m.component_map(set([Constraint, Objective]), active=True)),
        )

        self.assertEqual(
            ['a', 'x', 'v', 'c', 'y'],
            list(m.component_map([Constraint, Objective], active=True)),
        )

        self.assertEqual(
            ['a', 'x', 'v', 'c', 'y'],
            list(m.component_map([Objective, Constraint], active=True)),
        )

        self.assertEqual(['a', 'c'], list(m.component_map(Constraint, active=True)))

        self.assertEqual(
            ['x', 'v', 'y'], list(m.component_map(set([Objective]), active=True))
        )

        self.assertEqual(['z', 'b', 't', 'w'], list(m.component_map(active=False)))

        self.assertEqual(
            ['z', 'b', 'w'],
            list(m.component_map(set([Constraint, Objective]), active=False)),
        )

        self.assertEqual(
            ['z', 'b', 'w'],
            list(m.component_map([Constraint, Objective], active=False)),
        )

        self.assertEqual(
            ['z', 'b', 'w'],
            list(m.component_map([Objective, Constraint], active=False)),
        )

        self.assertEqual(['b'], list(m.component_map(Constraint, active=False)))

        self.assertEqual(
            ['z', 'w'], list(m.component_map(set([Objective]), active=False))
        )

        self.assertEqual(
            ['a', 'b', 'c', 's', 't', 'v', 'w', 'x', 'y', 'z'],
            list(m.component_map(sort=True)),
        )

        self.assertEqual(
            ['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'],
            list(m.component_map(set([Constraint, Objective]), sort=True)),
        )

        self.assertEqual(
            ['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'],
            list(m.component_map([Constraint, Objective], sort=True)),
        )

        self.assertEqual(
            ['a', 'b', 'c', 'v', 'w', 'x', 'y', 'z'],
            list(m.component_map([Objective, Constraint], sort=True)),
        )

        self.assertEqual(['a', 'b', 'c'], list(m.component_map(Constraint, sort=True)))

        self.assertEqual(
            ['v', 'w', 'x', 'y', 'z'],
            list(m.component_map(set([Objective]), sort=True)),
        )

        self.assertEqual(
            ['a', 'c', 's', 'v', 'x', 'y'],
            list(m.component_map(active=True, sort=True)),
        )

        self.assertEqual(
            ['a', 'c', 'v', 'x', 'y'],
            list(m.component_map(set([Constraint, Objective]), active=True, sort=True)),
        )

        self.assertEqual(
            ['a', 'c', 'v', 'x', 'y'],
            list(m.component_map([Constraint, Objective], active=True, sort=True)),
        )

        self.assertEqual(
            ['a', 'c', 'v', 'x', 'y'],
            list(m.component_map([Objective, Constraint], active=True, sort=True)),
        )

        self.assertEqual(
            ['a', 'c'], list(m.component_map(Constraint, active=True, sort=True))
        )

        self.assertEqual(
            ['v', 'x', 'y'],
            list(m.component_map(set([Objective]), active=True, sort=True)),
        )

        self.assertEqual(
            ['b', 't', 'w', 'z'], list(m.component_map(active=False, sort=True))
        )

        self.assertEqual(
            ['b', 'w', 'z'],
            list(
                m.component_map(set([Constraint, Objective]), active=False, sort=True)
            ),
        )

        self.assertEqual(
            ['b', 'w', 'z'],
            list(m.component_map([Constraint, Objective], active=False, sort=True)),
        )

        self.assertEqual(
            ['b', 'w', 'z'],
            list(m.component_map([Objective, Constraint], active=False, sort=True)),
        )

        self.assertEqual(
            ['b'], list(m.component_map(Constraint, active=False, sort=True))
        )

        self.assertEqual(
            ['w', 'z'], list(m.component_map(set([Objective]), active=False, sort=True))
        )

    def test_iterate_hierarchical_blocks(self):
        def def_var(b, *args):
            b.x = Var()

        def init_block(b):
            b.c = Block([1, 2], rule=def_var)
            b.e = Disjunct([1, 2], rule=def_var)
            b.b = Block(rule=def_var)
            b.d = Disjunct(rule=def_var)

        m = ConcreteModel()
        m.x = Var()
        init_block(m)
        init_block(m.b)
        init_block(m.c[1])
        init_block(m.c[2])
        init_block(m.d)
        init_block(m.e[1])
        init_block(m.e[2])

        ref = [
            x.name
            for x in (
                m,
                m.c[1],
                m.c[1].c[1],
                m.c[1].c[2],
                m.c[1].b,
                m.c[2],
                m.c[2].c[1],
                m.c[2].c[2],
                m.c[2].b,
                m.b,
                m.b.c[1],
                m.b.c[2],
                m.b.b,
            )
        ]
        test = list(x.name for x in m.block_data_objects())
        self.assertEqual(test, ref)

        test = list(x.name for x in m.block_data_objects(descend_into=Block))
        self.assertEqual(test, ref)

        test = list(x.name for x in m.block_data_objects(descend_into=(Block,)))
        self.assertEqual(test, ref)

        ref = [
            x.name
            for x in (
                m,
                m.e[1],
                m.e[1].e[1],
                m.e[1].e[2],
                m.e[1].d,
                m.e[2],
                m.e[2].e[1],
                m.e[2].e[2],
                m.e[2].d,
                m.d,
                m.d.e[1],
                m.d.e[2],
                m.d.d,
            )
        ]
        test = list(x.name for x in m.block_data_objects(descend_into=(Disjunct,)))
        self.assertEqual(test, ref)

        ref = [x.name for x in (m.d, m.d.e[1], m.d.e[2], m.d.d)]
        test = list(x.name for x in m.d.block_data_objects(descend_into=(Disjunct,)))
        self.assertEqual(test, ref)

        ref = [
            x.name
            for x in (
                m,
                m.c[1],
                m.c[1].c[1],
                m.c[1].c[2],
                m.c[1].e[1],
                m.c[1].e[2],
                m.c[1].b,
                m.c[1].d,
                m.c[2],
                m.c[2].c[1],
                m.c[2].c[2],
                m.c[2].e[1],
                m.c[2].e[2],
                m.c[2].b,
                m.c[2].d,
                m.e[1],
                m.e[1].c[1],
                m.e[1].c[2],
                m.e[1].e[1],
                m.e[1].e[2],
                m.e[1].b,
                m.e[1].d,
                m.e[2],
                m.e[2].c[1],
                m.e[2].c[2],
                m.e[2].e[1],
                m.e[2].e[2],
                m.e[2].b,
                m.e[2].d,
                m.b,
                m.b.c[1],
                m.b.c[2],
                m.b.e[1],
                m.b.e[2],
                m.b.b,
                m.b.d,
                m.d,
                m.d.c[1],
                m.d.c[2],
                m.d.e[1],
                m.d.e[2],
                m.d.b,
                m.d.d,
            )
        ]
        test = list(
            x.name for x in m.block_data_objects(descend_into=(Block, Disjunct))
        )
        self.assertEqual(test, ref)

        test = list(
            x.name for x in m.block_data_objects(descend_into=(Disjunct, Block))
        )
        self.assertEqual(test, ref)

        ref = [
            x.name
            for x in (
                m.x,
                m.c[1].x,
                m.c[1].c[1].x,
                m.c[1].c[2].x,
                m.c[1].b.x,
                m.c[2].x,
                m.c[2].c[1].x,
                m.c[2].c[2].x,
                m.c[2].b.x,
                m.b.x,
                m.b.c[1].x,
                m.b.c[2].x,
                m.b.b.x,
            )
        ]
        test = list(x.name for x in m.component_data_objects(Var))
        self.assertEqual(test, ref)

        test = list(x.name for x in m.component_data_objects(Var, descend_into=Block))
        self.assertEqual(test, ref)

        test = list(
            x.name for x in m.component_data_objects(Var, descend_into=(Block,))
        )
        self.assertEqual(test, ref)

        ref = [
            x.name
            for x in (
                m.x,
                m.e[1].binary_indicator_var,
                m.e[1].x,
                m.e[1].e[1].binary_indicator_var,
                m.e[1].e[1].x,
                m.e[1].e[2].binary_indicator_var,
                m.e[1].e[2].x,
                m.e[1].d.binary_indicator_var,
                m.e[1].d.x,
                m.e[2].binary_indicator_var,
                m.e[2].x,
                m.e[2].e[1].binary_indicator_var,
                m.e[2].e[1].x,
                m.e[2].e[2].binary_indicator_var,
                m.e[2].e[2].x,
                m.e[2].d.binary_indicator_var,
                m.e[2].d.x,
                m.d.binary_indicator_var,
                m.d.x,
                m.d.e[1].binary_indicator_var,
                m.d.e[1].x,
                m.d.e[2].binary_indicator_var,
                m.d.e[2].x,
                m.d.d.binary_indicator_var,
                m.d.d.x,
            )
        ]
        test = list(
            x.name for x in m.component_data_objects(Var, descend_into=Disjunct)
        )
        self.assertEqual(test, ref)

        ref = [
            x.name
            for x in (
                m.x,
                m.c[1].x,
                m.c[1].c[1].x,
                m.c[1].c[2].x,
                m.c[1].e[1].binary_indicator_var,
                m.c[1].e[1].x,
                m.c[1].e[2].binary_indicator_var,
                m.c[1].e[2].x,
                m.c[1].b.x,
                m.c[1].d.binary_indicator_var,
                m.c[1].d.x,
                m.c[2].x,
                m.c[2].c[1].x,
                m.c[2].c[2].x,
                m.c[2].e[1].binary_indicator_var,
                m.c[2].e[1].x,
                m.c[2].e[2].binary_indicator_var,
                m.c[2].e[2].x,
                m.c[2].b.x,
                m.c[2].d.binary_indicator_var,
                m.c[2].d.x,
                m.e[1].binary_indicator_var,
                m.e[1].x,
                m.e[1].c[1].x,
                m.e[1].c[2].x,
                m.e[1].e[1].binary_indicator_var,
                m.e[1].e[1].x,
                m.e[1].e[2].binary_indicator_var,
                m.e[1].e[2].x,
                m.e[1].b.x,
                m.e[1].d.binary_indicator_var,
                m.e[1].d.x,
                m.e[2].binary_indicator_var,
                m.e[2].x,
                m.e[2].c[1].x,
                m.e[2].c[2].x,
                m.e[2].e[1].binary_indicator_var,
                m.e[2].e[1].x,
                m.e[2].e[2].binary_indicator_var,
                m.e[2].e[2].x,
                m.e[2].b.x,
                m.e[2].d.binary_indicator_var,
                m.e[2].d.x,
                m.b.x,
                m.b.c[1].x,
                m.b.c[2].x,
                m.b.e[1].binary_indicator_var,
                m.b.e[1].x,
                m.b.e[2].binary_indicator_var,
                m.b.e[2].x,
                m.b.b.x,
                m.b.d.binary_indicator_var,
                m.b.d.x,
                m.d.binary_indicator_var,
                m.d.x,
                m.d.c[1].x,
                m.d.c[2].x,
                m.d.e[1].binary_indicator_var,
                m.d.e[1].x,
                m.d.e[2].binary_indicator_var,
                m.d.e[2].x,
                m.d.b.x,
                m.d.d.binary_indicator_var,
                m.d.d.x,
            )
        ]
        test = list(
            x.name
            for x in m.component_data_objects(Var, descend_into=(Block, Disjunct))
        )
        self.assertEqual(test, ref)

    def test_deepcopy(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.c = Constraint(expr=m.x**2 + m.y[1] <= 5)
        m.b = Block()
        m.b.x = Var()
        m.b.y = Var([1, 2])
        m.b.c = Constraint(expr=m.x**2 + m.y[1] + m.b.x**2 + m.b.y[1] <= 10)

        n = deepcopy(m)
        self.assertNotEqual(id(m), id(n))

        self.assertNotEqual(id(m.x), id(n.x))
        self.assertIs(m.x.parent_block(), m)
        self.assertIs(m.x.parent_component(), m.x)
        self.assertIs(n.x.parent_block(), n)
        self.assertIs(n.x.parent_component(), n.x)

        self.assertNotEqual(id(m.y), id(n.y))
        self.assertIs(m.y.parent_block(), m)
        self.assertIs(m.y[1].parent_component(), m.y)
        self.assertIs(n.y.parent_block(), n)
        self.assertIs(n.y[1].parent_component(), n.y)

        self.assertNotEqual(id(m.c), id(n.c))
        self.assertIs(m.c.parent_block(), m)
        self.assertIs(m.c.parent_component(), m.c)
        self.assertIs(n.c.parent_block(), n)
        self.assertIs(n.c.parent_component(), n.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.c.body)),
            sorted(id(x) for x in (m.x, m.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.c.body)),
            sorted(id(x) for x in (n.x, n.y[1])),
        )

        self.assertNotEqual(id(m.b), id(n.b))
        self.assertIs(m.b.parent_block(), m)
        self.assertIs(m.b.parent_component(), m.b)
        self.assertIs(n.b.parent_block(), n)
        self.assertIs(n.b.parent_component(), n.b)

        self.assertNotEqual(id(m.b.x), id(n.b.x))
        self.assertIs(m.b.x.parent_block(), m.b)
        self.assertIs(m.b.x.parent_component(), m.b.x)
        self.assertIs(n.b.x.parent_block(), n.b)
        self.assertIs(n.b.x.parent_component(), n.b.x)

        self.assertNotEqual(id(m.b.y), id(n.b.y))
        self.assertIs(m.b.y.parent_block(), m.b)
        self.assertIs(m.b.y[1].parent_component(), m.b.y)
        self.assertIs(n.b.y.parent_block(), n.b)
        self.assertIs(n.b.y[1].parent_component(), n.b.y)

        self.assertNotEqual(id(m.b.c), id(n.b.c))
        self.assertIs(m.b.c.parent_block(), m.b)
        self.assertIs(m.b.c.parent_component(), m.b.c)
        self.assertIs(n.b.c.parent_block(), n.b)
        self.assertIs(n.b.c.parent_component(), n.b.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.b.c.body)),
            sorted(id(x) for x in (m.x, m.y[1], m.b.x, m.b.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.b.c.body)),
            sorted(id(x) for x in (n.x, n.y[1], n.b.x, n.b.y[1])),
        )

    def test_clone_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.c = Constraint(expr=m.x**2 + m.y[1] <= 5)
        m.b = Block()
        m.b.x = Var()
        m.b.y = Var([1, 2])
        m.b.c = Constraint(expr=m.x**2 + m.y[1] + m.b.x**2 + m.b.y[1] <= 10)

        n = m.clone()
        self.assertNotEqual(id(m), id(n))

        self.assertNotEqual(id(m.x), id(n.x))
        self.assertIs(m.x.parent_block(), m)
        self.assertIs(m.x.parent_component(), m.x)
        self.assertIs(n.x.parent_block(), n)
        self.assertIs(n.x.parent_component(), n.x)

        self.assertNotEqual(id(m.y), id(n.y))
        self.assertIs(m.y.parent_block(), m)
        self.assertIs(m.y[1].parent_component(), m.y)
        self.assertIs(n.y.parent_block(), n)
        self.assertIs(n.y[1].parent_component(), n.y)

        self.assertNotEqual(id(m.c), id(n.c))
        self.assertIs(m.c.parent_block(), m)
        self.assertIs(m.c.parent_component(), m.c)
        self.assertIs(n.c.parent_block(), n)
        self.assertIs(n.c.parent_component(), n.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.c.body)),
            sorted(id(x) for x in (m.x, m.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.c.body)),
            sorted(id(x) for x in (n.x, n.y[1])),
        )

        self.assertNotEqual(id(m.b), id(n.b))
        self.assertIs(m.b.parent_block(), m)
        self.assertIs(m.b.parent_component(), m.b)
        self.assertIs(n.b.parent_block(), n)
        self.assertIs(n.b.parent_component(), n.b)

        self.assertNotEqual(id(m.b.x), id(n.b.x))
        self.assertIs(m.b.x.parent_block(), m.b)
        self.assertIs(m.b.x.parent_component(), m.b.x)
        self.assertIs(n.b.x.parent_block(), n.b)
        self.assertIs(n.b.x.parent_component(), n.b.x)

        self.assertNotEqual(id(m.b.y), id(n.b.y))
        self.assertIs(m.b.y.parent_block(), m.b)
        self.assertIs(m.b.y[1].parent_component(), m.b.y)
        self.assertIs(n.b.y.parent_block(), n.b)
        self.assertIs(n.b.y[1].parent_component(), n.b.y)

        self.assertNotEqual(id(m.b.c), id(n.b.c))
        self.assertIs(m.b.c.parent_block(), m.b)
        self.assertIs(m.b.c.parent_component(), m.b.c)
        self.assertIs(n.b.c.parent_block(), n.b)
        self.assertIs(n.b.c.parent_component(), n.b.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.b.c.body)),
            sorted(id(x) for x in (m.x, m.y[1], m.b.x, m.b.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.b.c.body)),
            sorted(id(x) for x in (n.x, n.y[1], n.b.x, n.b.y[1])),
        )

    def test_clone_subblock(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.c = Constraint(expr=m.x**2 + m.y[1] <= 5)
        m.b = Block()
        m.b.x = Var()
        m.b.y = Var([1, 2])
        m.b.c = Constraint(expr=m.x**2 + m.y[1] + m.b.x**2 + m.b.y[1] <= 10)

        nb = m.b.clone()

        self.assertNotEqual(id(m.b), id(nb))
        self.assertIs(m.b.parent_block(), m)
        self.assertIs(m.b.parent_component(), m.b)
        self.assertIs(nb.parent_block(), None)
        self.assertIs(nb.parent_component(), nb)

        self.assertNotEqual(id(m.b.x), id(nb.x))
        self.assertIs(m.b.x.parent_block(), m.b)
        self.assertIs(m.b.x.parent_component(), m.b.x)
        self.assertIs(nb.x.parent_block(), nb)
        self.assertIs(nb.x.parent_component(), nb.x)

        self.assertNotEqual(id(m.b.y), id(nb.y))
        self.assertIs(m.b.y.parent_block(), m.b)
        self.assertIs(m.b.y[1].parent_component(), m.b.y)
        self.assertIs(nb.y.parent_block(), nb)
        self.assertIs(nb.y[1].parent_component(), nb.y)

        self.assertNotEqual(id(m.b.c), id(nb.c))
        self.assertIs(m.b.c.parent_block(), m.b)
        self.assertIs(m.b.c.parent_component(), m.b.c)
        self.assertIs(nb.c.parent_block(), nb)
        self.assertIs(nb.c.parent_component(), nb.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.b.c.body)),
            sorted(id(x) for x in (m.x, m.y[1], m.b.x, m.b.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(nb.c.body)),
            sorted(id(x) for x in (m.x, m.y[1], nb.x, nb.y[1])),
        )

    def test_clone_indexed_subblock(self):
        m = ConcreteModel()

        @m.Block([1, 2, 3])
        def blk(b, i):
            b.IDX = RangeSet(i)
            b.x = Var(b.IDX)

        m.c = Block(rule=m.blk[2].clone())

        self.assertEqual([1, 2], list(m.c.IDX))
        self.assertEqual(list(m.blk[2].IDX), list(m.c.IDX))
        self.assertIsNot(m.blk[2].IDX, m.c.IDX)
        self.assertIsNot(m.blk[2].x, m.c.x)
        self.assertIsNot(m.blk[2].IDX, m.c.x.index_set())
        self.assertIs(m.c.IDX, m.c.x.index_set())
        self.assertIs(m.c.parent_component(), m.c)
        self.assertIs(m.c.parent_block(), m)

        m.c1 = Block()
        m.c1.transfer_attributes_from(m.blk[3].clone())

        self.assertEqual([1, 2, 3], list(m.c1.IDX))
        self.assertEqual(list(m.blk[3].IDX), list(m.c1.IDX))
        self.assertIsNot(m.blk[3].IDX, m.c1.IDX)
        self.assertIsNot(m.blk[3].x, m.c1.x)
        self.assertIsNot(m.blk[3].IDX, m.c1.x.index_set())
        self.assertIs(m.c1.IDX, m.c1.x.index_set())
        self.assertIs(m.c1.parent_component(), m.c1)
        self.assertIs(m.c1.parent_block(), m)

        @m.Block([1, 2, 3])
        def d(b, i):
            return b.model().blk[i].clone()

        for i in [1, 2, 3]:
            self.assertEqual(list(range(1, i + 1)), list(m.d[i].IDX))
            self.assertEqual(list(m.blk[i].IDX), list(m.d[i].IDX))
            self.assertIsNot(m.blk[i].IDX, m.d[i].IDX)
            self.assertIsNot(m.blk[i].x, m.d[i].x)
            self.assertIsNot(m.blk[i].IDX, m.d[i].x.index_set())
            self.assertIs(m.d[i].IDX, m.d[i].x.index_set())
            self.assertIs(m.d[i].parent_component(), m.d)
            self.assertIs(m.d[i].parent_block(), m)

    def test_clone_unclonable_attribute(self):
        class foo(object):
            def __deepcopy__(bogus):
                pass

        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.bad1 = foo()
        m.c = Constraint(expr=m.x**2 + m.y[1] <= 5)
        m.b = Block()
        m.b.x = Var()
        m.b.y = Var([1, 2])
        m.b.bad2 = foo()
        m.b.c = Constraint(expr=m.x**2 + m.y[1] + m.b.x**2 + m.b.y[1] <= 10)

        # Check the paranoid warning
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            nb = deepcopy(m.b)
        # without the scope, the whole model is cloned!
        self.assertIn(
            "'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue()
        )
        self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
        self.assertIn("outside the scope of Block.clone()", OUTPUT.getvalue())
        self.assertTrue(hasattr(m.b, 'bad2'))
        self.assertIsNotNone(m.b.bad2)
        self.assertTrue(hasattr(nb, 'bad2'))
        self.assertIsNone(nb.bad2)

        # Simple tests for the subblock
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            nb = m.b.clone()
        self.assertNotIn(
            "'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue()
        )
        self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
        self.assertNotIn("'__paranoid__'", OUTPUT.getvalue())
        self.assertTrue(hasattr(m.b, 'bad2'))
        self.assertIsNotNone(m.b.bad2)
        self.assertTrue(hasattr(nb, 'bad2'))
        self.assertIsNone(nb.bad2)

        # more involved tests for the model
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            n = m.clone()
        self.assertIn(
            "'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue()
        )
        self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
        self.assertNotIn("'__paranoid__'", OUTPUT.getvalue())
        self.assertTrue(hasattr(m, 'bad1'))
        self.assertIsNotNone(m.bad1)
        self.assertTrue(hasattr(n, 'bad1'))
        self.assertIsNone(n.bad1)
        self.assertTrue(hasattr(m.b, 'bad2'))
        self.assertIsNotNone(m.b.bad2)
        self.assertTrue(hasattr(n.b, 'bad2'))
        self.assertIsNone(n.b.bad2)

        self.assertNotEqual(id(m), id(n))

        self.assertNotEqual(id(m.x), id(n.x))
        self.assertIs(m.x.parent_block(), m)
        self.assertIs(m.x.parent_component(), m.x)
        self.assertIs(n.x.parent_block(), n)
        self.assertIs(n.x.parent_component(), n.x)

        self.assertNotEqual(id(m.y), id(n.y))
        self.assertIs(m.y.parent_block(), m)
        self.assertIs(m.y[1].parent_component(), m.y)
        self.assertIs(n.y.parent_block(), n)
        self.assertIs(n.y[1].parent_component(), n.y)

        self.assertNotEqual(id(m.c), id(n.c))
        self.assertIs(m.c.parent_block(), m)
        self.assertIs(m.c.parent_component(), m.c)
        self.assertIs(n.c.parent_block(), n)
        self.assertIs(n.c.parent_component(), n.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.c.body)),
            sorted(id(x) for x in (m.x, m.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.c.body)),
            sorted(id(x) for x in (n.x, n.y[1])),
        )

        self.assertNotEqual(id(m.b), id(n.b))
        self.assertIs(m.b.parent_block(), m)
        self.assertIs(m.b.parent_component(), m.b)
        self.assertIs(n.b.parent_block(), n)
        self.assertIs(n.b.parent_component(), n.b)

        self.assertNotEqual(id(m.b.x), id(n.b.x))
        self.assertIs(m.b.x.parent_block(), m.b)
        self.assertIs(m.b.x.parent_component(), m.b.x)
        self.assertIs(n.b.x.parent_block(), n.b)
        self.assertIs(n.b.x.parent_component(), n.b.x)

        self.assertNotEqual(id(m.b.y), id(n.b.y))
        self.assertIs(m.b.y.parent_block(), m.b)
        self.assertIs(m.b.y[1].parent_component(), m.b.y)
        self.assertIs(n.b.y.parent_block(), n.b)
        self.assertIs(n.b.y[1].parent_component(), n.b.y)

        self.assertNotEqual(id(m.b.c), id(n.b.c))
        self.assertIs(m.b.c.parent_block(), m.b)
        self.assertIs(m.b.c.parent_component(), m.b.c)
        self.assertIs(n.b.c.parent_block(), n.b)
        self.assertIs(n.b.c.parent_component(), n.b.c)
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(m.b.c.body)),
            sorted(id(x) for x in (m.x, m.y[1], m.b.x, m.b.y[1])),
        )
        self.assertEqual(
            sorted(id(x) for x in EXPR.identify_variables(n.b.c.body)),
            sorted(id(x) for x in (n.x, n.y[1], n.b.x, n.b.y[1])),
        )

    def test_pprint(self):
        m = HierarchicalModel().model
        buf = StringIO()
        m.pprint(ostream=buf)
        ref = """2 Set Declarations
    a1_IDX : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :    2 : {5, 4}
    a3_IDX : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :    2 : {6, 7}

3 Block Declarations
    a : Size=3, Index={1, 2, 3}, Active=True
        a[1] : Active=True
            2 Block Declarations
                c : Size=2, Index=a1_IDX, Active=True
                    a[1].c[4] : Active=True
                        0 Declarations: 
                    a[1].c[5] : Active=True
                        0 Declarations: 
                d : Size=1, Index=None, Active=True
                    0 Declarations: 

            2 Declarations: d c
        a[2] : Active=True
            0 Declarations: 
        a[3] : Active=True
            2 Block Declarations
                e : Size=1, Index=None, Active=True
                    0 Declarations: 
                f : Size=2, Index=a3_IDX, Active=True
                    a[3].f[6] : Active=True
                        0 Declarations: 
                    a[3].f[7] : Active=True
                        0 Declarations: 

            2 Declarations: e f
    b : Size=1, Index=None, Active=True
        0 Declarations: 
    c : Size=1, Index=None, Active=True
        0 Declarations: 

5 Declarations: a1_IDX a3_IDX c a b
"""
        self.assertEqual(ref, buf.getvalue())

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_solve1(self):
        model = Block(concrete=True)
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, bounds=(-1, 1))

        def obj_rule(model):
            return sum_product(model.x)

        model.obj = Objective(rule=obj_rule)

        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i * model.x[i]
            return expr == 0

        model.c = Constraint(rule=c_rule)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "solve1.out"), format='json')
        with (
            open(join(currdir, "solve1.out"), 'r') as out,
            open(join(currdir, "solve1.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

        #
        def d_rule(model):
            return model.x[1] >= 0

        model.d = Constraint(rule=d_rule)
        model.d.deactivate()
        results = opt.solve(model)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "solve1x.out"), format='json')
        with (
            open(join(currdir, "solve1x.out"), 'r') as out,
            open(join(currdir, "solve1.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )
        #
        model.d.activate()
        results = opt.solve(model)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "solve1a.out"), format='json')
        with (
            open(join(currdir, "solve1a.out"), 'r') as out,
            open(join(currdir, "solve1a.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )
        #
        model.d.deactivate()

        def e_rule(model, i):
            return model.x[i] >= 0

        model.e = Constraint(model.A, rule=e_rule)
        for i in model.A:
            model.e[i].deactivate()
        results = opt.solve(model)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "solve1y.out"), format='json')
        with (
            open(join(currdir, "solve1y.out"), 'r') as out,
            open(join(currdir, "solve1.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )
        #
        model.e.activate()
        results = opt.solve(model)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, "solve1b.out"), format='json')
        with (
            open(join(currdir, "solve1b.out"), 'r') as out,
            open(join(currdir, "solve1b.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_solve4(self):
        model = Block(concrete=True)
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, bounds=(-1, 1))

        def obj_rule(model):
            return sum_product(model.x)

        model.obj = Objective(rule=obj_rule)

        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i * model.x[i]
            return expr == 0

        model.c = Constraint(rule=c_rule)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, 'solve4.out'), format='json')
        with (
            open(join(currdir, "solve4.out"), 'r') as out,
            open(join(currdir, "solve1.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_solve6(self):
        #
        # Test that solution values have complete block names:
        #   b.obj
        #   b.x
        #
        model = Block(concrete=True)
        model.y = Var(bounds=(-1, 1))
        model.b = Block()
        model.b.A = RangeSet(1, 4)
        model.b.x = Var(model.b.A, bounds=(-1, 1))

        def obj_rule(block):
            return sum_product(block.x)

        model.b.obj = Objective(rule=obj_rule)

        def c_rule(model):
            expr = model.y
            for i in model.b.A:
                expr += i * model.b.x[i]
            return expr == 0

        model.c = Constraint(rule=c_rule)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=join(currdir, 'solve6.out'), format='json')
        with (
            open(join(currdir, "solve6.out"), 'r') as out,
            open(join(currdir, "solve6.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_solve7(self):
        #
        # Test that solution values are written with appropriate
        # quotations in results
        #
        model = Block(concrete=True)
        model.y = Var(bounds=(-1, 1))
        model.A = RangeSet(1, 4)
        model.B = Set(initialize=['A B', 'C,D', 'E'])
        model.x = Var(model.A, model.B, bounds=(-1, 1))

        def obj_rule(model):
            return sum_product(model.x)

        model.obj = Objective(rule=obj_rule)

        def c_rule(model):
            expr = model.y
            for i in model.A:
                for j in model.B:
                    expr += i * model.x[i, j]
            return expr == 0

        model.c = Constraint(rule=c_rule)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        # model.display()
        model.solutions.store_to(results)
        results.write(filename=join(currdir, 'solve7.out'), format='json')
        with (
            open(join(currdir, "solve7.out"), 'r') as out,
            open(join(currdir, "solve7.txt"), 'r') as txt,
        ):
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Block(model.C)

    def test_decorated_definition(self):
        model = ConcreteModel()
        model.I = Set(initialize=[1, 2, 3])
        model.x = Var(model.I)

        @model.Constraint()
        def scalar_constraint(m):
            return m.x[1] ** 2 <= 0

        self.assertTrue(hasattr(model, 'scalar_constraint'))
        self.assertIs(model.scalar_constraint.ctype, Constraint)
        self.assertEqual(len(model.scalar_constraint), 1)
        self.assertIs(type(scalar_constraint), types.FunctionType)

        @model.Constraint(model.I)
        def vector_constraint(m, i):
            return m.x[i] ** 2 <= 0

        self.assertTrue(hasattr(model, 'vector_constraint'))
        self.assertIs(model.vector_constraint.ctype, Constraint)
        self.assertEqual(len(model.vector_constraint), 3)
        self.assertIs(type(vector_constraint), types.FunctionType)

    def test_reserved_words(self):
        m = ConcreteModel()
        self.assertRaisesRegex(
            ValueError,
            ".*using the name of a reserved attribute",
            m.add_component,
            "add_component",
            Var(),
        )
        with self.assertRaisesRegex(
            ValueError, ".*using the name of a reserved attribute"
        ):
            m.add_component = Var()
        m.foo = Var()

        m.b = DerivedBlock()
        self.assertRaisesRegex(
            ValueError,
            ".*using the name of a reserved attribute",
            m.b.add_component,
            "add_component",
            Var(),
        )
        self.assertRaisesRegex(
            ValueError,
            ".*using the name of a reserved attribute",
            m.b.add_component,
            "foo",
            Var(),
        )
        with self.assertRaisesRegex(
            ValueError, ".*using the name of a reserved attribute"
        ):
            m.b.foo = Var()

        class DerivedBlockReservedComp(DerivedBlock):
            def __init__(self, *args, **kwargs):
                """Constructor"""
                super(DerivedBlock, self).__init__(*args, **kwargs)
                with self._declare_reserved_components():
                    self.x = Var()

        DerivedBlockReservedComp._Block_reserved_words = set(
            dir(DerivedBlockReservedComp())
        )

        m.c = DerivedBlockReservedComp()

        with self.assertRaisesRegex(
            ValueError, "Attempting to delete a reserved block component"
        ):
            m.c.del_component('x')

        with self.assertRaisesRegex(
            ValueError, "Attempting to delete a reserved block component"
        ):
            m.c.x = Var()

        class RestrictedBlock(ScalarBlock):
            _Block_reserved_words = Any - {'start', 'end'}

        m.d = RestrictedBlock()
        m.d.start = v = Var()
        self.assertIs(m.d.start, v)
        with self.assertRaisesRegex(
            ValueError, "using the name of a reserved attribute"
        ):
            m.d.step = Var()

        #
        # Overriding attributes with non-components is (currently) allowed
        #
        m.add_component = 5
        self.assertIs(m.add_component, 5)
        m.b.add_component = 6
        self.assertIs(m.b.add_component, 6)
        m.b.foo = 7
        self.assertIs(m.b.foo, 7)

    def test_write_exceptions(self):
        m = Block()
        with self.assertRaisesRegex(
            ValueError, ".*Could not infer file format from file name"
        ):
            m.write(filename="foo.bogus")

        with self.assertRaisesRegex(ValueError, ".*Cannot write model in format"):
            m.write(format="bogus")

    def test_custom_block(self):
        @declare_custom_block('TestingBlock')
        class TestingBlockData(BlockData):
            def __init__(self, component):
                BlockData.__init__(self, component)
                logging.getLogger(__name__).warning("TestingBlockData.__init__")

        self.assertIn('TestingBlock', globals())
        self.assertIn('ScalarTestingBlock', globals())
        self.assertIn('IndexedTestingBlock', globals())
        self.assertIs(TestingBlock.__module__, __name__)
        self.assertIs(ScalarTestingBlock.__module__, __name__)
        self.assertIs(IndexedTestingBlock.__module__, __name__)

        with LoggingIntercept() as LOG:
            obj = TestingBlock()
        self.assertIs(type(obj), ScalarTestingBlock)
        self.assertEqual(LOG.getvalue().strip(), "TestingBlockData.__init__")

        with LoggingIntercept() as LOG:
            obj = TestingBlock([1, 2])
        self.assertIs(type(obj), IndexedTestingBlock)
        self.assertEqual(LOG.getvalue(), "")

        # Test that we can derive from a ScalarCustomBlock
        class DerivedScalarTestingBlock(ScalarTestingBlock):
            pass

        with LoggingIntercept() as LOG:
            obj = DerivedScalarTestingBlock()
        self.assertIs(type(obj), DerivedScalarTestingBlock)
        self.assertEqual(LOG.getvalue().strip(), "TestingBlockData.__init__")

    def test_custom_block_ctypes(self):
        @declare_custom_block('TestingBlock')
        class TestingBlockData(BlockData):
            pass

        self.assertIs(TestingBlock().ctype, Block)

        @declare_custom_block('TestingBlock', True)
        class TestingBlockData(BlockData):
            pass

        self.assertIs(TestingBlock().ctype, TestingBlock)

        @declare_custom_block('TestingBlock', Constraint)
        class TestingBlockData(BlockData):
            pass

        self.assertIs(TestingBlock().ctype, Constraint)

        with self.assertRaisesRegex(
            ValueError,
            r"Expected new_ctype to be either type or 'True'; received: \[\]",
        ):

            @declare_custom_block('TestingBlock', [])
            class TestingBlockData(BlockData):
                pass

    def test_custom_block_override_pprint(self):
        @declare_custom_block('TempBlock')
        class TempBlockData(BlockData):
            def pprint(self, ostream=None, verbose=False, prefix=""):
                ostream.write('Testing pprint of a custom block.')

        correct_s = 'Testing pprint of a custom block.'
        b = TempBlock(concrete=True)
        stream = StringIO()
        b.pprint(ostream=stream)
        self.assertEqual(correct_s, stream.getvalue())

    def test_custom_block_default_rule(self):
        """Tests the decorator with `build` method, but without options"""

        @declare_custom_block("LocalFooBlock", rule="build")
        class LocalFooBlockData(BlockData):
            def build(self, *args):
                self.x = Var(list(args))
                self.y = Var()

        m = ConcreteModel()
        m.blk_without_index = LocalFooBlock()
        m.blk_1 = LocalFooBlock([1, 2, 3])
        m.blk_2 = LocalFooBlock([4, 5], [6, 7])

        self.assertIn("x", m.blk_without_index.component_map())
        self.assertIn("y", m.blk_without_index.component_map())
        self.assertIn("x", m.blk_1[3].component_map())
        self.assertIn("x", m.blk_2[4, 6].component_map())

        self.assertEqual(len(m.blk_1), 3)
        self.assertEqual(len(m.blk_2), 4)

        self.assertEqual(len(m.blk_1[2].x), 1)
        self.assertEqual(len(m.blk_2[4, 6].x), 2)

    def test_custom_block_default_rule_options(self):
        """Tests the decorator with `build` method and model options"""

        options = {"capex": 42, "opex": 24}
        m = ConcreteModel()
        m.blk_without_index = FooBlock(capex=42, opex=24)
        m.blk_1 = FooBlock([1, 2, 3], **options)
        m.blk_2 = FooBlock([4, 5], [6, 7], **options)

        self.assertEqual(m.blk_without_index.capex, 42)
        self.assertEqual(m.blk_without_index.opex, 24)

        self.assertEqual(m.blk_1[3].capex, 42)
        self.assertEqual(m.blk_2[4, 7].opex, 24)

        new_m = pickle.loads(pickle.dumps(m))
        self.assertIs(new_m.blk_without_index.__class__, m.blk_without_index.__class__)
        self.assertIs(new_m.blk_1.__class__, m.blk_1.__class__)
        self.assertIs(new_m.blk_2.__class__, m.blk_2.__class__)

        self.assertIsNot(new_m.blk_without_index, m.blk_without_index)
        self.assertIsNot(new_m.blk_1, m.blk_1)
        self.assertIsNot(new_m.blk_2, m.blk_2)

        with self.assertRaisesRegex(
            TypeError, "missing 2 required keyword-only arguments"
        ):
            # missing 2 required keyword arguments
            m.blk_3 = FooBlock()

    def test_custom_block_user_rule(self):
        """Tests if the default rule can be overwritten"""

        @declare_custom_block("FooBlock")
        class FooBlockData(BlockData):
            def build(self, *args):
                self.x = Var(list(args))
                self.y = Var()

        def _new_rule(blk):
            blk.a = Var()
            blk.b = Var()

        m = ConcreteModel()
        m.blk = FooBlock(rule=_new_rule)

        self.assertNotIn("x", m.blk.component_map())
        self.assertNotIn("y", m.blk.component_map())
        self.assertIn("a", m.blk.component_map())
        self.assertIn("b", m.blk.component_map())

    def test_block_rules(self):
        m = ConcreteModel()
        m.I = Set()
        _rule_ = []

        def _block_rule(b, i):
            _rule_.append(i)
            b.x = Var(range(i))

        m.b = Block(m.I, rule=_block_rule)
        # I is empty: no rules called
        self.assertEqual(_rule_, [])
        m.I.update([1, 3, 5])
        # Fetching a new block will call the rule
        _b = m.b[3]
        self.assertEqual(len(m.b), 1)
        self.assertEqual(_rule_, [3])
        self.assertIn('x', _b.component_map())
        self.assertIn('x', m.b[3].component_map())

        # If you transfer the attributes directly, the rule will still
        # be called.
        _tmp = Block()
        _tmp.y = Var(range(3))
        m.b[5].transfer_attributes_from(_tmp)
        self.assertEqual(len(m.b), 2)
        self.assertEqual(_rule_, [3, 5])
        self.assertIn('x', m.b[5].component_map())
        self.assertIn('y', m.b[5].component_map())

        # We do not support block assignment (and the rule will NOT be
        # called)
        _tmp = Block()
        _tmp.y = Var(range(3))
        with self.assertRaisesRegex(
            RuntimeError, "Block components do not support assignment or set_value"
        ):
            m.b[1] = _tmp
        self.assertEqual(len(m.b), 2)
        self.assertEqual(_rule_, [3, 5])

        # Blocks with non-finite indexing sets cannot be automatically
        # populated (even if they have a rule!)
        def _bb_rule(b, i, j):
            _rule_.append((i, j))
            b.x = Var(RangeSet(i))
            b.y = Var(RangeSet(j))

        m.bb = Block(m.I, NonNegativeIntegers, rule=_bb_rule)
        self.assertEqual(_rule_, [3, 5])
        _b = m.bb[3, 5]
        self.assertEqual(_rule_, [3, 5, (3, 5)])
        self.assertEqual(len(m.bb), 1)
        self.assertEqual(len(_b.x), 3)
        self.assertEqual(len(_b.y), 5)

    def test_derived_block_construction(self):
        # This tests a case where a derived block doesn't follow the
        # assumption that unconstructed scalar blocks initialize
        # `_data[None] = self` (therefore doesn't fully support abstract
        # models).  At one point, that was causing the block rule to
        # fire twice during construction.
        class ConcreteBlock(Block):
            pass

        class ScalarConcreteBlock(BlockData, ConcreteBlock):
            def __init__(self, *args, **kwds):
                BlockData.__init__(self, component=self)
                ConcreteBlock.__init__(self, *args, **kwds)

        _buf = []

        def _rule(b):
            _buf.append(1)

        m = ConcreteModel()
        m.b = ScalarConcreteBlock(rule=_rule)
        self.assertEqual(_buf, [1])

    def test_abstract_construction(self):
        m = AbstractModel()
        m.I = Set()

        def b_rule(b, i):
            b.p = Param(default=i)
            b.J = Set(initialize=range(i))

        m.b = Block(m.I, rule=b_rule)

        i = m.create_instance(
            {
                None: {
                    'I': {None: [1, 2, 3, 4]},
                    'b': {
                        1: {'p': {None: 10}, 'J': {None: [7, 8]}},
                        2: {'p': {None: 12}},
                        3: {'J': {None: [9]}},
                    },
                }
            }
        )
        self.assertEqual(list(i.I), [1, 2, 3, 4])
        self.assertEqual(len(i.b), 4)
        self.assertEqual(list(i.b[1].J), [7, 8])
        self.assertEqual(list(i.b[2].J), [0, 1])
        self.assertEqual(list(i.b[3].J), [9])
        self.assertEqual(list(i.b[4].J), [0, 1, 2, 3])
        self.assertEqual(value(i.b[1].p), 10)
        self.assertEqual(value(i.b[2].p), 12)
        self.assertEqual(value(i.b[3].p), 3)
        self.assertEqual(value(i.b[4].p), 4)

    def test_abstract_transfer_construction(self):
        m = AbstractModel()
        m.I = Set()

        def b_rule(_b, i):
            b = Block()
            b.p = Param(default=i)
            b.J = Set(initialize=range(i))
            return b

        m.b = Block(m.I, rule=b_rule)

        i = m.create_instance(
            {
                None: {
                    'I': {None: [1, 2, 3, 4]},
                    'b': {
                        1: {'p': {None: 10}, 'J': {None: [7, 8]}},
                        2: {'p': {None: 12}},
                        3: {'J': {None: [9]}},
                    },
                }
            }
        )
        self.assertEqual(list(i.I), [1, 2, 3, 4])
        self.assertEqual(len(i.b), 4)
        self.assertEqual(list(i.b[1].J), [7, 8])
        self.assertEqual(list(i.b[2].J), [0, 1])
        self.assertEqual(list(i.b[3].J), [9])
        self.assertEqual(list(i.b[4].J), [0, 1, 2, 3])
        self.assertEqual(value(i.b[1].p), 10)
        self.assertEqual(value(i.b[2].p), 12)
        self.assertEqual(value(i.b[3].p), 3)
        self.assertEqual(value(i.b[4].p), 4)

    def test_deprecated_options(self):
        m = ConcreteModel()

        def b_rule(b, a=None):
            b.p = Param(initialize=a)

        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            m.b = Block(rule=b_rule, options={'a': 5})
        self.assertIn("The Block 'options=' keyword is deprecated.", OUTPUT.getvalue())
        self.assertEqual(value(m.b.p), 5)

        m = ConcreteModel()

        def b_rule(b, i, **kwds):
            b.p = Param(initialize=kwds.get('a', {}).get(i, 0))

        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            m.b = Block([1, 2, 3], rule=b_rule, options={'a': {1: 5, 2: 10}})
        self.assertIn("The Block 'options=' keyword is deprecated.", OUTPUT.getvalue())
        self.assertEqual(value(m.b[1].p), 5)
        self.assertEqual(value(m.b[2].p), 10)
        self.assertEqual(value(m.b[3].p), 0)

    def test_find_component_name(self):
        b = Block(concrete=True)
        b.v1 = Var()
        b.v2 = Var([1, 2])
        self.assertIs(b.find_component("v1"), b.v1)
        self.assertIs(b.find_component("v2[2]"), b.v2[2])

    def test_find_component_cuid(self):
        b = Block(concrete=True)
        b.v1 = Var()
        b.v2 = Var([1, 2])
        cuid1 = ComponentUID("v1")
        cuid2 = ComponentUID("v2[2]")
        self.assertIs(b.find_component(cuid1), b.v1)
        self.assertIs(b.find_component(cuid2), b.v2[2])

    def test_find_component_hierarchical(self):
        b1 = Block(concrete=True)
        b1.b2 = Block()
        b1.b2.v1 = Var()
        b1.b2.v2 = Var([1, 2])
        self.assertIs(b1.find_component("b2.v1"), b1.b2.v1)
        self.assertIs(b1.find_component("b2.v2[2]"), b1.b2.v2[2])

    def test_find_component_hierarchical_cuid(self):
        b1 = Block(concrete=True)
        b1.b2 = Block()
        b1.b2.v1 = Var()
        b1.b2.v2 = Var([1, 2])
        cuid1 = ComponentUID("b2.v1")
        cuid2 = ComponentUID("b2.v2[2]")
        self.assertIs(b1.find_component(cuid1), b1.b2.v1)
        self.assertIs(b1.find_component(cuid2), b1.b2.v2[2])

    def test_deduplicate_component_data_objects(self):
        m = ConcreteModel()
        m.b = Block()
        # Scalar, then reference
        m.x = Var()
        m.z_x = Reference(m.x)
        # Indexed, then reference
        m.I = Var([1, 3, 2])
        m.z_I = Reference(m.I)

        # Reference, then scalar
        m.b.y = Var()
        m.z_y = Reference(m.b.y)
        # Reference, then indexed
        m.b.J = Var([4, 6, 5])
        m.z_J = Reference(m.b.J)

        # Partial reference, then components
        m.c = Block([2, 1])
        m.c[1].A = Var([(0, 2), (1, 1)])
        m.c[2].A = Var([(0, 3), (1, 1)])
        m.z_AA = Reference(m.c[:].A[1, :])
        # duplicate rederence
        m.z_A = Reference(m.c[:].A[1, :])

        ans = list(m.component_data_objects(Var))
        self.assertEqual(
            ans,
            [
                m.x,
                m.I[1],
                m.I[3],
                m.I[2],
                m.b.y,
                m.b.J[4],
                m.b.J[6],
                m.b.J[5],
                m.c[2].A[1, 1],
                m.c[1].A[1, 1],
                m.c[2].A[0, 3],
                m.c[1].A[0, 2],
            ],
        )

        ans = list(m.component_data_objects(Var, sort=SortComponents.SORTED_INDICES))
        self.assertEqual(
            ans,
            [
                m.x,
                m.I[1],
                m.I[2],
                m.I[3],
                m.b.y,
                m.b.J[4],
                m.b.J[5],
                m.b.J[6],
                m.c[1].A[1, 1],
                m.c[2].A[1, 1],
                m.c[1].A[0, 2],
                m.c[2].A[0, 3],
            ],
        )

        ans = list(m.component_data_objects(Var, sort=SortComponents.ALPHABETICAL))
        self.assertEqual(
            ans,
            [
                m.I[1],
                m.I[3],
                m.I[2],
                m.x,
                m.c[2].A[1, 1],
                m.c[1].A[1, 1],
                m.b.J[4],
                m.b.J[6],
                m.b.J[5],
                m.b.y,
                m.c[2].A[0, 3],
                m.c[1].A[0, 2],
            ],
        )

        ans = list(
            m.component_data_objects(
                Var, sort=SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES
            )
        )
        self.assertEqual(
            ans,
            [
                m.I[1],
                m.I[2],
                m.I[3],
                m.x,
                m.c[1].A[1, 1],
                m.c[2].A[1, 1],
                m.b.J[4],
                m.b.J[5],
                m.b.J[6],
                m.b.y,
                m.c[1].A[0, 2],
                m.c[2].A[0, 3],
            ],
        )

    def test_deduplicate_component_data_iterindex(self):
        m = ConcreteModel()
        m.b = Block()
        # Scalar, then reference
        m.x = Var()
        m.z_x = Reference(m.x)
        # Indexed, then reference
        m.I = Var([1, 3, 2])
        m.z_I = Reference(m.I)

        # Reference, then scalar
        m.b.y = Var()
        m.z_y = Reference(m.b.y)
        # Reference, then indexed
        m.b.J = Var([4, 6, 5])
        m.z_J = Reference(m.b.J)

        # Partial reference, then components
        m.c = Block([2, 1])
        m.c[1].A = Var([(0, 2), (1, 1)])
        m.c[2].A = Var([(0, 3), (1, 1)])
        m.z_AA = Reference(m.c[:].A[1, :])
        # duplicate rederence
        m.z_A = Reference(m.c[:].A[1, :])

        ans = list(m.component_data_iterindex(Var))
        self.assertEqual(
            ans,
            [
                (('x', None), m.x),
                (('I', 1), m.I[1]),
                (('I', 3), m.I[3]),
                (('I', 2), m.I[2]),
                (('z_y', None), m.b.y),
                (('z_J', 4), m.b.J[4]),
                (('z_J', 6), m.b.J[6]),
                (('z_J', 5), m.b.J[5]),
                (('z_AA', (2, 1)), m.c[2].A[1, 1]),
                (('z_AA', (1, 1)), m.c[1].A[1, 1]),
                (('A', (0, 3)), m.c[2].A[0, 3]),
                (('A', (0, 2)), m.c[1].A[0, 2]),
            ],
        )

        ans = list(m.component_data_iterindex(Var, sort=SortComponents.SORTED_INDICES))
        self.assertEqual(
            ans,
            [
                (('x', None), m.x),
                (('I', 1), m.I[1]),
                (('I', 2), m.I[2]),
                (('I', 3), m.I[3]),
                (('z_y', None), m.b.y),
                (('z_J', 4), m.b.J[4]),
                (('z_J', 5), m.b.J[5]),
                (('z_J', 6), m.b.J[6]),
                (('z_AA', (1, 1)), m.c[1].A[1, 1]),
                (('z_AA', (2, 1)), m.c[2].A[1, 1]),
                (('A', (0, 2)), m.c[1].A[0, 2]),
                (('A', (0, 3)), m.c[2].A[0, 3]),
            ],
        )

        ans = list(m.component_data_iterindex(Var, sort=SortComponents.ALPHABETICAL))
        self.assertEqual(
            ans,
            [
                (('I', 1), m.I[1]),
                (('I', 3), m.I[3]),
                (('I', 2), m.I[2]),
                (('x', None), m.x),
                (('z_A', (2, 1)), m.c[2].A[1, 1]),
                (('z_A', (1, 1)), m.c[1].A[1, 1]),
                (('z_J', 4), m.b.J[4]),
                (('z_J', 6), m.b.J[6]),
                (('z_J', 5), m.b.J[5]),
                (('z_y', None), m.b.y),
                (('A', (0, 3)), m.c[2].A[0, 3]),
                (('A', (0, 2)), m.c[1].A[0, 2]),
            ],
        )

        ans = list(
            m.component_data_iterindex(
                Var, sort=SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES
            )
        )
        self.assertEqual(
            ans,
            [
                (('I', 1), m.I[1]),
                (('I', 2), m.I[2]),
                (('I', 3), m.I[3]),
                (('x', None), m.x),
                (('z_A', (1, 1)), m.c[1].A[1, 1]),
                (('z_A', (2, 1)), m.c[2].A[1, 1]),
                (('z_J', 4), m.b.J[4]),
                (('z_J', 5), m.b.J[5]),
                (('z_J', 6), m.b.J[6]),
                (('z_y', None), m.b.y),
                (('A', (0, 2)), m.c[1].A[0, 2]),
                (('A', (0, 3)), m.c[2].A[0, 3]),
            ],
        )

    def test_private_data(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.b = Block([1, 2])

        mfe = m.private_data()
        self.assertIsInstance(mfe, dict)
        self.assertEqual(len(mfe), 0)
        self.assertEqual(len(m._private_data), 1)
        self.assertIn('pyomo.core.tests.unit.test_block', m._private_data)
        self.assertIs(mfe, m._private_data['pyomo.core.tests.unit.test_block'])

        with self.assertRaisesRegex(
            ValueError,
            "All keys in the 'private_data' dictionary must "
            "be substrings of the caller's module name. "
            "Received 'no mice here' when calling private_data on Block "
            "'b'.",
        ):
            mfe2 = m.b.private_data('no mice here')

        mfe3 = m.b.b[1].private_data('pyomo.core.tests')
        self.assertIsInstance(mfe3, dict)
        self.assertEqual(len(mfe3), 0)
        self.assertIsInstance(m.b.b[1]._private_data, dict)
        self.assertEqual(len(m.b.b[1]._private_data), 1)
        self.assertIn('pyomo.core.tests', m.b.b[1]._private_data)
        self.assertIs(mfe3, m.b.b[1]._private_data['pyomo.core.tests'])
        mfe3['there are cookies'] = 'but no mice'

        mfe4 = m.b.b[1].private_data('pyomo.core.tests')
        self.assertIs(mfe4, mfe3)

    def test_register_private_data(self):
        _save = Block._private_data_initializers

        Block._private_data_initializers = pdi = _save.copy()
        pdi.clear()
        try:
            self.assertEqual(len(pdi), 0)
            b = Block(concrete=True)
            ps = b.private_data()
            self.assertEqual(ps, {})
            self.assertEqual(len(pdi), 1)
        finally:
            Block._private_data_initializers = _save

        def init():
            return {'a': None, 'b': 1}

        Block._private_data_initializers = pdi = _save.copy()
        pdi.clear()
        try:
            self.assertEqual(len(pdi), 0)
            Block.register_private_data_initializer(init)
            self.assertEqual(len(pdi), 1)

            b = Block(concrete=True)
            ps = b.private_data()
            self.assertEqual(ps, {'a': None, 'b': 1})
            self.assertEqual(len(pdi), 1)
        finally:
            Block._private_data_initializers = _save

        Block._private_data_initializers = pdi = _save.copy()
        pdi.clear()
        try:
            Block.register_private_data_initializer(init)
            self.assertEqual(len(pdi), 1)
            Block.register_private_data_initializer(init, 'pyomo')
            self.assertEqual(len(pdi), 2)

            with self.assertRaisesRegex(
                RuntimeError,
                r"Duplicate initializer registration for 'private_data' "
                r"dictionary \(scope=pyomo.core.tests.unit.test_block\)",
            ):
                Block.register_private_data_initializer(init)

            with self.assertRaisesRegex(
                ValueError,
                r"'private_data' scope must be substrings of the caller's "
                r"module name. Received 'invalid' when calling "
                r"register_private_data_initializer\(\).",
            ):
                Block.register_private_data_initializer(init, 'invalid')

            self.assertEqual(len(pdi), 2)
        finally:
            Block._private_data_initializers = _save


if __name__ == "__main__":
    unittest.main()
