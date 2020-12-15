#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Unit Tests for pyomo.dae.misc
"""
import os
from os.path import abspath, dirname

from six import StringIO, iterkeys

import pyutilib.th as unittest

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Expression, Block,
    TransformationFactory, Piecewise, Objective, ExternalFunction,
    Suffix, value,
)
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
    generate_finite_elements, generate_colloc_points,
    update_contset_indexed_component, expand_components,
    get_index_information,
)

currdir = dirname(abspath(__file__)) + os.sep


class TestDaeMisc(unittest.TestCase):
    
    # test generate_finite_elements method
    def test_generate_finite_elements(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.t2 = ContinuousSet(bounds=(0, 10))
        m.t3 = ContinuousSet(bounds=(0, 1))

        oldt = sorted(m.t)
        generate_finite_elements(m.t, 1)
        self.assertTrue(oldt == sorted(m.t))
        self.assertFalse(m.t.get_changed())
        generate_finite_elements(m.t, 2)
        self.assertFalse(oldt == sorted(m.t))
        self.assertTrue(m.t.get_changed())
        self.assertTrue([0, 5.0, 10] == sorted(m.t))
        generate_finite_elements(m.t, 3)
        self.assertTrue([0, 2.5, 5.0, 10] == sorted(m.t))
        generate_finite_elements(m.t, 5)
        self.assertTrue([0, 1.25, 2.5, 5.0, 7.5, 10] == sorted(m.t))

        generate_finite_elements(m.t2, 10)
        self.assertTrue(len(m.t2) == 11)
        self.assertTrue([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] == sorted(m.t2))

        generate_finite_elements(m.t3, 7)
        self.assertTrue(len(m.t3) == 8)
        t = sorted(m.t3)
        print(t[1])
        self.assertTrue(t[1] == 0.142857)
      
    # test generate_collocation_points method
    def test_generate_collocation_points(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[0, 1])
        m.t2 = ContinuousSet(initialize=[0, 2, 4, 6])
        
        tau1 = [1]
        oldt = sorted(m.t)
        generate_colloc_points(m.t, tau1)
        self.assertTrue(oldt == sorted(m.t))
        self.assertFalse(m.t.get_changed())

        tau1 = [0.5]
        oldt = sorted(m.t)
        generate_colloc_points(m.t, tau1)
        self.assertFalse(oldt == sorted(m.t))
        self.assertTrue(m.t.get_changed())
        self.assertTrue([0, 0.5, 1] == sorted(m.t))

        tau2 = [0.2, 0.3, 0.7, 0.8, 1]
        generate_colloc_points(m.t, tau2)
        self.assertTrue(len(m.t) == 11)
        self.assertTrue(
            [0, 0.1, 0.15, 0.35, 0.4, 0.5, 0.6, 0.65, 0.85, 0.9, 1] == 
            sorted(m.t))

        generate_colloc_points(m.t2, tau2)
        self.assertTrue(len(m.t2) == 16)
        self.assertTrue(m.t2.get_changed())
        t = sorted(m.t2)
        self.assertTrue(t[1] == 0.4)
        self.assertTrue(t[13] == 5.4)

    # test Params indexed only by a ContinuousSet after discretizing
    def test_discretized_params_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=[(1, 1), (2, 2)])
        m.p1 = Param(m.t, initialize=1)
        m.p2 = Param(m.t, default=2)
        m.p3 = Param(m.t, initialize=1, default=2)
        
        def _rule1(m, i):
            return i**2
        
        def _rule2(m, i):
            return 2 * i
        m.p4 = Param(m.t, initialize={0: 5, 10: 5}, default=_rule1)
        m.p5 = Param(m.t, initialize=_rule1, default=_rule2)

        generate_finite_elements(m.t, 5)
        # Expected ValueError because no default value was specified
        with self.assertRaises(ValueError):
            for i in m.t:
                m.p1[i]

        for i in m.t:
            self.assertEqual(m.p2[i], 2)

            if i == 0 or i == 10:
                self.assertEqual(m.p3[i], 1)
                self.assertEqual(m.p4[i], 5)
                self.assertEqual(m.p5[i], i**2)
            else:
                self.assertEqual(m.p3[i], 2)
                self.assertEqual(m.p4[i], i**2)
                self.assertEqual(m.p5[i], 2 * i)

    # test Params with multiple indexing sets after discretizing
    def test_discretized_params_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=[(1, 1), (2, 2)])

        def _rule1(m, i):
            return i**2
        m.p1 = Param(m.s1, m.t, initialize=2, default=_rule1)
        m.p2 = Param(m.t, m.s1, default=5)

        def _rule2(m, i, j):
            return i + j
        m.p3 = Param(m.s1, m.t, initialize=2, default=_rule2)

        def _rule3(m, i, j, k):
            return i + j + k
        m.p4 = Param(m.s2, m.t, default=_rule3)

        generate_finite_elements(m.t, 5)

        # Expected TypeError because a function with the wrong number of
        # arguments was specified as the default

        with self.assertRaises(TypeError):
            for i in m.p1:
                m.p1[i]

        for i in m.p2:
            self.assertEqual(m.p2[i], 5)

        for i in m.t:
            for j in m.s1:
                if i == 0 or i == 10:
                    self.assertEqual(m.p3[j, i], 2)
                else:
                    self.assertEqual(m.p3[j, i], i + j)

        for i in m.t:
            for j in m.s2:
                self.assertEqual(m.p4[j, i], sum(j, i))

    # test update_contset_indexed_component method for Vars with 
    # single index of the ContinuousSet
    def test_update_contset_indexed_component_vars_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.t2 = ContinuousSet(initialize=[1, 2, 3])
        m.s = Set(initialize=[1, 2, 3])
        m.v1 = Var(m.t, initialize=3)
        m.v2 = Var(m.t, bounds=(4, 10), initialize={0: 2, 10: 12})

        def _init(m, i):
            return i
        m.v3 = Var(m.t, bounds=(-5, 5), initialize=_init)
        m.v4 = Var(m.s, initialize=7, dense=True)
        m.v5 = Var(m.t2, dense=True)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.v1, expansion_map)
        update_contset_indexed_component(m.v2, expansion_map)
        update_contset_indexed_component(m.v3, expansion_map)
        update_contset_indexed_component(m.v4, expansion_map)
        update_contset_indexed_component(m.v5, expansion_map)

        self.assertTrue(len(m.v1) == 6)
        self.assertTrue(len(m.v2) == 6)
        self.assertTrue(len(m.v3) == 6)
        self.assertTrue(len(m.v4) == 3)
        self.assertTrue(len(m.v5) == 3)

        self.assertTrue(value(m.v1[2]) == 3)
        self.assertTrue(m.v1[4].ub is None)
        self.assertTrue(m.v1[6].lb is None)
        
        self.assertTrue(m.v2[2].value is None)
        self.assertTrue(m.v2[4].lb == 4)
        self.assertTrue(m.v2[8].ub == 10)
        self.assertTrue(value(m.v2[0]) == 2)

        self.assertTrue(value(m.v3[2]) == 2)
        self.assertTrue(m.v3[4].lb == -5)
        self.assertTrue(m.v3[6].ub == 5)
        self.assertTrue(value(m.v3[8]) == 8)

    # test update_contset_indexed_component method for Vars with 
    # multiple indices
    def test_update_contset_indexed_component_vars_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.t2 = ContinuousSet(initialize=[1, 2, 3])
        m.s = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=[(1, 1), (2, 2)])
        m.v1 = Var(m.s, m.t, initialize=3)
        m.v2 = Var(m.s, m.t, m.t2, bounds=(4, 10),
                   initialize={(1, 0, 1): 22, (2, 10, 2): 22})

        def _init(m, i, j, k):
            return i
        m.v3 = Var(m.t, m.s2, bounds=(-5, 5), initialize=_init)
        m.v4 = Var(m.s, m.t2, initialize=7, dense=True)
        m.v5 = Var(m.s2)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.v1, expansion_map)
        update_contset_indexed_component(m.v2, expansion_map)
        update_contset_indexed_component(m.v3, expansion_map)
        update_contset_indexed_component(m.v4, expansion_map)
        update_contset_indexed_component(m.v5, expansion_map)

        self.assertTrue(len(m.v1) == 18)
        self.assertTrue(len(m.v2) == 54)
        self.assertTrue(len(m.v3) == 12)
        self.assertTrue(len(m.v4) == 9)

        self.assertTrue(value(m.v1[1, 4]) == 3)
        self.assertTrue(m.v1[2, 2].ub is None)
        self.assertTrue(m.v1[3, 8].lb is None)
        
        self.assertTrue(value(m.v2[1, 0, 1]) == 22)
        self.assertTrue(m.v2[1, 2, 1].value is None)
        self.assertTrue(m.v2[2, 4, 3].lb == 4)
        self.assertTrue(m.v2[3, 8, 1].ub == 10)

        self.assertTrue(value(m.v3[2, 2, 2]) == 2)
        self.assertTrue(m.v3[4, 1, 1].lb == -5)
        self.assertTrue(m.v3[8, 2, 2].ub == 5)
        self.assertTrue(value(m.v3[6, 1, 1]) == 6)

    # test update_contset_indexed_component method for Constraints with
    # single index of the ContinuousSet
    def test_update_contset_indexed_component_constraints_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.p = Param(m.t, default=3)
        m.v = Var(m.t, initialize=5)
        
        def _con1(m, i):
            return m.p[i] * m.v[i] <= 20
        m.con1 = Constraint(m.t, rule=_con1)
        
        # Rules that iterate over a ContinuouSet implicitly are not updated
        # after the discretization
        def _con2(m):
            return sum(m.v[i] for i in m.t) >= 0
        m.con2 = Constraint(rule=_con2)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.v, expansion_map)
        update_contset_indexed_component(m.p, expansion_map)
        update_contset_indexed_component(m.con1, expansion_map)
        update_contset_indexed_component(m.con2, expansion_map)

        self.assertTrue(len(m.con1) == 6)
        self.assertEqual(m.con1[2](), 15)
        self.assertEqual(m.con1[8](), 15)
        self.assertEqual(m.con2(), 10)

    # test update_contset_indexed_component method for Constraints with
    # multiple indices
    def test_update_contset_indexed_component_constraints_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.t2 = ContinuousSet(initialize=[1, 2, 3])
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=[(1, 1), (2, 2)])

        def _init(m, i, j):
            return j + i
        m.p1 = Param(m.s1, m.t, default=_init)
        m.v1 = Var(m.s1, m.t, initialize=5)
        m.v2 = Var(m.s2, m.t, initialize=2)
        m.v3 = Var(m.t2, m.s2, initialize=1)

        def _con1(m, si, ti):
            return m.v1[si, ti] * m.p1[si, ti] >= 0
        m.con1 = Constraint(m.s1, m.t, rule=_con1)

        def _con2(m, i, j, ti):
            return m.v2[i, j, ti] + m.p1[1, ti] == 10
        m.con2 = Constraint(m.s2, m.t, rule=_con2)

        def _con3(m, i, ti, ti2, j, k):
            return m.v1[i, ti] - m.v3[ti2, j, k] * m.p1[i, ti] <= 20
        m.con3 = Constraint(m.s1, m.t, m.t2, m.s2, rule=_con3)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.p1, expansion_map)
        update_contset_indexed_component(m.v1, expansion_map)
        update_contset_indexed_component(m.v2, expansion_map)
        update_contset_indexed_component(m.v3, expansion_map)
        update_contset_indexed_component(m.con1, expansion_map)
        update_contset_indexed_component(m.con2, expansion_map)
        update_contset_indexed_component(m.con3, expansion_map)
        
        self.assertTrue(len(m.con1) == 18)
        self.assertTrue(len(m.con2) == 12)
        self.assertTrue(len(m.con3) == 108)

        self.assertEqual(m.con1[1, 4](), 25)
        self.assertEqual(m.con1[2, 6](), 40)
        self.assertEqual(m.con1[3, 8](), 55)
        self.assertTrue(value(m.con1[2, 4].lower) == 0)
        self.assertTrue(value(m.con1[1, 8].upper) is None)

        self.assertEqual(m.con2[1, 1, 2](), 5)
        self.assertEqual(m.con2[2, 2, 4](), 7)
        self.assertEqual(m.con2[1, 1, 8](), 11)
        self.assertTrue(value(m.con2[2, 2, 6].lower) == 10)
        self.assertTrue(value(m.con2[1, 1, 10].upper) == 10)
        
        self.assertEqual(m.con3[1, 2, 1, 1, 1](), 2)
        self.assertEqual(m.con3[1, 4, 1, 2, 2](), 0)
        self.assertEqual(m.con3[2, 6, 3, 1, 1](), -3)
        self.assertEqual(m.con3[3, 8, 2, 2, 2](), -6)
        self.assertTrue(value(m.con3[2, 0, 2, 1, 1].lower) is None)
        self.assertTrue(value(m.con3[3, 2, 3, 2, 2].upper) == 20)

    # test update_contset_indexed_component method for Expression with
    # single index of the ContinuouSet
    def test_update_contset_indexed_component_expressions_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.p = Param(m.t, default=3)
        m.v = Var(m.t, initialize=5)

        def _con1(m, i):
            return m.p[i] * m.v[i]
        m.con1 = Expression(m.t, rule=_con1)

        # Rules that iterate over a ContinuousSet implicitly are not updated
        # after the discretization
        def _con2(m):
            return sum(m.v[i] for i in m.t)
        m.con2 = Expression(rule=_con2)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.v, expansion_map)
        update_contset_indexed_component(m.p, expansion_map)
        update_contset_indexed_component(m.con1, expansion_map)
        update_contset_indexed_component(m.con2, expansion_map)

        self.assertTrue(len(m.con1) == 6)
        self.assertEqual(m.con1[2](), 15)
        self.assertEqual(m.con1[8](), 15)
        self.assertEqual(m.con2(), 10)

    # test update_contset_indexed_component method for Expressions with
    # multiple indices
    def test_update_contset_indexed_component_expressions_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.t2 = ContinuousSet(initialize=[1, 2, 3])
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=[(1, 1), (2, 2)])

        def _init(m, i, j):
            return j + i
        m.p1 = Param(m.s1, m.t, default=_init)
        m.v1 = Var(m.s1, m.t, initialize=5)
        m.v2 = Var(m.s2, m.t, initialize=2)
        m.v3 = Var(m.t2, m.s2, initialize=1)

        def _con1(m, si, ti):
            return m.v1[si, ti] * m.p1[si, ti]
        m.con1 = Expression(m.s1, m.t, rule=_con1)

        def _con2(m, i, j, ti):
            return m.v2[i, j, ti] + m.p1[1, ti]
        m.con2 = Expression(m.s2, m.t, rule=_con2)

        def _con3(m, i, ti, ti2, j, k):
            return m.v1[i, ti] - m.v3[ti2, j, k] * m.p1[i, ti]
        m.con3 = Expression(m.s1, m.t, m.t2, m.s2, rule=_con3)

        expansion_map = ComponentMap()

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.p1, expansion_map)
        update_contset_indexed_component(m.v1, expansion_map)
        update_contset_indexed_component(m.v2, expansion_map)
        update_contset_indexed_component(m.v3, expansion_map)
        update_contset_indexed_component(m.con1, expansion_map)
        update_contset_indexed_component(m.con2, expansion_map)
        update_contset_indexed_component(m.con3, expansion_map)

        self.assertTrue(len(m.con1) == 18)
        self.assertTrue(len(m.con2) == 12)
        self.assertTrue(len(m.con3) == 108)

        self.assertEqual(m.con1[1, 4](), 25)
        self.assertEqual(m.con1[2, 6](), 40)
        self.assertEqual(m.con1[3, 8](), 55)

        self.assertEqual(m.con2[1, 1, 2](), 5)
        self.assertEqual(m.con2[2, 2, 4](), 7)
        self.assertEqual(m.con2[1, 1, 8](), 11)

        self.assertEqual(m.con3[1, 2, 1, 1, 1](), 2)
        self.assertEqual(m.con3[1, 4, 1, 2, 2](), 0)
        self.assertEqual(m.con3[2, 6, 3, 1, 1](), -3)
        self.assertEqual(m.con3[3, 8, 2, 2, 2](), -6)

    # test update_contset_indexed_component method for Blocks 
    # indexed by a ContinuousSet
    def test_update_contset_indexed_component_block_single(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))

        def _block_rule(b, t):
            m = b.model()
    
            b.s1 = Set(initialize=['A1', 'A2', 'A3'])

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
            b.v2 = Var(m.t, initialize=2)
            b.v3 = Var(m.t, b.s1, initialize=1)

            def _con1(_b, ti):
                return _b.v1[ti] * _b.p1[ti] == _b.v1[t]**2
            b.con1 = Constraint(m.t, rule=_con1)

            def _con2(_b, i, ti):
                return _b.v2[ti] - _b.v3[ti, i] + _b.p1[ti]
            b.con2 = Expression(b.s1, m.t, rule=_con2)
    
        model.blk = Block(model.t, rule=_block_rule)
     
        self.assertTrue(len(model.blk), 2)

        expansion_map = ComponentMap()

        generate_finite_elements(model.t, 5)

        missing_idx = set(model.blk._index) - set(iterkeys(model.blk._data))
        model.blk._dae_missing_idx = missing_idx

        update_contset_indexed_component(model.blk, expansion_map)

        self.assertEqual(len(model.blk), 6)
        self.assertEqual(len(model.blk[10].con1), 2)
        self.assertEqual(len(model.blk[2].con1), 6)
        self.assertEqual(len(model.blk[10].v2), 2)

        self.assertEqual(model.blk[2].p1[2], 4)
        self.assertEqual(model.blk[8].p1[6], 12)
        
        self.assertEqual(model.blk[4].con1[4](), 15)
        self.assertEqual(model.blk[6].con1[8](), 55)

        self.assertEqual(model.blk[0].con2['A1', 10](), 21)
        self.assertEqual(model.blk[4].con2['A2', 6](), 13)

    # test update_contset_indexed_component method for Blocks with
    # multiple indices
    def test_update_contset_indexed_component_block_multiple(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))
        model.s1 = Set(initialize=['A', 'B', 'C'])
        model.s2 = Set(initialize=[('x1', 'x1'), ('x2', 'x2')])

        def _block_rule(b, t, s1):
            m = b.model()
            
            def _init(m, i, j):
                return j * 2
            b.p1 = Param(m.s1, m.t, mutable=True, default=_init)
            b.v1 = Var(m.s1, m.t, initialize=5)
            b.v2 = Var(m.s2, m.t, initialize=2)
            b.v3 = Var(m.t, m.s2, initialize=1)

            def _con1(_b, si, ti):
                return _b.v1[si, ti] * _b.p1[si, ti] == _b.v1[si, t]**2
            b.con1 = Constraint(m.s1, m.t, rule=_con1)

            def _con2(_b, i, j, ti):
                return _b.v2[i, j, ti] - _b.v3[ti, i, j] + _b.p1['A', ti]
            b.con2 = Expression(m.s2, m.t, rule=_con2)
    
        model.blk = Block(model.t, model.s1, rule=_block_rule)

        expansion_map = ComponentMap()

        self.assertTrue(len(model.blk), 6)

        generate_finite_elements(model.t, 5)

        missing_idx = set(model.blk._index) - set(iterkeys(model.blk._data))
        model.blk._dae_missing_idx = missing_idx

        update_contset_indexed_component(model.blk, expansion_map)

        self.assertEqual(len(model.blk), 18)
        self.assertEqual(len(model.blk[10, 'C'].con1), 6)
        self.assertEqual(len(model.blk[2, 'B'].con1), 18)
        self.assertEqual(len(model.blk[10, 'C'].v2), 4)

        self.assertEqual(model.blk[2, 'A'].p1['A', 2], 4)
        self.assertEqual(model.blk[8, 'C'].p1['B', 6], 12)
        
        self.assertEqual(model.blk[4, 'B'].con1['B', 4](), 15)
        self.assertEqual(model.blk[6, 'A'].con1['C', 8](), 55)

        self.assertEqual(model.blk[0, 'A'].con2['x1', 'x1', 10](), 21)
        self.assertEqual(model.blk[4, 'C'].con2['x2', 'x2', 6](), 13)

    # test update_contset_indexed_component method for Blocks 
    # indexed by a ContinuousSet. Block rule returns new block
    def test_update_contset_indexed_component_block_single2(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))

        def _block_rule(_b_, t):
            m = _b_.model()
            b = Block()

            b.s1 = Set(initialize=['A1', 'A2', 'A3'])

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
            b.v2 = Var(m.t, initialize=2)
            b.v3 = Var(m.t, b.s1, initialize=1)

            def _con1(_b, ti):
                return _b.v1[ti] * _b.p1[ti] == _b.v1[t]**2
            b.con1 = Constraint(m.t, rule=_con1)

            def _con2(_b, i, ti):
                return _b.v2[ti] - _b.v3[ti, i] + _b.p1[ti]
            b.con2 = Expression(b.s1, m.t, rule=_con2)
            return b
    
        model.blk = Block(model.t, rule=_block_rule)

        expansion_map = ComponentMap()
     
        self.assertTrue(len(model.blk), 2)

        generate_finite_elements(model.t, 5)

        missing_idx = set(model.blk._index) - set(iterkeys(model.blk._data))
        model.blk._dae_missing_idx = missing_idx

        update_contset_indexed_component(model.blk, expansion_map)

        self.assertEqual(len(model.blk), 6)
        self.assertEqual(len(model.blk[10].con1), 2)
        self.assertEqual(len(model.blk[2].con1), 6)
        self.assertEqual(len(model.blk[10].v2), 2)

        self.assertEqual(model.blk[2].p1[2], 4)
        self.assertEqual(model.blk[8].p1[6], 12)
        
        self.assertEqual(model.blk[4].con1[4](), 15)
        self.assertEqual(model.blk[6].con1[8](), 55)

        self.assertEqual(model.blk[0].con2['A1', 10](), 21)
        self.assertEqual(model.blk[4].con2['A2', 6](), 13)

    # test update_contset_indexed_component method for Blocks with
    # multiple indices. Block rule returns new block
    def test_update_contset_indexed_component_block_multiple2(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))
        model.s1 = Set(initialize=['A', 'B', 'C'])
        model.s2 = Set(initialize=[('x1', 'x1'), ('x2', 'x2')])

        def _block_rule(_b_, t, s1):
            m = _b_.model()
            b = Block()

            def _init(m, i, j):
                return j * 2
            b.p1 = Param(m.s1, m.t, mutable=True, default=_init)
            b.v1 = Var(m.s1, m.t, initialize=5)
            b.v2 = Var(m.s2, m.t, initialize=2)
            b.v3 = Var(m.t, m.s2, initialize=1)

            def _con1(_b, si, ti):
                return _b.v1[si, ti] * _b.p1[si, ti] == _b.v1[si, t]**2
            b.con1 = Constraint(m.s1, m.t, rule=_con1)

            def _con2(_b, i, j, ti):
                return _b.v2[i, j, ti] - _b.v3[ti, i, j] + _b.p1['A', ti]
            b.con2 = Expression(m.s2, m.t, rule=_con2)
            return b
    
        model.blk = Block(model.t, model.s1, rule=_block_rule)

        expansion_map = ComponentMap()

        self.assertTrue(len(model.blk), 6)

        generate_finite_elements(model.t, 5)

        missing_idx = set(model.blk._index) - set(iterkeys(model.blk._data))
        model.blk._dae_missing_idx = missing_idx

        update_contset_indexed_component(model.blk, expansion_map)

        self.assertEqual(len(model.blk), 18)
        self.assertEqual(len(model.blk[10, 'C'].con1), 6)
        self.assertEqual(len(model.blk[2, 'B'].con1), 18)
        self.assertEqual(len(model.blk[10, 'C'].v2), 4)

        self.assertEqual(model.blk[2, 'A'].p1['A', 2], 4)
        self.assertEqual(model.blk[8, 'C'].p1['B', 6], 12)
        
        self.assertEqual(model.blk[4, 'B'].con1['B', 4](), 15)
        self.assertEqual(model.blk[6, 'A'].con1['C', 8](), 55)

        self.assertEqual(model.blk[0, 'A'].con2['x1', 'x1', 10](), 21)
        self.assertEqual(model.blk[4, 'C'].con2['x2', 'x2', 6](), 13)

    # test update_contset_indexed_component method for Piecewise
    # component indexed by a ContinuousSet
    def test_update_contset_indexed_component_piecewise_single(self):
        x = [0.0, 1.5, 3.0, 5.0]
        y = [1.1, -1.1, 2.0, 1.1]
        
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))

        model.x = Var(model.t, bounds=(min(x), max(x)))
        model.y = Var(model.t)

        model.fx = Piecewise(model.t, 
                             model.y, model.x, 
                             pw_pts=x, 
                             pw_constr_type='EQ', 
                             f_rule=y)
        
        self.assertEqual(len(model.fx), 2)

        expansion_map = ComponentMap()

        generate_finite_elements(model.t, 5)
        update_contset_indexed_component(model.fx, expansion_map)

        self.assertEqual(len(model.fx), 6)
        self.assertEqual(len(model.fx[2].SOS2_constraint), 3)

    # test update_contset_indexed_component method for Piecewise 
    # component with multiple indices
    def test_update_contset_indexed_component_piecewise_multiple(self):
        x = [0.0, 1.5, 3.0, 5.0]
        y = [1.1, -1.1, 2.0, 1.1]
        
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 10))
        model.s = Set(initialize=['A', 'B', 'C'])

        model.x = Var(model.s, model.t, bounds=(min(x), max(x)))
        model.y = Var(model.s, model.t)

        model.fx = Piecewise(model.s, model.t, 
                             model.y, model.x, 
                             pw_pts=x, 
                             pw_constr_type='EQ', 
                             f_rule=y)
        
        self.assertEqual(len(model.fx), 6)

        expansion_map = ComponentMap()

        generate_finite_elements(model.t, 5)
        update_contset_indexed_component(model.fx, expansion_map)

        self.assertEqual(len(model.fx), 18)
        self.assertEqual(len(model.fx['A', 2].SOS2_constraint), 3)

    # test update_contset_indexed_component method for other components
    def test_update_contset_indexed_component_other(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.junk = Suffix()
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.s)

        def _obj(m):
            return sum(m.v[i] for i in m.s)
        m.obj = Objective(rule=_obj)

        expansion_map = ComponentMap

        generate_finite_elements(m.t, 5)
        update_contset_indexed_component(m.junk, expansion_map)
        update_contset_indexed_component(m.s, expansion_map)
        update_contset_indexed_component(m.obj, expansion_map)

    # test unsupported components indexed by a single ContinuousSet
    def test_update_contset_indexed_component_unsupported_single(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s = Set(m.t)
        generate_finite_elements(m.t, 5)
        expansion_map = ComponentMap()

        # Expected TypeError because Set is not a component that supports
        # indexing by a ContinuousSet
        with self.assertRaises(TypeError):
            update_contset_indexed_component(m.s, expansion_map)

    # test unsupported components indexed by multiple sets
    def test_update_contset_indexed_component_unsupported_multiple(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.i = Set(initialize=[1, 2, 3])
        m.s = Set(m.i, m.t)
        generate_finite_elements(m.t, 5)
        expansion_map = ComponentMap()

        # Expected TypeError because Set is not a component that supports
        # indexing by a ContinuousSet
        with self.assertRaises(TypeError):
            update_contset_indexed_component(m.s, expansion_map)

    def test_update_block_derived(self):
        class Foo(Block):
            pass

        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))

        def _block_rule(b, t):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, rule=_block_rule)

        generate_finite_elements(m.t, 5)
        expand_components(m)

        self.assertEqual(len(m.foo), 6)
        self.assertEqual(len(m.foo[0].p1), 6)
        self.assertEqual(len(m.foo[2].v1), 6)
        self.assertEqual(m.foo[0].p1[6], 12)

    def test_update_block_derived_override_construct_nofcn(self):
        class Foo(Block):

            def construct(self, data=None):
                Block.construct(self, data)

        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))

        def _block_rule(b, t):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, rule=_block_rule)
        generate_finite_elements(m.t, 5)

        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.dae'):
            expand_components(m)
        self.assertIn('transformation to the Block-derived component', 
                      OUTPUT.getvalue())
        self.assertEqual(len(m.foo), 6)
        self.assertEqual(len(m.foo[0].p1), 6)
        self.assertEqual(len(m.foo[2].v1), 6)
        self.assertEqual(m.foo[0].p1[6], 12)

    def test_update_block_derived_override_construct_withfcn(self):
        class Foo(Block):
            updated = False

            def construct(self, data=None):
                Block.construct(self, data)
            
            def update_after_discretization(self):
                self.updated = True
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))

        def _block_rule(b, t):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, rule=_block_rule)

        generate_finite_elements(m.t, 5)
        expand_components(m)

        self.assertTrue(m.foo.updated)
        self.assertEqual(len(m.foo), 2)
        self.assertEqual(len(m.foo[0].v1), 6)

    def test_update_block_derived2(self):
        class Foo(Block):
            pass

        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s = Set(initialize=[1, 2, 3])

        def _block_rule(b, t, s):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, m.s, rule=_block_rule)

        generate_finite_elements(m.t, 5)
        expand_components(m)

        self.assertEqual(len(m.foo), 18)
        self.assertEqual(len(m.foo[0, 1].p1), 6)
        self.assertEqual(len(m.foo[2, 2].v1), 6)
        self.assertEqual(m.foo[0, 3].p1[6], 12)

    def test_update_block_derived_override_construct_nofcn2(self):
        class Foo(Block):

            def construct(self, data=None):
                Block.construct(self, data)

        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s = Set(initialize=[1, 2, 3])

        def _block_rule(b, t, s):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, m.s, rule=_block_rule)

        generate_finite_elements(m.t, 5)

        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.dae'):
            expand_components(m)
        self.assertIn('transformation to the Block-derived component', 
                      OUTPUT.getvalue())
        self.assertEqual(len(m.foo), 18)
        self.assertEqual(len(m.foo[0, 1].p1), 6)
        self.assertEqual(len(m.foo[2, 2].v1), 6)
        self.assertEqual(m.foo[0, 3].p1[6], 12)

    def test_update_block_derived_override_construct_withfcn2(self):
        class Foo(Block):
            updated = False

            def construct(self, data=None):
                Block.construct(self, data)
            
            def update_after_discretization(self):
                self.updated = True
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s = Set(initialize=[1, 2, 3])

        def _block_rule(b, t, s):
            m = b.model()

            def _init(m, j):
                return j * 2
            b.p1 = Param(m.t, default=_init)
            b.v1 = Var(m.t, initialize=5)
        m.foo = Foo(m.t, m.s, rule=_block_rule)

        generate_finite_elements(m.t, 5)
        expand_components(m)

        self.assertTrue(m.foo.updated)
        self.assertEqual(len(m.foo), 6)
        self.assertEqual(len(m.foo[0, 1].v1), 6)

    def test_hierarchical_blocks(self):
        m = ConcreteModel()

        m.b = Block()
        m.b.t = ContinuousSet(bounds=(0, 10))

        m.b.c = Block()

        def _d_rule(d, t):
            m = d.model()
            d.x = Var()
            return d

        m.b.c.d = Block(m.b.t, rule=_d_rule)

        m.b.y = Var(m.b.t)

        def _con_rule(b, t):
            return b.y[t] <= b.c.d[t].x

        m.b.con = Constraint(m.b.t, rule=_con_rule)

        generate_finite_elements(m.b.t, 5)
        expand_components(m)

        self.assertEqual(len(m.b.c.d), 6)
        self.assertEqual(len(m.b.con), 6)
        self.assertEqual(len(m.b.y), 6)

    def test_external_function(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        
        def _fun(x):
            return x**2
        m.x_func = ExternalFunction(_fun)

        m.y = Var(m.t, initialize=3)
        m.dy = DerivativeVar(m.y, initialize=3)
        
        def _con(m, t):
            return m.dy[t] == m.x_func(m.y[t])
        m.con = Constraint(m.t, rule=_con)
        
        generate_finite_elements(m.t, 5)
        expand_components(m)

        self.assertEqual(len(m.y), 6)
        self.assertEqual(len(m.con), 6)

    def test_get_index_information(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.x = ContinuousSet(bounds=(0,10))
        m.s = Set(initialize=['a','b','c'])
        m.v = Var(m.t, m.x, m.s, initialize=1)
        m.v2 = Var(m.t, m.s, initialize=1)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, wrt=m.t, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')
        disc.apply_to(m, wrt=m.x, nfe=5, ncp=2, scheme='LAGRANGE-RADAU')

        info = get_index_information(m.v, m.t)
        nts = info['non_ds']
        index_getter = info['index function']
        
        self.assertEqual(len(nts), 33)
        self.assertTrue(m.x in nts.set_tuple)
        self.assertTrue(m.s in nts.set_tuple)
        self.assertEqual(index_getter((8.0,'a'),1,0),(2.0,8.0,'a'))

        info = get_index_information(m.v2, m.t)
        nts = info['non_ds']
        index_getter = info['index function']
        
        self.assertEqual(len(nts), 3)
        self.assertTrue(m.s is nts)
        self.assertEqual(index_getter('a',1,0),(2.0,'a'))



if __name__ == "__main__":
    unittest.main()
