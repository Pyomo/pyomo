#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#

import pyutilib.th as unittest

from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, Set, value, Constraint) 
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
    Simulator, 
    _check_getitemexpression, 
    _check_productexpression,
    _check_sumexpression)
from pyomo.core.base import expr as EXPR
from pyomo.core.base.template_expr import (
    IndexTemplate, 
    _GetItemIndexer,
    substitute_template_expression, 
    substitute_getitem_with_param,
    substitute_template_with_value,
)

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)
        m.s = Set(initialize=[1,2,3])

    def test_sim_initialization_single_index(self):
        m = self.m
        m.w = Var(m.t)
        m.dw = DerivativeVar(m.w)

        t = IndexTemplate(m.t)
        
        def _deq1(m,i):
            return m.dv[i] == m.v[i]
        m.deq1 = Constraint(m.t, rule=_deq1)

        def _deq2(m, i):
            return m.dw[i] == m.v[i]
        m.deq2 = Constraint(m.t, rule=_deq2)

        mysim = Simulator(m)

        self.assertIs(mysim._contset, m.t)
        self.assertEqual(len(mysim._diffvars), 2)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t]))
        self.assertEqual(len(mysim._derivlist), 2)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t]))
        self.assertEqual(len(mysim._templatemap), 1)
        self.assertTrue(_GetItemIndexer(m.v[t]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w[t]) in mysim._templatemap)
        self.assertEqual(len(mysim._rhsdict), 2)
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dv[t])], Param))
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dv[t])].name, 'v[{t}]')
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw[t])], Param))
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw[t])].name, 'v[{t}]')
        self.assertEqual(len(mysim._rhsfun([0,0],0)), 2)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)
        m.del_component('deq1')
        m.del_component('deq2')
        m.del_component('dw')
        m.del_component('w')

    def test_sim_initialization_multi_index(self):
        m = self.m
        m.w1 = Var(m.t, m.s)
        m.dw1 = DerivativeVar(m.w1)

        m.w2 = Var(m.s, m.t)
        m.dw2 = DerivativeVar(m.w2)

        m.w3 = Var([0,1], m.t, m.s)
        m.dw3 = DerivativeVar(m.w3)

        t = IndexTemplate(m.t)
        
        def _deq1(m, t, s):
            return m.dw1[t,s] == m.w1[t,s]
        m.deq1 = Constraint(m.t, m.s, rule=_deq1)

        def _deq2(m, s, t):
            return m.dw2[s,t] == m.w2[s,t]
        m.deq2 = Constraint(m.s, m.t, rule=_deq2)

        def _deq3(m, i, t, s):
            return m.dw3[i,t,s] == m.w1[t,s] + m.w2[i+1,t]
        m.deq3 = Constraint([0,1],m.t,m.s,rule=_deq3)

        mysim = Simulator(m)

        self.assertIs(mysim._contset, m.t)
        self.assertEqual(len(mysim._diffvars), 12)
        self.assertTrue(_GetItemIndexer(m.w1[t,1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w1[t,3]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[1,t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[3,t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[0,t,1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[1,t,3]) in mysim._diffvars)

        self.assertEqual(len(mysim._derivlist), 12)
        self.assertTrue(_GetItemIndexer(m.dw1[t,1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw1[t,3]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[1,t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[3,t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[0,t,1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[1,t,3]) in mysim._derivlist)

        self.assertEqual(len(mysim._templatemap), 6)
        self.assertTrue(_GetItemIndexer(m.w1[t,1]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w1[t,3]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[1,t]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[3,t]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[0,t,1]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[1,t,3]) in mysim._templatemap)

        self.assertEqual(len(mysim._rhsdict), 12)
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t,1])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t,3])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[1,t])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[3,t])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[0,t,1])],
                                   EXPR._SumExpression))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[1,t,3])],
                                   EXPR._SumExpression))
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t,1])].name, 'w1[{t},1]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t,3])].name, 'w1[{t},3]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[1,t])].name, 'w2[1,{t}]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[3,t])].name, 'w2[3,{t}]')

        self.assertEqual(len(mysim._rhsfun([0]*12,0)), 12)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)

        m.del_component('deq1')
        m.del_component('deq1_index')
        m.del_component('deq2')
        m.del_component('deq2_index')
        m.del_component('deq3')
        m.del_component('deq3_index')
        
    def test_non_supported_single_index(self):
        
        # Only Scipy is supported
        m = self.m
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m, package='casadi')

        # Can't simulate a model with no ContinuousSet 
        m = ConcreteModel()
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)

        # Can't simulate a model with multiple ContinuousSets
        m = ConcreteModel()
        m.s = ContinuousSet(bounds=(0,10))
        m.t = ContinuousSet(bounds=(0,5))
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        
        # Can't simulate a model with no Derivatives
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)

        # Can't simulate a model with multiple RHS for a derivative 
        m = self.m
        def _diffeq(m, t):
            return m.dv[t] == m.v[t]**2 + m.v[t]
        m.con1 = Constraint(m.t, rule=_diffeq)
        m.con2 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        
        # Can't simulate a model with multiple derivatives in an
        # equation
        m = self.m
        def _diffeq(m, t):
            return m.dv[t] == m.dv[t] + m.v[t]**2
        m.con1 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')

        # Can't simulate a model with time indexed algebraic variables
        m = self.m
        m.a = Var(m.t)
        def _diffeq(m, t):
            return m.dv[t] == m.v[t]**2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con')

    def test_non_supported_multi_index(self):
        m = self.m
        m.v2 = Var(m.t,m.s)
        m.v3 = Var(m.s,m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.dv3 = DerivativeVar(m.v3)

        # Can't simulate a model with multiple RHS for a derivative 
        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.v2[t, s]**2 + m.v2[t, s]
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        m.con2 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.v3[s, t]**2 + m.v3[s, t]
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        m.con2 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        # Can't simulate a model with multiple derivatives in an
        # equation
        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.dv2[t,s] + m.v2[t,s]**2
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.dv3[s, t] + m.v3[s, t]**2
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

        # Can't simulate a model with time indexed algebraic variables
        m.a2 = Var(m.t, m.s)
        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.v2[t, s]**2 + m.a2[t, s]
        m.con = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con')
        m.del_component('con_index')

        m.a3 = Var(m.s, m.t)
        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.v3[s, t]**2 + m.a3[s, t]
        m.con = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)
        m.del_component('con')
        m.del_component('con_index')

class TestExpressionCheckers(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    def test_check_getitemexpression(self):
        m = self.m
        t = IndexTemplate(m.t)

        e = m.dv[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIs(e._args[0], temp[0])
        self.assertIs(e._args[1], temp[1])
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(m.v, temp[1]._base)
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)

        e = m.v[t] == m.dv[t]
        temp = _check_getitemexpression(e, 1)
        self.assertIs(e._args[0], temp[1])
        self.assertIs(e._args[1], temp[0])
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(m.v, temp[1]._base)
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)

        e = m.v[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)

    def test_check_productexpression(self):
        m = self.m 
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)

        # Check multiplication by constant
        e = 5*m.dv[t] == m.v[t]
        temp = _check_productexpression(e,0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)

        e = m.v[t] == 5*m.dv[t]
        temp = _check_productexpression(e,1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)

        # Check multiplication by fixed param
        e = m.p*m.dv[t] == m.v[t]
        temp = _check_productexpression(e,0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)

        e = m.v[t] == m.p*m.dv[t]
        temp = _check_productexpression(e,1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)

        # Check multiplication by mutable param
        e = m.mp*m.dv[t] == m.v[t]
        temp = _check_productexpression(e,0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.mp, temp[1]._denominator[0])

        e = m.v[t] == m.mp*m.dv[t]
        temp = _check_productexpression(e,1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.mp, temp[1]._denominator[0])

        # Check multiplication by var
        e = m.y*m.dv[t]/m.z == m.v[t]
        temp = _check_productexpression(e,0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.y, temp[1]._denominator[0])
        self.assertIs(m.z, temp[1]._numerator[1])

        e = m.v[t] == m.y*m.dv[t]/m.z
        temp = _check_productexpression(e,1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.y, temp[1]._denominator[0])
        self.assertIs(m.z, temp[1]._numerator[1])

        # Check having the DerivativeVar in the denominator
        e = m.y/(m.dv[t]*m.z) == m.mp
        temp = _check_productexpression(e,0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.mp, temp[1]._denominator[0])
        self.assertIs(m.y, temp[1]._numerator[0])
        self.assertIs(m.z, temp[1]._denominator[1])

        e = m.mp == m.y/(m.dv[t]*m.z)
        temp = _check_productexpression(e,1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertIs(m.mp, temp[1]._denominator[0])
        self.assertIs(m.y, temp[1]._numerator[0])
        self.assertIs(m.z, temp[1]._denominator[1])
        
        # Check expression with no DerivativeVar
        e = m.v[t]*m.y/m.z == m.v[t]*m.y/m.z
        temp = _check_productexpression(e,0)
        self.assertIsNone(temp)
        temp = _check_productexpression(e,1)
        self.assertIsNone(temp)
        
    def test_check_sumexpression(self):
        m = self.m 
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)

        e = m.dv[t] + m.y + m.z == m.v[t]
        temp = _check_sumexpression(e, 0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._SumExpression)
        self.assertIs(m.y, temp[1]._args[1])
        self.assertEqual(temp[1]._coef[1], -1)
        self.assertIs(m.z, temp[1]._args[2])
        self.assertEqual(temp[1]._coef[2], -1)

        e = m.v[t] == m.y + m.dv[t] + m.z
        temp = _check_sumexpression(e, 1)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._SumExpression)
        self.assertIs(m.y, temp[1]._args[1])
        self.assertEqual(temp[1]._coef[1], -1)
        self.assertIs(m.z, temp[1]._args[2])
        self.assertEqual(temp[1]._coef[2], -1)

        e = 5*m.dv[t] + 5*m.y - m.z == m.v[t]
        temp = _check_sumexpression(e, 0)
        self.assertIs(m.dv, temp[0]._base)
        self.assertIs(type(temp[1]), EXPR._ProductExpression)
        self.assertEqual(temp[1]._coef, 0.2)
        self.assertIs(m.y, temp[1]._numerator[0]._args[1])
        self.assertEqual(temp[1]._numerator[0]._coef[1], -5)
        self.assertIs(m.z, temp[1]._numerator[0]._args[2])
        self.assertEqual(temp[1]._numerator[0]._coef[2], 1)
