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
    substitute_template_expression, 
    substitute_template_with_param,
    substitute_template_with_index,
)

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    def test_sim_initialization(self):
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
        self.assertEqual(mysim._diffvars[0], m.v.name())
        self.assertEqual(mysim._diffvars[1], m.w.name())
        self.assertEqual(len(mysim._diffvarids), 2)
        self.assertEqual(len(mysim._derivlist), 2)
        self.assertEqual(mysim._derivlist[0], m.dv.name())
        self.assertEqual(mysim._derivlist[1], m.dw.name())
        self.assertEqual(len(mysim._templatemap), 1)
        self.assertTrue(id(m.v) in mysim._templatemap)
        self.assertFalse(id(m.w) in mysim._templatemap)
        self.assertEqual(len(mysim._rhsdict), 2)
        self.assertTrue(isinstance(mysim._rhsdict[m.dv.name()], Param))
        self.assertIs(mysim._rhsdict[m.dv.name()].name(), m.v.name())
        self.assertTrue(isinstance(mysim._rhsdict[m.dw.name()], Param))
        self.assertIs(mysim._rhsdict[m.dw.name()].name(), m.v.name())
        self.assertEqual(len(mysim._rhsfun([0,0],0)), 2)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)
        
    def test_non_supported(self):
        
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

        # Can't simulate a model with multiple derivatives in an
        # equation
        m = self.m
        def _diffeq(m, t):
            return m.dv[t] == m.dv[t] + m.v[t]**2
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)

        # Can't simulate a model with time indexed algebraic variables
        m = self.m
        m.a = Var(m.t)
        def _diffeq(m, t):
            return m.dv[t] == m.v[t]**2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            mysim = Simulator(m)

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
