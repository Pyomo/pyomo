#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import print_function
import pyutilib.th as unittest

from pyomo.core.expr import current as EXPR
from pyomo.environ import (
    ConcreteModel, Param, Var, Set, Constraint, 
    sin, log, sqrt, TransformationFactory)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
    is_pypy,
    scipy_available,
    casadi,
    casadi_available,
    Simulator, 
    _check_getitemexpression, 
    _check_productexpression,
    _check_negationexpression,
    _check_viewsumexpression, 
    substitute_pyomo2casadi,
)
from pyomo.core.expr.template_expr import (
    IndexTemplate, 
    _GetItemIndexer,
)

import os
from pyutilib.misc import setup_redirect, reset_redirect
from pyutilib.misc import import_file

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'dae'))

# We will skip tests unless we have scipy and not running in pypy
scipy_available = scipy_available and not is_pypy


class TestSimulator(unittest.TestCase):
    """
    Class for testing the pyomo.DAE simulator
    """

    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)
        m.s = Set(initialize=[1, 2, 3], ordered=True)

    # Testing invalid simulator arguments
    def test_invalid_argument_values(self):

        m = self.m
        m.w = Var(m.t)
        m.y = Var()

        with self.assertRaises(DAE_Error):
            Simulator(m, package='foo')

        def _con(m, i):
            return m.v[i] == m.w[i]**2 + m.y
        m.con = Constraint(m.t, rule=_con)

        with self.assertRaises(DAE_Error):
            Simulator(m, package='scipy')

        m.del_component('con')
        m.del_component('con_index')
        m.del_component('w')
        m.del_component('y')

    # Testing the simulator's handling of inequality constraints
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_inequality_constraints(self):

        m = self.m

        def _deq(m, i):
            return m.dv[i] >= m.v[i]**2 + m.v[i]
        m.deq = Constraint(m.t, rule=_deq)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 0)
        self.assertEqual(len(mysim._derivlist), 0)
        self.assertEqual(len(mysim._rhsdict), 0)

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case2(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == m.dw[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')
        m.del_component('w')
        m.del_component('dw')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case3(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.p * m.dv[i] == m.v[i]**2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.p * m.dw[i, j] == m.w[i, j]**2 + m.w[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

        def _deqv(m, i):
            return m.mp * m.dv[i] == m.v[i]**2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.y * m.dw[i, j] == m.w[i, j]**2 + m.w[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')
        m.del_component('w')
        m.del_component('dw')
        m.del_component('p')
        m.del_component('mp')
        m.del_component('y')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case4(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == m.p * m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == m.p * m.dw[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == m.mp * m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == m.y * m.dw[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')
        m.del_component('w')
        m.del_component('dw')
        m.del_component('p')
        m.del_component('mp')
        m.del_component('y')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case5(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.dv[i] + m.y == m.v[i]**2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.y + m.dw[i, j] == m.w[i, j]**2 + m.w[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

        def _deqv(m, i):
            return m.mp + m.dv[i] == m.v[i]**2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.dw[i, j] + m.p == m.w[i, j]**2 + m.w[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')
        m.del_component('w')
        m.del_component('dw')
        m.del_component('p')
        m.del_component('mp')
        m.del_component('y')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case6(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == m.dv[i] + m.y
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == m.y + m.dw[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == m.mp + m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == m.dw[i, j] + m.p
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')
        m.del_component('w')
        m.del_component('dw')
        m.del_component('p')
        m.del_component('mp')
        m.del_component('y')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case8(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return -m.dv[i] == m.v[i]**2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return -m.dw[i, j] == m.w[i, j]**2 + m.w[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

    # Testing various cases of separable differential equations to ensure
    # the simulator generates the correct RHS expression
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_separable_diffeq_case9(self):

        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i]**2 + m.v[i] == -m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j]**2 + m.w[i, j] == -m.dw[i, j]
        m.deqw = Constraint(m.t, m.s, rule=_deqw)

        mysim = Simulator(m)

        self.assertEqual(len(mysim._diffvars), 4)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
        self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
        self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
        self.assertEqual(len(mysim._derivlist), 4)
        self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
        self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
        self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
        self.assertEqual(len(mysim._rhsdict), 4)
        m.del_component('deqv')
        m.del_component('deqw')
        m.del_component('deqv_index')
        m.del_component('deqw_index')

    # Testing Simulator construction on differential variables with a
    # single index
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_sim_initialization_single_index(self):

        m = self.m
        m.w = Var(m.t)
        m.dw = DerivativeVar(m.w)

        t = IndexTemplate(m.t)
        
        def _deq1(m, i):
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
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dv[t])], Param))
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dv[t])].name, 'v[{t}]')
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw[t])], Param))
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dw[t])].name, 'v[{t}]')
        self.assertEqual(len(mysim._rhsfun(0, [0, 0])), 2)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)
        m.del_component('deq1')
        m.del_component('deq2')
        m.del_component('dw')
        m.del_component('w')

    # Testing Simulator construction on differential variables with
    # two indexing sets
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_sim_initialization_multi_index(self):

        m = self.m
        m.w1 = Var(m.t, m.s)
        m.dw1 = DerivativeVar(m.w1)

        m.w2 = Var(m.s, m.t)
        m.dw2 = DerivativeVar(m.w2)

        m.w3 = Var([0, 1], m.t, m.s)
        m.dw3 = DerivativeVar(m.w3)

        t = IndexTemplate(m.t)
        
        def _deq1(m, t, s):
            return m.dw1[t, s] == m.w1[t, s]
        m.deq1 = Constraint(m.t, m.s, rule=_deq1)

        def _deq2(m, s, t):
            return m.dw2[s, t] == m.w2[s, t]
        m.deq2 = Constraint(m.s, m.t, rule=_deq2)

        def _deq3(m, i, t, s):
            return m.dw3[i, t, s] == m.w1[t, s] + m.w2[i + 1, t]
        m.deq3 = Constraint([0, 1], m.t, m.s, rule=_deq3)

        mysim = Simulator(m)

        self.assertIs(mysim._contset, m.t)
        self.assertEqual(len(mysim._diffvars), 12)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w1[t, 3]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[1, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[3, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[0, t, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[1, t, 3]) in mysim._diffvars)

        self.assertEqual(len(mysim._derivlist), 12)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 3]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[1, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[3, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[0, t, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[1, t, 3]) in mysim._derivlist)

        self.assertEqual(len(mysim._templatemap), 6)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w1[t, 3]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[1, t]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[3, t]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[0, t, 1]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[1, t, 3]) in mysim._templatemap)

        self.assertEqual(len(mysim._rhsdict), 12)
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1])], Param))
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 3])], Param))
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[1, t])], Param))
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[3, t])], Param))
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[0, t, 1])],
                       EXPR.SumExpression))
        self.assertTrue(
            isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[1, t, 3])],
                       EXPR.SumExpression))
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1])].name, 'w1[{t},1]')
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dw1[t, 3])].name, 'w1[{t},3]')
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dw2[1, t])].name, 'w2[1,{t}]')
        self.assertEqual(
            mysim._rhsdict[_GetItemIndexer(m.dw2[3, t])].name, 'w2[3,{t}]')

        self.assertEqual(len(mysim._rhsfun(0, [0] * 12)), 12)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)

        m.del_component('deq1')
        m.del_component('deq1_index')
        m.del_component('deq2')
        m.del_component('deq2_index')
        m.del_component('deq3')
        m.del_component('deq3_index')

    # Testing Simulator construction on differential variables with
    # multi-dimensional and multiple indexing sets
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_sim_initialization_multi_index2(self):

        m = self.m
        m.s2 = Set(initialize=[(1, 1), (2, 2)])
        m.w1 = Var(m.t, m.s2)
        m.dw1 = DerivativeVar(m.w1)

        m.w2 = Var(m.s2, m.t)
        m.dw2 = DerivativeVar(m.w2)

        m.w3 = Var([0, 1], m.t, m.s2)
        m.dw3 = DerivativeVar(m.w3)

        t = IndexTemplate(m.t)
        
        def _deq1(m, t, i, j):
            return m.dw1[t, i, j] == m.w1[t, i, j]
        m.deq1 = Constraint(m.t, m.s2, rule=_deq1)

        def _deq2(m, *idx):
            return m.dw2[idx] == m.w2[idx]
        m.deq2 = Constraint(m.s2, m.t, rule=_deq2)

        def _deq3(m, i, t, j, k):
            return m.dw3[i, t, j, k] == m.w1[t, j, k] + m.w2[j, k, t]
        m.deq3 = Constraint([0, 1], m.t, m.s2, rule=_deq3)

        mysim = Simulator(m)

        self.assertIs(mysim._contset, m.t)
        self.assertEqual(len(mysim._diffvars), 8)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w1[t, 2, 2]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[1, 1, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[2, 2, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[0, t, 1, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[1, t, 2, 2]) in mysim._diffvars)

        self.assertEqual(len(mysim._derivlist), 8)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 1, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 2, 2]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[1, 1, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[2, 2, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[0, t, 1, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[1, t, 2, 2]) in mysim._derivlist)

        self.assertEqual(len(mysim._templatemap), 4)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1, 1]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w1[t, 2, 2]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[1, 1, t]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[2, 2, t]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[0, t, 1, 1]) in
                         mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[1, t, 2, 2]) in
                         mysim._templatemap)

        self.assertEqual(len(mysim._rhsdict), 8)
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1, 1])], Param))
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw1[t, 2, 2])], Param))
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw2[1, 1, t])], Param))
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw2[2, 2, t])], Param))
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw3[0, t, 1, 1])],
            EXPR.SumExpression))
        self.assertTrue(isinstance(
            mysim._rhsdict[_GetItemIndexer(m.dw3[1, t, 2, 2])],
            EXPR.SumExpression))
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1, 1])].name,
                         'w1[{t},1,1]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 2, 2])].name,
                         'w1[{t},2,2]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[1, 1, t])].name,
                         'w2[1,1,{t}]')
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[2, 2, t])].name,
                         'w2[2,2,{t}]')

        self.assertEqual(len(mysim._rhsfun(0, [0] * 8)), 8)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)

        m.del_component('deq1')
        m.del_component('deq1_index')
        m.del_component('deq2')
        m.del_component('deq2_index')
        m.del_component('deq3')
        m.del_component('deq3_index')

    # Testing the Simulator construction on un-supported models and
    # components with a single indexing set
    def test_non_supported_single_index(self):

        # Can't simulate a model with no ContinuousSet 
        m = ConcreteModel()
        with self.assertRaises(DAE_Error):
            Simulator(m)

        # Can't simulate a model with multiple ContinuousSets
        m = ConcreteModel()
        m.s = ContinuousSet(bounds=(0, 10))
        m.t = ContinuousSet(bounds=(0, 5))
        with self.assertRaises(DAE_Error):
            Simulator(m)
        
        # Can't simulate a model with no Derivatives
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        with self.assertRaises(DAE_Error):
            Simulator(m)

        # Can't simulate a model with multiple RHS for a derivative 
        m = self.m

        def _diffeq(m, t):
            return m.dv[t] == m.v[t]**2 + m.v[t]
        m.con1 = Constraint(m.t, rule=_diffeq)
        m.con2 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        
        # Can't simulate a model with multiple derivatives in an
        # equation
        m = self.m

        def _diffeq(m, t):
            return m.dv[t] == m.dv[t] + m.v[t]**2
        m.con1 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')

    # Testing the Simulator construction on un-supported models and
    # components with multiple indexing sets
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_non_supported_multi_index(self):

        m = self.m
        m.v2 = Var(m.t, m.s)
        m.v3 = Var(m.s, m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.dv3 = DerivativeVar(m.v3)

        # Can't simulate a model with multiple RHS for a derivative 
        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.v2[t, s]**2 + m.v2[t, s]
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        m.con2 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.v3[s, t]**2 + m.v3[s, t]
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        m.con2 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        # Can't simulate a model with multiple derivatives in an
        # equation
        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.dv2[t, s] + m.v2[t, s]**2
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.dv3[s, t] + m.v3[s, t]**2
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

    # Testing the Simulator using scipy on unsupported models
    def test_scipy_unsupported(self):

        m = self.m
        m.a = Var(m.t)

        def _diffeq(m, t):
            return 0 == m.v[t]**2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)

        # Can't simulate a model with algebraic equations using scipy
        with self.assertRaises(DAE_Error):
            Simulator(m, package='scipy')
        m.del_component('con')

    # Testing Simulator construction on models with time-indexed algebraic
    # variables
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_time_indexed_algebraic(self):

        m = self.m
        m.a = Var(m.t)

        def _diffeq(m, t):
            return m.dv[t] == m.v[t]**2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)
        mysim = Simulator(m)

        t = IndexTemplate(m.t)

        self.assertEqual(len(mysim._algvars), 1)
        self.assertTrue(_GetItemIndexer(m.a[t]) in mysim._algvars)
        self.assertEqual(len(mysim._alglist), 0)
        m.del_component('con')

    # Testing Simulator construction on models with algebraic variables
    # indexed by time and other indexing sets
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_time_multi_indexed_algebraic(self):

        m = self.m
        m.v2 = Var(m.t, m.s)
        m.v3 = Var(m.s, m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.dv3 = DerivativeVar(m.v3)

        m.a2 = Var(m.t, m.s)

        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.v2[t, s]**2 + m.a2[t, s]
        m.con = Constraint(m.t, m.s, rule=_diffeq)

        m.a3 = Var(m.s, m.t)

        def _diffeq2(m, s, t):
            return m.dv3[s, t] == m.v3[s, t]**2 + m.a3[s, t]
        m.con2 = Constraint(m.s, m.t, rule=_diffeq2)
        mysim = Simulator(m)
        t = IndexTemplate(m.t)

        self.assertEqual(len(mysim._algvars), 6)
        self.assertTrue(_GetItemIndexer(m.a2[t, 1]) in mysim._algvars)
        self.assertTrue(_GetItemIndexer(m.a2[t, 3]) in mysim._algvars)
        self.assertTrue(_GetItemIndexer(m.a3[1, t]) in mysim._algvars)
        self.assertTrue(_GetItemIndexer(m.a3[3, t]) in mysim._algvars)
        m.del_component('con')
        m.del_component('con_index')
        m.del_component('con2')
        m.del_component('con2_index')

    # check that all diffvars have been added to templatemap even if not
    # appearing in RHS of a differential equation
    @unittest.skipIf(not casadi_available, "casadi not available")
    def test_nonRHS_vars(self):

        m = self.m
        m.v2 = Var(m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.p = Param(initialize=5)
        t = IndexTemplate(m.t)

        def _con(m, t):
            return m.dv2[t] == 10 + m.p
        m.con = Constraint(m.t, rule=_con)

        mysim = Simulator(m,package='casadi')
        self.assertEqual(len(mysim._templatemap), 1)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v2[t]))
        m.del_component('con')

class TestExpressionCheckers(unittest.TestCase):
    """
    Class for testing the pyomo.DAE simulator expression checkers.
    """
    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    # Testing checker for GetItemExpression objects
    def test_check_getitemexpression(self):

        m = self.m
        t = IndexTemplate(m.t)

        e = m.dv[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIs(e.arg(0), temp[0])
        self.assertIs(e.arg(1), temp[1])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0))
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)

        e = m.v[t] == m.dv[t]
        temp = _check_getitemexpression(e, 1)
        self.assertIs(e.arg(0), temp[1])
        self.assertIs(e.arg(1), temp[0])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0))
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)

        e = m.v[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)

    # Testing checker for ProductExpressions
    def test_check_productexpression(self):
        m = self.m
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)

        # Check multiplication by constant
        e = 5 * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)

        e = m.v[t] == 5 * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)

        # Check multiplication by fixed param
        e = m.p * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)

        e = m.v[t] == m.p * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)

        # Check multiplication by mutable param
        e = m.mp * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.mp, temp[1].arg(1))      # Reciprocal
        self.assertIs(e.arg(1), temp[1].arg(0))

        e = m.v[t] == m.mp * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.mp, temp[1].arg(1))      # Reciprocal
        self.assertIs(e.arg(0), temp[1].arg(0))

        # Check multiplication by var
        e = m.y * m.dv[t] / m.z == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(e.arg(1), temp[1].arg(0).arg(0))
        self.assertIs(m.z,        temp[1].arg(0).arg(1))

        e = m.v[t] == m.y * m.dv[t] / m.z
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(e.arg(0), temp[1].arg(0).arg(0))
        self.assertIs(m.z, temp[1].arg(0).arg(1))

        # Check having the DerivativeVar in the denominator
        e = m.y / (m.dv[t] * m.z) == m.mp
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.y,        temp[1].arg(0))
        self.assertIs(e.arg(1), temp[1].arg(1).arg(0))

        e = m.mp == m.y / (m.dv[t] * m.z)
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.y,        temp[1].arg(0))
        self.assertIs(e.arg(0), temp[1].arg(1).arg(0))
        
        # Check expression with no DerivativeVar
        e = m.v[t] * m.y / m.z == m.v[t] * m.y / m.z
        temp = _check_productexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_productexpression(e, 1)
        self.assertIsNone(temp)

    # Testing the checker for NegationExpressions
    def test_check_negationexpression(self):

        m = self.m
        t = IndexTemplate(m.t)

        e = -m.dv[t] == m.v[t]
        temp = _check_negationexpression(e, 0)
        self.assertIs(e.arg(0).arg(0), temp[0])
        self.assertIs(e.arg(1), temp[1].arg(0))
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0).arg(0))
        temp = _check_negationexpression(e, 1)
        self.assertIsNone(temp)

        e = m.v[t] == -m.dv[t]
        temp = _check_negationexpression(e, 1)
        self.assertIs(e.arg(0), temp[1].arg(0))
        self.assertIs(e.arg(1).arg(0), temp[0])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0).arg(0))
        temp = _check_negationexpression(e, 0)
        self.assertIsNone(temp)

        e = -m.v[t] == -m.v[t]
        temp = _check_negationexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_negationexpression(e, 1)
        self.assertIsNone(temp)


    # Testing the checker for SumExpressions
    def test_check_viewsumexpression(self):

        m = self.m 
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)

        e = m.dv[t] + m.y + m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.SumExpression)
        self.assertIs(type(temp[1].arg(0)), EXPR.GetItemExpression)
        self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
        self.assertEqual(-1, temp[1].arg(1).arg(0))
        self.assertIs(m.y, temp[1].arg(1).arg(1))
        self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
        self.assertEqual(-1, temp[1].arg(2).arg(0))
        self.assertIs(m.z, temp[1].arg(2).arg(1))

        e = m.v[t] == m.y + m.dv[t] + m.z
        temp = _check_viewsumexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.SumExpression)
        self.assertIs(type(temp[1].arg(0)), EXPR.GetItemExpression)
        self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
        self.assertIs(m.y, temp[1].arg(1).arg(1))
        self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
        self.assertIs(m.z, temp[1].arg(2).arg(1))

        e = 5 * m.dv[t] + 5 * m.y - m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)

        self.assertIs(type(temp[1].arg(0).arg(0)), EXPR.GetItemExpression)
        self.assertIs(m.y, temp[1].arg(0).arg(1).arg(1))
        self.assertIs(m.z, temp[1].arg(0).arg(2).arg(1))

        e = 2 + 5 * m.y - m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(temp, None)

@unittest.skipIf(not casadi_available, "Casadi is not available")
class TestCasadiSubstituters(unittest.TestCase):
    """
    Class for testing the Expression substituters for creating valid CasADi
    expressions
    """

    def setUp(self):
        """
        Setting up the testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    # Testing substituter for replacing GetItemExpression objects with
    # CasADi sym objects
    def test_substitute_casadi_sym(self):

        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)

        e = m.dv[t] + m.v[t] + m.y + t
        templatemap = {}
        e2 = substitute_pyomo2casadi(e, templatemap)

        self.assertEqual(len(templatemap), 2)
        self.assertIs(type(e2.arg(0)), casadi.SX)
        self.assertIs(type(e2.arg(1)), casadi.SX)
        self.assertIsNot(type(e2.arg(2)), casadi.SX)
        self.assertIs(type(e2.arg(3)), IndexTemplate)

        m.del_component('y')

    # Testing substituter for replacing Pyomo intrinsic functions with
    # CasADi intrinsic functions
    def test_substitute_casadi_intrinsic1(self):

        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)

        e = m.v[t] 
        templatemap = {}

        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(type(e3), casadi.SX)
        
        m.del_component('y')

    # Testing substituter for replacing Pyomo intrinsic functions with
    # CasADi intrinsic functions
    def test_substitute_casadi_intrinsic2(self):

        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)

        e = sin(m.dv[t]) + log(m.v[t]) + sqrt(m.y) + m.v[t] + t
        templatemap = {}

        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(e3.arg(0)._fcn, casadi.sin)
        self.assertIs(e3.arg(1)._fcn, casadi.log)
        self.assertIs(e3.arg(2)._fcn, casadi.sqrt)

        m.del_component('y')

    # Testing substituter for replacing Pyomo intrinsic functions with
    # CasADi intrinsic functions
    def test_substitute_casadi_intrinsic3(self):

        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)

        e = sin(m.dv[t] + m.v[t]) + log(m.v[t] * m.y + m.dv[t]**2)
        templatemap = {}

        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(e3.arg(0)._fcn, casadi.sin)
        self.assertIs(e3.arg(1)._fcn, casadi.log)

        m.del_component('y')

    # Testing substituter for replacing Pyomo intrinsic functions with
    # CasADi intrinsic functions
    def test_substitute_casadi_intrinsic4(self):

        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)

        e = m.v[t] * sin(m.dv[t] + m.v[t]) * t
        templatemap = {}

        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(type(e3.arg(0).arg(0)), casadi.SX)
        self.assertIs(e3.arg(0).arg(1)._fcn, casadi.sin)
        self.assertIs(type(e3.arg(1)), IndexTemplate)

        m.del_component('y')


class TestSimulationInterface():
    """
    Class to test running a simulation
    """

    def _print(self, model, profiles):
        import numpy as np
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        model.pprint()
        print(profiles)

    def _test(self, tname):

        ofile = join(currdir, tname + '.' + self.sim_mod + '.out')
        bfile = join(currdir, tname + '.' + self.sim_mod + '.txt')
        setup_redirect(ofile)

        # create model
        exmod = import_file(join(exdir, tname + '.py'))
        m = exmod.create_model()

        # Simulate model
        sim = Simulator(m, package=self.sim_mod)

        if hasattr(m, 'var_input'):
            tsim, profiles = sim.simulate(numpoints=100,
                                          varying_inputs=m.var_input)
        else:
            tsim, profiles = sim.simulate(numpoints=100)

        # Discretize model
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(m, nfe=10, ncp=5)

        # Initialize model
        sim.initialize_model()

        self._print(m, profiles)

        reset_redirect()
        if not os.path.exists(bfile):
            os.rename(ofile, bfile)

        # os.system('diff ' + ofile + ' ' + bfile)
        self.assertFileEqualsBaseline(ofile, bfile, tolerance=0.01)

    def _test_disc_first(self, tname):

        ofile = join(currdir, tname + '.' + self.sim_mod + '.out')
        bfile = join(currdir, tname + '.' + self.sim_mod + '.txt')
        setup_redirect(ofile)

        # create model
        exmod = import_file(join(exdir, tname + '.py'))
        m = exmod.create_model()

        # Discretize model
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(m, nfe=10, ncp=5)

        # Simulate model
        sim = Simulator(m, package=self.sim_mod)

        if hasattr(m, 'var_input'):
            tsim, profiles = sim.simulate(numpoints=100,
                                          varying_inputs=m.var_input)
        else:
            tsim, profiles = sim.simulate(numpoints=100)

        # Initialize model
        sim.initialize_model()

        self._print(m, profiles)

        reset_redirect()
        if not os.path.exists(bfile):
            os.rename(ofile, bfile)

        # os.system('diff ' + ofile + ' ' + bfile)
        self.assertFileEqualsBaseline(ofile, bfile, tolerance=0.01)


@unittest.skipIf(not scipy_available, "Scipy is not available")
class TestScipySimulation(unittest.TestCase, TestSimulationInterface):
    sim_mod = 'scipy'

    def test_ode_example(self):
        tname = 'simulator_ode_example'
        self._test(tname)

    def test_ode_example2(self):
        tname = 'simulator_ode_example'
        self._test_disc_first(tname)

    def test_ode_multindex_example(self):
        tname = 'simulator_ode_multindex_example'
        self._test(tname)

    def test_ode_multindex_example2(self):
        tname = 'simulator_ode_multindex_example'
        self._test_disc_first(tname)


@unittest.skipIf(not casadi_available, "Casadi is not available")
class TestCasadiSimulation(unittest.TestCase, TestSimulationInterface):
    sim_mod = 'casadi'

    def test_ode_example(self):
        tname = 'simulator_ode_example'
        self._test(tname)

    def test_ode_example2(self):
        tname = 'simulator_ode_example'
        self._test_disc_first(tname)

    def test_ode_multindex_example(self):
        tname = 'simulator_ode_multindex_example'
        self._test(tname)

    def test_ode_multindex_example2(self):
        tname = 'simulator_ode_multindex_example'
        self._test_disc_first(tname)

    def test_dae_example(self):
        tname = 'simulator_dae_example'
        self._test(tname)

    def test_dae_example2(self):
        tname = 'simulator_dae_example'
        self._test_disc_first(tname)

    def test_dae_multindex_example(self):
        tname = 'simulator_dae_multindex_example'
        self._test(tname)

    def test_dae_multindex_example2(self):
        tname = 'simulator_dae_multindex_example'
        self._test_disc_first(tname)


if __name__ == "__main__":

    unittest.main()
