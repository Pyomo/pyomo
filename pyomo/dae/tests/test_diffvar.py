#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# 
# Unit Tests for DerivativeVar Objects
#

import os
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, Var, Set, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from six import StringIO

currdir = dirname(abspath(__file__)) + os.sep


class TestDerivativeVar(unittest.TestCase):

    # test valid declarations
    def test_valid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set()
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)
        m.dv2 = DerivativeVar(m.v, wrt=(m.t, m.t))

        self.assertTrue(isinstance(m.dv, Var))
        self.assertTrue(isinstance(m.dv, DerivativeVar))
        self.assertTrue(m.dv._wrt[0] is m.t)
        self.assertTrue(m.dv._sVar is m.v)
        self.assertTrue(m.v._derivative[('t',)]() is m.dv)
        self.assertTrue(m.dv.type() is DerivativeVar)
        self.assertTrue(m.dv._index is m.t)
        self.assertTrue(m.dv2._wrt[0] is m.t)
        self.assertTrue(m.dv2._wrt[1] is m.t)
        self.assertTrue(m.v._derivative[('t', 't')]() is m.dv2)
        self.assertTrue(m.dv.get_state_var() is m.v)
        self.assertTrue(m.dv2.get_state_var() is m.v)
        del m.dv
        del m.dv2
        del m.v

        m.v = Var(m.s, m.t)
        m.dv = DerivativeVar(m.v)
        m.dv2 = DerivativeVar(m.v, wrt=(m.t, m.t))
        self.assertTrue(isinstance(m.dv, Var))
        self.assertTrue(isinstance(m.dv, DerivativeVar))
        self.assertTrue(m.dv._wrt[0] is m.t)
        self.assertTrue(m.dv._sVar is m.v)
        self.assertTrue(m.v._derivative[('t',)]() is m.dv)
        self.assertTrue(m.dv.type() is DerivativeVar)
        self.assertTrue(m.t in m.dv._implicit_subsets)
        self.assertTrue(m.s in m.dv._implicit_subsets)
        self.assertTrue(m.dv2._wrt[0] is m.t)
        self.assertTrue(m.dv2._wrt[1] is m.t)
        self.assertTrue(m.v._derivative[('t', 't')]() is m.dv2)
        del m.dv
        del m.dv2
        del m.v
        del m.v_index
        del m.dv_index
        del m.dv2_index

        m.v = Var(m.x, m.t)
        m.dv = DerivativeVar(m.v, wrt=m.x)
        m.dv2 = DerivativeVar(m.v, wrt=m.t)
        m.dv3 = DerivativeVar(m.v, wrt=(m.t, m.x))
        m.dv4 = DerivativeVar(m.v, wrt=[m.t, m.t])
        self.assertTrue(isinstance(m.dv, Var))
        self.assertTrue(isinstance(m.dv, DerivativeVar))
        self.assertTrue(m.dv._wrt[0] is m.x)
        self.assertTrue(m.dv._sVar is m.v)
        self.assertTrue(m.v._derivative[('x',)]() is m.dv)
        self.assertTrue(m.v._derivative[('t',)]() is m.dv2)
        self.assertTrue(m.v._derivative[('t', 'x')]() is m.dv3)
        self.assertTrue(m.v._derivative[('t', 't')]() is m.dv4)
        self.assertTrue(m.dv.type() is DerivativeVar)
        self.assertTrue(m.x in m.dv._implicit_subsets)
        self.assertTrue(m.t in m.dv._implicit_subsets)
        self.assertTrue(m.dv3._wrt[0] is m.t)
        self.assertTrue(m.dv3._wrt[1] is m.x)
        self.assertTrue(m.dv4._wrt[0] is m.t)
        self.assertTrue(m.dv4._wrt[1] is m.t)

    # test invalid declarations
    def test_invalid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.x, m.t)
        m.y = Var()

        # Not passing a Var as the first positional argument
        try:
            m.ds = DerivativeVar(m.s)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # Specifying both option aliases
        try:
            m.dv = DerivativeVar(m.v, wrt=m.t, withrespectto=m.t)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # Passing in Var not indexed by a ContinuousSet
        try:
            m.dy = DerivativeVar(m.y)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # Not specifying 'wrt' when Var indexed by multiple ContinuousSets
        try:
            m.dv3 = DerivativeVar(m.v3)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # 'wrt' is not a ContinuousSet
        try:
            m.dv2 = DerivativeVar(m.v2, wrt=m.s)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        try:
            m.dv2 = DerivativeVar(m.v2, wrt=(m.t, m.s))
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # Specified ContinuousSet does not index the Var
        try:
            m.dv = DerivativeVar(m.v, wrt=m.x)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        try:
            m.dv2 = DerivativeVar(m.v2, wrt=[m.t, m.x])
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # Declaring the same derivative twice
        m.dvdt = DerivativeVar(m.v)
        try:
            m.dvdt2 = DerivativeVar(m.v)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
        try:
            m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        m.dv3 = DerivativeVar(m.v3, wrt=(m.x, m.x))
        try:
            m.dv4 = DerivativeVar(m.v3, wrt=(m.x, m.x))
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

    # test DerivativeVar reclassification after discretization
    def test_reclassification(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.x, m.t)

        m.dv = DerivativeVar(m.v)
        m.dv2 = DerivativeVar(m.v2, wrt=(m.t, m.t))
        m.dv3 = DerivativeVar(m.v3, wrt=m.x)

        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)

        self.assertTrue(m.dv.type() is Var)
        self.assertTrue(m.dv2.type() is Var)
        self.assertTrue(m.dv.is_fully_discretized())
        self.assertTrue(m.dv2.is_fully_discretized())
        self.assertTrue(m.dv3.type() is DerivativeVar)
        self.assertFalse(m.dv3.is_fully_discretized())

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.x)
        self.assertTrue(m.dv3.type() is Var)
        self.assertTrue(m.dv3.is_fully_discretized())


if __name__ == "__main__":
    unittest.main()
