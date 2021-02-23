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

currdir = dirname(abspath(__file__)) + os.sep


class TestDerivativeVar(unittest.TestCase):

    # test valid declarations
    def test_valid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(dimen=1)
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)
        m.dv2 = DerivativeVar(m.v, wrt=(m.t, m.t))

        self.assertTrue(isinstance(m.dv, Var))
        self.assertTrue(isinstance(m.dv, DerivativeVar))
        self.assertTrue(m.dv._wrt[0] is m.t)
        self.assertTrue(m.dv._sVar is m.v)
        self.assertTrue(m.v._derivative[('t',)]() is m.dv)
        self.assertTrue(m.dv.ctype is DerivativeVar)
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
        self.assertTrue(m.dv.ctype is DerivativeVar)
        self.assertTrue(m.t in m.dv.index_set().set_tuple)
        self.assertTrue(m.s in m.dv.index_set().set_tuple)
        self.assertTrue(m.dv2._wrt[0] is m.t)
        self.assertTrue(m.dv2._wrt[1] is m.t)
        self.assertTrue(m.v._derivative[('t', 't')]() is m.dv2)
        del m.dv
        del m.dv2
        del m.v
        del m.v_index

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
        self.assertTrue(m.dv.ctype is DerivativeVar)
        self.assertTrue(m.x in m.dv.index_set().set_tuple)
        self.assertTrue(m.t in m.dv.index_set().set_tuple)
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
        with self.assertRaises(DAE_Error):
            m.ds = DerivativeVar(m.s)

        # Specifying both option aliases
        with self.assertRaises(TypeError):
            m.dv = DerivativeVar(m.v, wrt=m.t, withrespectto=m.t)

        # Passing in Var not indexed by a ContinuousSet
        with self.assertRaises(DAE_Error):
            m.dy = DerivativeVar(m.y)

        # Not specifying 'wrt' when Var indexed by multiple ContinuousSets
        with self.assertRaises(DAE_Error):
            m.dv3 = DerivativeVar(m.v3)

        # 'wrt' is not a ContinuousSet
        with self.assertRaises(DAE_Error):
            m.dv2 = DerivativeVar(m.v2, wrt=m.s)

        with self.assertRaises(DAE_Error):
            m.dv2 = DerivativeVar(m.v2, wrt=(m.t, m.s))

        # Specified ContinuousSet does not index the Var
        with self.assertRaises(DAE_Error):
            m.dv = DerivativeVar(m.v, wrt=m.x)

        with self.assertRaises(DAE_Error):
            m.dv2 = DerivativeVar(m.v2, wrt=[m.t, m.x])

        # Declaring the same derivative twice
        m.dvdt = DerivativeVar(m.v)
        with self.assertRaises(DAE_Error):
            m.dvdt2 = DerivativeVar(m.v)

        m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
        with self.assertRaises(DAE_Error):
            m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t)

        m.dv3 = DerivativeVar(m.v3, wrt=(m.x, m.x))
        with self.assertRaises(DAE_Error):
            m.dv4 = DerivativeVar(m.v3, wrt=(m.x, m.x))

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

        self.assertTrue(m.dv.ctype is Var)
        self.assertTrue(m.dv2.ctype is Var)
        self.assertTrue(m.dv.is_fully_discretized())
        self.assertTrue(m.dv2.is_fully_discretized())
        self.assertTrue(m.dv3.ctype is DerivativeVar)
        self.assertFalse(m.dv3.is_fully_discretized())

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.x)
        self.assertTrue(m.dv3.ctype is Var)
        self.assertTrue(m.dv3.is_fully_discretized())


if __name__ == "__main__":
    unittest.main()
