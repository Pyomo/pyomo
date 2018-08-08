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

from pyomo.environ import (Var, Set, ConcreteModel, value, Constraint,
                           TransformationFactory, pyomo)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

import os
from six import StringIO
from pyutilib.misc import setup_redirect, reset_redirect
from pyutilib.misc import import_file

from pyomo.common.log import  LoggingIntercept

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'dae'))

try:
    import numpy
    numpy_available = True
except ImportError:
    numpy_available = False


class TestCollocation(unittest.TestCase):
    """
    Class for testing the pyomo.DAE collocation discretization
    """

    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v1 = Var(m.t)
        m.dv1 = DerivativeVar(m.v1)
        m.s = Set(initialize=[1, 2, 3], ordered=True)
        
    # test collocation discretization with radau points 
    # on var indexed by single ContinuousSet
    def test_disc_single_index_radau(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)
         
        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 16)

        expected_tau_points = [0.0, 0.1550510257216822,
                               0.64494897427831788, 1.0]
        expected_disc_points = [0, 0.310102, 1.289898, 2.0, 2.310102,
                                3.289898, 4.0, 4.310102, 5.289898, 6.0,
                                6.310102, 7.289898, 8.0, 8.310102, 9.289898,
                                10]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        output = \
"""\
dv1_disc_eq : Size=15, Index=t, Active=True
    Key      : Lower : Body                                                                                                                        : Upper : Active
    0.310102 :   0.0 :   dv1[0.310102] - (-2.06969384567*v1[0] + 1.6123724357*v1[0.310102] + 0.583920042345*v1[1.289898] - 0.126598632371*v1[2.0]) :   0.0 :   True
    1.289898 :   0.0 :   dv1[1.289898] - (0.86969384567*v1[0] - 1.78392004235*v1[0.310102] + 0.387627564304*v1[1.289898] + 0.526598632371*v1[2.0]) :   0.0 :   True
         2.0 :   0.0 :                             dv1[2.0] - (-1.5*v1[0] + 2.76598632371*v1[0.310102] - 3.76598632371*v1[1.289898] + 2.5*v1[2.0]) :   0.0 :   True
    2.310102 :   0.0 : dv1[2.310102] - (-2.06969384567*v1[2.0] + 1.6123724357*v1[2.310102] + 0.583920042345*v1[3.289898] - 0.126598632371*v1[4.0]) :   0.0 :   True
    3.289898 :   0.0 : dv1[3.289898] - (0.86969384567*v1[2.0] - 1.78392004235*v1[2.310102] + 0.387627564304*v1[3.289898] + 0.526598632371*v1[4.0]) :   0.0 :   True
         4.0 :   0.0 :                           dv1[4.0] - (-1.5*v1[2.0] + 2.76598632371*v1[2.310102] - 3.76598632371*v1[3.289898] + 2.5*v1[4.0]) :   0.0 :   True
    4.310102 :   0.0 : dv1[4.310102] - (-2.06969384567*v1[4.0] + 1.6123724357*v1[4.310102] + 0.583920042345*v1[5.289898] - 0.126598632371*v1[6.0]) :   0.0 :   True
    5.289898 :   0.0 : dv1[5.289898] - (0.86969384567*v1[4.0] - 1.78392004235*v1[4.310102] + 0.387627564304*v1[5.289898] + 0.526598632371*v1[6.0]) :   0.0 :   True
         6.0 :   0.0 :                           dv1[6.0] - (-1.5*v1[4.0] + 2.76598632371*v1[4.310102] - 3.76598632371*v1[5.289898] + 2.5*v1[6.0]) :   0.0 :   True
    6.310102 :   0.0 : dv1[6.310102] - (-2.06969384567*v1[6.0] + 1.6123724357*v1[6.310102] + 0.583920042345*v1[7.289898] - 0.126598632371*v1[8.0]) :   0.0 :   True
    7.289898 :   0.0 : dv1[7.289898] - (0.86969384567*v1[6.0] - 1.78392004235*v1[6.310102] + 0.387627564304*v1[7.289898] + 0.526598632371*v1[8.0]) :   0.0 :   True
         8.0 :   0.0 :                           dv1[8.0] - (-1.5*v1[6.0] + 2.76598632371*v1[6.310102] - 3.76598632371*v1[7.289898] + 2.5*v1[8.0]) :   0.0 :   True
    8.310102 :   0.0 :  dv1[8.310102] - (-2.06969384567*v1[8.0] + 1.6123724357*v1[8.310102] + 0.583920042345*v1[9.289898] - 0.126598632371*v1[10]) :   0.0 :   True
    9.289898 :   0.0 :  dv1[9.289898] - (0.86969384567*v1[8.0] - 1.78392004235*v1[8.310102] + 0.387627564304*v1[9.289898] + 0.526598632371*v1[10]) :   0.0 :   True
          10 :   0.0 :                             dv1[10] - (-1.5*v1[8.0] + 2.76598632371*v1[8.310102] - 3.76598632371*v1[9.289898] + 2.5*v1[10]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test collocation discretization with radau points
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_radau(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)
         
        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertTrue(len(m.dv1dt2_disc_eq) == 4)
        self.assertTrue(len(m.v1) == 5)

        output = \
"""\
dv1dt2_disc_eq : Size=4, Index=t, Active=True
    Key      : Lower : Body                                                                : Upper : Active
    1.666667 :   0.0 :  dv1dt2[1.666667] - (0.24*v1[0] - 0.36*v1[1.666667] + 0.12*v1[5.0]) :   0.0 :   True
         5.0 :   0.0 :       dv1dt2[5.0] - (0.24*v1[0] - 0.36*v1[1.666667] + 0.12*v1[5.0]) :   0.0 :   True
    6.666667 :   0.0 : dv1dt2[6.666667] - (0.24*v1[5.0] - 0.36*v1[6.666667] + 0.12*v1[10]) :   0.0 :   True
          10 :   0.0 :       dv1dt2[10] - (0.24*v1[5.0] - 0.36*v1[6.666667] + 0.12*v1[10]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1dt2_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test collocation discretization with legendre points 
    # on var indexed by single ContinuousSet
    def test_disc_single_index_legendre(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3, scheme='LAGRANGE-LEGENDRE')
         
        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'v1_t_cont_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1_t_cont_eq) == 5)
        self.assertTrue(len(m.v1) == 21)

        expected_tau_points = [0.0, 0.11270166537925834, 0.49999999999999989,
                               0.88729833462074226]
        expected_disc_points = [0, 0.225403, 1.0, 1.774597, 2.0, 2.225403,
                                3.0, 3.774597, 4.0, 4.225403, 5.0, 5.774597,
                                6.0, 6.225403, 7.0, 7.774597, 8.0,
                                8.225403, 9.0, 9.774597, 10]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-LEGENDRE')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        output = \
"""\
dv1_disc_eq : Size=15, Index=t, Active=True
    Key      : Lower : Body                                                                                                      : Upper : Active
    0.225403 :   0.0 :   dv1[0.225403] - (-3.0*v1[0] + 2.5*v1[0.225403] + 0.581988897472*v1[1.0] - 0.0819888974716*v1[1.774597]) :   0.0 :   True
         1.0 :   0.0 :                dv1[1.0] - (1.5*v1[0] - 2.86374306092*v1[0.225403] + v1[1.0] + 0.36374306092*v1[1.774597]) :   0.0 :   True
    1.774597 :   0.0 :      dv1[1.774597] - (-3.0*v1[0] + 5.08198889747*v1[0.225403] - 4.58198889747*v1[1.0] + 2.5*v1[1.774597]) :   0.0 :   True
    2.225403 :   0.0 : dv1[2.225403] - (-3.0*v1[2.0] + 2.5*v1[2.225403] + 0.581988897472*v1[3.0] - 0.0819888974716*v1[3.774597]) :   0.0 :   True
         3.0 :   0.0 :              dv1[3.0] - (1.5*v1[2.0] - 2.86374306092*v1[2.225403] + v1[3.0] + 0.36374306092*v1[3.774597]) :   0.0 :   True
    3.774597 :   0.0 :    dv1[3.774597] - (-3.0*v1[2.0] + 5.08198889747*v1[2.225403] - 4.58198889747*v1[3.0] + 2.5*v1[3.774597]) :   0.0 :   True
    4.225403 :   0.0 : dv1[4.225403] - (-3.0*v1[4.0] + 2.5*v1[4.225403] + 0.581988897472*v1[5.0] - 0.0819888974716*v1[5.774597]) :   0.0 :   True
         5.0 :   0.0 :              dv1[5.0] - (1.5*v1[4.0] - 2.86374306092*v1[4.225403] + v1[5.0] + 0.36374306092*v1[5.774597]) :   0.0 :   True
    5.774597 :   0.0 :    dv1[5.774597] - (-3.0*v1[4.0] + 5.08198889747*v1[4.225403] - 4.58198889747*v1[5.0] + 2.5*v1[5.774597]) :   0.0 :   True
    6.225403 :   0.0 : dv1[6.225403] - (-3.0*v1[6.0] + 2.5*v1[6.225403] + 0.581988897472*v1[7.0] - 0.0819888974716*v1[7.774597]) :   0.0 :   True
         7.0 :   0.0 :              dv1[7.0] - (1.5*v1[6.0] - 2.86374306092*v1[6.225403] + v1[7.0] + 0.36374306092*v1[7.774597]) :   0.0 :   True
    7.774597 :   0.0 :    dv1[7.774597] - (-3.0*v1[6.0] + 5.08198889747*v1[6.225403] - 4.58198889747*v1[7.0] + 2.5*v1[7.774597]) :   0.0 :   True
    8.225403 :   0.0 : dv1[8.225403] - (-3.0*v1[8.0] + 2.5*v1[8.225403] + 0.581988897472*v1[9.0] - 0.0819888974716*v1[9.774597]) :   0.0 :   True
         9.0 :   0.0 :              dv1[9.0] - (1.5*v1[8.0] - 2.86374306092*v1[8.225403] + v1[9.0] + 0.36374306092*v1[9.774597]) :   0.0 :   True
    9.774597 :   0.0 :    dv1[9.774597] - (-3.0*v1[8.0] + 5.08198889747*v1[8.225403] - 4.58198889747*v1[9.0] + 2.5*v1[9.774597]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test collocation discretization with legendre points
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_legendre(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, scheme='LAGRANGE-LEGENDRE')
         
        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertTrue(hasattr(m, 'v1_t_cont_eq'))
        self.assertTrue(len(m.dv1dt2_disc_eq) == 4)
        self.assertTrue(len(m.v1_t_cont_eq) == 2)
        self.assertTrue(len(m.v1) == 7)

        output = \
"""\
dv1dt2_disc_eq : Size=4, Index=t, Active=True
    Key      : Lower : Body                                                                                          : Upper : Active
    1.056624 :   0.0 :   dv1dt2[1.056624] - (0.48*v1[0] - 0.655692193817*v1[1.056624] + 0.175692193817*v1[3.943376]) :   0.0 :   True
    3.943376 :   0.0 :   dv1dt2[3.943376] - (0.48*v1[0] - 0.655692193817*v1[1.056624] + 0.175692193817*v1[3.943376]) :   0.0 :   True
    6.056624 :   0.0 : dv1dt2[6.056624] - (0.48*v1[5.0] - 0.655692193817*v1[6.056624] + 0.175692193817*v1[8.943376]) :   0.0 :   True
    8.943376 :   0.0 : dv1dt2[8.943376] - (0.48*v1[5.0] - 0.655692193817*v1[6.056624] + 0.175692193817*v1[8.943376]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1dt2_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test collocation discretization on var indexed by ContinuousSet and Set
    def test_disc_multi_index(self):
        m = self.m.clone()
        m.v2 = Var(m.t, m.s)
        m.dv2 = DerivativeVar(m.v2)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2_disc_eq'))
        self.assertTrue(len(m.dv2_disc_eq) == 45)
        self.assertTrue(len(m.v2) == 48)

        expected_tau_points = [0.0, 0.1550510257216822, 0.64494897427831788,
                               1.0]
        expected_disc_points = [0, 0.310102, 1.289898, 2.0, 2.310102, 3.289898,
                                4.0, 4.310102, 5.289898, 6.0, 6.310102,
                                7.289898, 8.0, 8.310102, 9.289898, 10]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

    # test collocation discretization on var indexed by multiple ContinuousSets
    def test_disc_multi_index2(self):
        m = self.m.clone()
        m.t2 = ContinuousSet(bounds=(0, 5))
        m.v2 = Var(m.t, m.t2)
        m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
        m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t2)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)

        self.assertTrue(hasattr(m, 'dv2dt_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2dt2_disc_eq'))
        self.assertTrue(len(m.dv2dt_disc_eq) == 20)
        self.assertTrue(len(m.dv2dt2_disc_eq) == 20)
        self.assertTrue(len(m.v2) == 25)

        expected_t_disc_points = [0, 1.666667, 5.0, 6.666667, 10]
        expected_t2_disc_points = [0, 0.833333, 2.5, 3.333333, 5]
        
        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_t_disc_points[idx])

        for idx, val in enumerate(list(m.t2)):
            self.assertAlmostEqual(val, expected_t2_disc_points[idx])

    # test passing the discretization invalid options
    def test_disc_invalid_options(self):
        m = self.m.clone()

        try:
            TransformationFactory('dae.collocation').apply_to(m, wrt=m.s)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        try:
            TransformationFactory('dae.collocation').apply_to(m, nfe=-1)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        try:
            TransformationFactory('dae.collocation').apply_to(m, ncp=0)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        try:
            TransformationFactory('dae.collocation').apply_to(m, scheme='foo')
            self.fail('Expected ValueError')
        except ValueError:
            pass

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)
        try:
            TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m)
        try:
            disc.apply_to(m)
            self.fail('Expected ValueError')
        except ValueError:
            pass

    # test looking up radau collocation points
    def test_lookup_radau_collocation_points(self):
        # Save initial flag value
        colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available

        # Numpy flag must be False to test lookup
        pyomo.dae.plugins.colloc.numpy_available = False

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 16)

        expected_tau_points = [0.0, 0.1550510257216822,
                               0.64494897427831788,
                               1.0]
        expected_disc_points = [0, 0.310102, 1.289898, 2.0, 2.310102,
                                3.289898,
                                4.0, 4.310102, 5.289898, 6.0, 6.310102,
                                7.289898, 8.0, 8.310102, 9.289898, 10]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        m = self.m.clone()
        try:
            disc = TransformationFactory('dae.collocation')
            disc.apply_to(m, ncp=15, scheme='LAGRANGE-RADAU')
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # Restore initial flag value
        pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail

    # test looking up legendre collocation points
    def test_lookup_legendre_collocation_points(self):
        # Save initial flag value
        colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available

        # Numpy flag must be False to test lookup
        pyomo.dae.plugins.colloc.numpy_available = False

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3, scheme='LAGRANGE-LEGENDRE')

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 21)

        expected_tau_points = [0.0, 0.11270166537925834, 0.49999999999999989,
                               0.88729833462074226]
        expected_disc_points = [0, 0.225403, 1.0, 1.774597, 2.0, 2.225403,
                                3.0, 3.774597, 4.0, 4.225403, 5.0, 5.774597,
                                6.0, 6.225403, 7.0, 7.774597, 8.0, 8.225403,
                                9.0, 9.774597, 10]

        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-LEGENDRE')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        m = self.m.clone()
        try:
            disc = TransformationFactory('dae.collocation')
            disc.apply_to(m, ncp=15, scheme='LAGRANGE-LEGENDRE')
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # Restore initial flag value
        pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail

    # test discretization using fewer points than ContinuousSet initialized
    # with
    def test_initialized_continuous_set(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[0, 1, 2, 3, 4])
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            TransformationFactory('dae.collocation').apply_to(m, nfe=2)
        self.assertIn('More finite elements', log_out.getvalue())

    # Test discretizing an invalid derivative
    def test_invalid_derivative(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v, wrt=(m.t, m.t, m.t))

        try:
            TransformationFactory('dae.collocation').apply_to(m)
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

    # test reduce_collocation_points invalid options
    def test_reduce_colloc_invalid(self):
        m = self.m.clone()
        m.u = Var(m.t)
        m2 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc2 = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        # No ContinuousSet specified
        try:
            disc.reduce_collocation_points(m, contset=None)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # Component passed in is not a ContinuousSet
        try:
            disc.reduce_collocation_points(m, contset=m.s)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # Call reduce_collocation_points method before applying discretization
        try:
            disc2.reduce_collocation_points(m2, contset=m2.t)
            self.fail('Expected RuntimeError')
        except RuntimeError:
            pass

        # Call reduce_collocation_points on a ContinuousSet that hasn't been
        #  discretized
        m2.tt = ContinuousSet(bounds=(0, 1))
        disc2.apply_to(m2, wrt=m2.t)
        try:
            disc2.reduce_collocation_points(m2, contset=m2.tt)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # No Var specified
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=None)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # Component passed in is not a Var
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.s)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # New ncp not specified
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=None)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # Negative ncp specified
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=-3)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # Too large ncp specified
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=10)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # Passing Vars not indexed by the ContinuousSet
        m.v2 = Var()
        m.v3 = Var(m.s)
        m.v4 = Var(m.s, m.s)

        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v2, ncp=1)
            self.fail('Expected IndexError')
        except IndexError:
            pass

        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v3, ncp=1)
            self.fail('Expected IndexError')
        except IndexError:
            pass

        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.v4, ncp=1)
            self.fail('Expected IndexError')
        except IndexError:
            pass

        # Calling reduce_collocation_points more than once
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)
        try:
            disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)
            self.fail('Expected RuntimeError')
        except RuntimeError:
            pass

    # test reduce_collocation_points on var indexed by single ContinuousSet
    def test_reduce_colloc_single_index(self):
        m = self.m.clone()
        m.u = Var(m.t)

        m2 = m.clone()
        m3 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)

        self.assertTrue(hasattr(m, 'u_interpolation_constraints'))
        self.assertEqual(len(m.u_interpolation_constraints), 10)

        disc2 = TransformationFactory('dae.collocation')
        disc2.apply_to(m2, wrt=m2.t, nfe=5, ncp=3)
        disc2.reduce_collocation_points(m2, contset=m2.t, var=m2.u, ncp=3)

        self.assertFalse(hasattr(m2, 'u_interpolation_constraints'))

        disc3 = TransformationFactory('dae.collocation')
        disc3.apply_to(m3, wrt=m3.t, nfe=5, ncp=3)
        disc3.reduce_collocation_points(m3, contset=m3.t, var=m3.u, ncp=2)

        self.assertTrue(hasattr(m3, 'u_interpolation_constraints'))
        self.assertEqual(len(m3.u_interpolation_constraints), 5)

    # test reduce_collocation_points on var indexed by ContinuousSet and Set
    def test_reduce_colloc_multi_index(self):
        m = self.m.clone()
        m.u = Var(m.t, m.s)

        m2 = m.clone()
        m3 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)

        self.assertTrue(hasattr(m, 'u_interpolation_constraints'))
        self.assertEqual(len(m.u_interpolation_constraints), 30)

        disc2 = TransformationFactory('dae.collocation')
        disc2.apply_to(m2, wrt=m2.t, nfe=5, ncp=3)
        disc2.reduce_collocation_points(m2, contset=m2.t, var=m2.u, ncp=3)

        self.assertFalse(hasattr(m2, 'u_interpolation_constraints'))

        disc3 = TransformationFactory('dae.collocation')
        disc3.apply_to(m3, wrt=m3.t, nfe=5, ncp=3)
        disc3.reduce_collocation_points(m3, contset=m3.t, var=m3.u, ncp=2)

        self.assertTrue(hasattr(m3, 'u_interpolation_constraints'))
        self.assertEqual(len(m3.u_interpolation_constraints), 15)
