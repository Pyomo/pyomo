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

from pyomo.environ import (Var, Set, ConcreteModel, 
                           TransformationFactory)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

from six import StringIO

from pyomo.common.log import LoggingIntercept

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'dae'))


class TestFiniteDiff(unittest.TestCase):
    """
    Class for testing the pyomo.DAE finite difference discretization
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
        
    # test backward finite difference discretization
    # on var indexed by single ContinuousSet
    def test_disc_single_index_backward(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=5)
         
        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertEqual(len(m.dv1_disc_eq), 5)
        self.assertEqual(len(m.v1), 6)

        expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
        disc_info = m.t.get_discretization_info()

        self.assertEqual(disc_info['scheme'], 'BACKWARD Difference')

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertIn(m.dv1, m._pyomo_dae_reclassified_derivativevars)

        output = \
"""\
dv1_disc_eq : Size=5, Index=t, Active=True
    Key : Lower : Body                               : Upper : Active
    2.0 :   0.0 :   dv1[2.0] - 0.5*(v1[2.0] - v1[0]) :   0.0 :   True
    4.0 :   0.0 : dv1[4.0] - 0.5*(v1[4.0] - v1[2.0]) :   0.0 :   True
    6.0 :   0.0 : dv1[6.0] - 0.5*(v1[6.0] - v1[4.0]) :   0.0 :   True
    8.0 :   0.0 : dv1[8.0] - 0.5*(v1[8.0] - v1[6.0]) :   0.0 :   True
     10 :   0.0 :   dv1[10] - 0.5*(v1[10] - v1[8.0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test backward finite difference discretization on
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_backward(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=2)
         
        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertEqual(len(m.dv1dt2_disc_eq), 1)
        self.assertEqual(len(m.v1), 3)

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertIn(m.dv1, m._pyomo_dae_reclassified_derivativevars)
        self.assertIn(m.dv1dt2, m._pyomo_dae_reclassified_derivativevars)

        output = \
"""\
dv1dt2_disc_eq : Size=1, Index=t, Active=True
    Key : Lower : Body                                           : Upper : Active
     10 :   0.0 : dv1dt2[10] - 0.04*(v1[10] - 2*v1[5.0] + v1[0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1dt2_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test forward finite difference discretization
    # on var indexed by single ContinuousSet
    def test_disc_single_index_forward(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=5, scheme='FORWARD')

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertEqual(len(m.dv1_disc_eq), 5)
        self.assertEqual(len(m.v1), 6)

        expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
        disc_info = m.t.get_discretization_info()

        self.assertEqual(disc_info['scheme'], 'FORWARD Difference')

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertIn(m.dv1, m._pyomo_dae_reclassified_derivativevars)

        output = \
"""\
dv1_disc_eq : Size=5, Index=t, Active=True
    Key : Lower : Body                               : Upper : Active
      0 :   0.0 :     dv1[0] - 0.5*(v1[2.0] - v1[0]) :   0.0 :   True
    2.0 :   0.0 : dv1[2.0] - 0.5*(v1[4.0] - v1[2.0]) :   0.0 :   True
    4.0 :   0.0 : dv1[4.0] - 0.5*(v1[6.0] - v1[4.0]) :   0.0 :   True
    6.0 :   0.0 : dv1[6.0] - 0.5*(v1[8.0] - v1[6.0]) :   0.0 :   True
    8.0 :   0.0 :  dv1[8.0] - 0.5*(v1[10] - v1[8.0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test forward finite difference discretization
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_forward(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=2, scheme='FORWARD')

        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertEqual(len(m.dv1dt2_disc_eq), 1)
        self.assertEqual(len(m.v1), 3)

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertIn(m.dv1, m._pyomo_dae_reclassified_derivativevars)
        self.assertIn(m.dv1dt2, m._pyomo_dae_reclassified_derivativevars)

        output = \
"""\
dv1dt2_disc_eq : Size=1, Index=t, Active=True
    Key : Lower : Body                                          : Upper : Active
      0 :   0.0 : dv1dt2[0] - 0.04*(v1[10] - 2*v1[5.0] + v1[0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1dt2_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test central finite difference discretization
    # on var indexed by single ContinuousSet
    def test_disc_single_index_central(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=5, scheme='CENTRAL')

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertEqual(len(m.dv1_disc_eq), 4)
        self.assertEqual(len(m.v1), 6)

        expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
        disc_info = m.t.get_discretization_info()

        self.assertEqual(disc_info['scheme'], 'CENTRAL Difference')

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        output = \
"""\
dv1_disc_eq : Size=4, Index=t, Active=True
    Key : Lower : Body                                : Upper : Active
    2.0 :   0.0 :   dv1[2.0] - 0.25*(v1[4.0] - v1[0]) :   0.0 :   True
    4.0 :   0.0 : dv1[4.0] - 0.25*(v1[6.0] - v1[2.0]) :   0.0 :   True
    6.0 :   0.0 : dv1[6.0] - 0.25*(v1[8.0] - v1[4.0]) :   0.0 :   True
    8.0 :   0.0 :  dv1[8.0] - 0.25*(v1[10] - v1[6.0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test central finite difference discretization
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_central(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=2, scheme='CENTRAL')

        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertEqual(len(m.dv1dt2_disc_eq), 1)
        self.assertEqual(len(m.v1), 3)

        output = \
"""\
dv1dt2_disc_eq : Size=1, Index=t, Active=True
    Key : Lower : Body                                            : Upper : Active
    5.0 :   0.0 : dv1dt2[5.0] - 0.04*(v1[10] - 2*v1[5.0] + v1[0]) :   0.0 :   True
"""
        out = StringIO()
        m.dv1dt2_disc_eq.pprint(ostream=out)
        self.assertEqual(output, out.getvalue())

    # test collocation discretization on var indexed by ContinuousSet and Set
    def test_disc_multi_index(self):
        m = self.m.clone()
        m.v2 = Var(m.t, m.s)
        m.dv2 = DerivativeVar(m.v2)

        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=5)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2_disc_eq'))
        self.assertEqual(len(m.dv2_disc_eq), 15)
        self.assertEqual(len(m.v2), 18)

        expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
        disc_info = m.t.get_discretization_info()

        self.assertEqual(disc_info['scheme'], 'BACKWARD Difference')

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

    # test collocation discretization on var indexed by multiple ContinuousSets
    def test_disc_multi_index2(self):
        m = self.m.clone()
        m.t2 = ContinuousSet(bounds=(0, 5))
        m.v2 = Var(m.t, m.t2)
        m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
        m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t2)

        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=2)

        self.assertTrue(hasattr(m, 'dv2dt_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2dt2_disc_eq'))
        self.assertEqual(len(m.dv2dt_disc_eq), 6)
        self.assertEqual(len(m.dv2dt2_disc_eq), 6)
        self.assertEqual(len(m.v2), 9)

        expected_t_disc_points = [0, 5.0, 10]
        expected_t2_disc_points = [0, 2.5, 5]
        
        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_t_disc_points[idx])

        for idx, val in enumerate(list(m.t2)):
            self.assertAlmostEqual(val, expected_t2_disc_points[idx])

    # test collocation discretization on var indexed by ContinuousSet and
    # multi-dimensional Set
    def test_disc_multidimen_index(self):
        m = self.m.clone()
        m.s2 = Set(initialize=[('A', 'B'), ('C', 'D'), ('E', 'F')])
        m.v2 = Var(m.t, m.s2)
        m.dv2 = DerivativeVar(m.v2)
        m.v3 = Var(m.s2, m.t)
        m.dv3 = DerivativeVar(m.v3)

        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=5)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2_disc_eq'))
        self.assertTrue(hasattr(m, 'dv3_disc_eq'))
        self.assertEqual(len(m.dv2_disc_eq), 15)
        self.assertEqual(len(m.v2), 18)
        self.assertEqual(len(m.dv3_disc_eq), 15)
        self.assertEqual(len(m.v3), 18)

        expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
        disc_info = m.t.get_discretization_info()

        self.assertEqual(disc_info['scheme'], 'BACKWARD Difference')

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

    # test passing the discretization invalid options
    def test_disc_invalid_options(self):
        m = self.m.clone()

        with self.assertRaises(TypeError):
            TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.s)

        with self.assertRaises(ValueError):
            TransformationFactory('dae.finite_difference').apply_to(m, nfe=-1)

        with self.assertRaises(ValueError):
            TransformationFactory('dae.finite_difference').apply_to(m,
                                                                    scheme='foo')

        with self.assertRaises(ValueError):
            TransformationFactory('dae.finite_difference').apply_to(m,
                                                                    foo=True)

        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)
        with self.assertRaises(ValueError):
            TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)

        m = self.m.clone()
        disc = TransformationFactory('dae.finite_difference')
        disc.apply_to(m)
        with self.assertRaises(ValueError):
            disc.apply_to(m)

    # test discretization using fewer points than ContinuousSet initialized
    # with
    def test_initialized_continuous_set(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[0, 1, 2, 3, 4])
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            TransformationFactory('dae.finite_difference').apply_to(m, nfe=2)
        self.assertIn('More finite elements', log_out.getvalue())

    # Test discretizing an invalid derivative
    def test_invalid_derivative(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v, wrt=(m.t, m.t, m.t))

        with self.assertRaises(DAE_Error):
            TransformationFactory('dae.finite_difference').apply_to(m)

    # test trying to discretize a ContinuousSet twice
    def test_discretize_twice(self):
        m = self.m.clone()

        disc1 = TransformationFactory('dae.finite_difference')
        disc1.apply_to(m, nfe=5)

        disc2 = TransformationFactory('dae.finite_difference')

        with self.assertRaises(DAE_Error):
            disc2.apply_to(m, nfe=5)


if __name__ == "__main__":
    unittest.main()
