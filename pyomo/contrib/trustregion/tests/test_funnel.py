#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Testing newly added funnel.py file

import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.funnel import Funnel


class TestFunnel(unittest.TestCase):
    def setUp(self):
        self.phi_init = 1.0
        self.f_best_init = 0.0
        self.phi_min = 0.1
        self.kappa_f = 0.5
        self.kappa_r = 1.5
        self.alpha = 0.9
        self.beta = 0.7
        self.mu_s = 0.01
        self.eta = 0.1
        self.funnel = Funnel(
            phi_init=self.phi_init,
            f_best_init=self.f_best_init,
            phi_min=self.phi_min,
            kappa_f=self.kappa_f,
            kappa_r=self.kappa_r,
            alpha=self.alpha,
            beta=self.beta,
            mu_s=self.mu_s,
            eta=self.eta,
        )

    def tearDown(self):
        pass

    def test_accept_f(self):
        """Test accept_f method."""
        self.funnel.accept_f(theta_new=0.5, f_new=-0.5)  # Use f_new < f_best
        self.assertEqual(self.funnel.f_best, -0.5)       # Expect f_best to update

    def test_classify_step(self):
        """Test classify_step method."""
        # Test 'f' condition
        status = self.funnel.classify_step(theta_old=1.0, theta_new=0.5, f_old=1.5, f_new=1.0, delta=0.1)
        self.assertEqual(status, 'f')

        # Test 'reject' condition
        status = self.funnel.classify_step(theta_old=1.0, theta_new=2.0, f_old=1.5, f_new=1.0, delta=0.1)
        self.assertEqual(status, 'reject')