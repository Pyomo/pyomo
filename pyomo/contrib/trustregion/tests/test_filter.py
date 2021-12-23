#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.filter import Filter, FilterElement

class TestFilter(unittest.TestCase):
    def setUp(self):
        self.objective = 1.0
        self.feasible = 0.5
        self.theta_max = 10.0
        self.tmpFilter = Filter()

    def tearDown(self):
        pass

    def test_FilterElement(self):
        fe = FilterElement(self.objective, self.feasible)
        self.assertEqual(fe.objective, self.objective)
        self.assertEqual(fe.feasible, self.feasible)

    def test_addToFilter(self):
        fe = FilterElement(self.objective, self.feasible)
        self.tmpFilter.addToFilter(fe)
        self.assertIn(fe, self.tmpFilter.TrustRegionFilter)

    def test_isAcceptable(self):
        fe = FilterElement(0.5, 0.25)
        # A sufficiently feasible element
        self.assertTrue(self.tmpFilter.isAcceptable(fe,
                                                    self.theta_max))
        fe = FilterElement(10.0, 15.0)
        # A sufficiently infeasible element
        self.assertFalse(self.tmpFilter.isAcceptable(fe,
                                                     self.theta_max))
