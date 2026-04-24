# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy,
    scipy_available,
    matplotlib,
    matplotlib_available,
)

import pyomo.common.unittest as unittest
import sys
import os

import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics

from pyomo.contrib.parmest.tests.test_parmest import _RANDOM_SEED_FOR_TESTING

testdir = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(
    not graphics.imports_available, "parmest.graphics imports are unavailable"
)
class TestGraphics(unittest.TestCase):
    def setUp(self):
        np.random.seed(_RANDOM_SEED_FOR_TESTING)
        self.A = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')
        )
        self.B = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')
        )

    def test_pairwise_plot(self):
        graphics.pairwise_plot(
            self.A,
            alpha=0.8,
            distributions=['Rect', 'MVN', 'KDE'],
            seed=_RANDOM_SEED_FOR_TESTING,
        )

    def test_grouped_boxplot(self):
        graphics.grouped_boxplot(self.A, self.B, normalize=True, group_names=['A', 'B'])

    def test_grouped_violinplot(self):
        graphics.grouped_violinplot(self.A, self.B)

    def test_profile_plot_smoke_single_parameter(self):
        prof = pd.DataFrame(
            {
                "profiled_theta": ["theta"] * 3,
                "theta_value": [0.8, 1.0, 1.2],
                "lr_stat": [3.0, 0.0, 4.0],
                "success": [True, True, False],
            }
        )
        graphics.profile_likelihood_plot({"profiles": prof}, alpha=0.95)

    def test_profile_plot_smoke_multi_parameter(self):
        prof = pd.DataFrame(
            {
                "profiled_theta": ["theta_a", "theta_a", "theta_b", "theta_b"],
                "theta_value": [1.0, 2.0, -1.0, 0.0],
                "lr_stat": [2.0, 0.0, 1.0, 0.0],
                "success": [True, True, True, True],
            }
        )
        graphics.profile_likelihood_plot({"profiles": prof}, alpha=0.9)


if __name__ == '__main__':
    unittest.main()
