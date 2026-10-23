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
                "obj": [2.5, 1.0, float("nan")],
                "lr_stat": [3.0, 0.0, float("nan")],
                "success": [True, True, False],
            }
        ).set_index(["profiled_theta", "theta_value"])
        intervals = pd.DataFrame(
            {"profiled_theta": ["theta"], "alpha": [0.95], "threshold": [3.84]}
        ).set_index(["profiled_theta", "alpha"])
        _, axes = graphics.profile_likelihood_plot(prof, intervals)
        ax = axes[0, 0]
        # The failed solve is marked on the x-axis
        self.assertEqual(ax.collections[0].get_offsets()[:, 0].tolist(), [1.2])
        # Profile line and threshold line
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(ax.lines[1].get_ydata()[0], 3.84)

        # The lr_stat threshold is not drawn on the objective axis
        _, axes = graphics.profile_likelihood_plot(prof, intervals, y="obj")
        self.assertEqual(len(axes[0, 0].lines), 1)

    def test_profile_plot_smoke_multi_parameter(self):
        prof = pd.DataFrame(
            {
                "profiled_theta": ["theta_a", "theta_a", "theta_b", "theta_b"],
                "theta_value": [1.0, 2.0, -1.0, 0.0],
                "lr_stat": [2.0, 0.0, 1.0, 0.0],
                "success": [True, True, True, True],
            }
        ).set_index(["profiled_theta", "theta_value"])
        graphics.profile_likelihood_plot(prof)


if __name__ == '__main__':
    unittest.main()
