#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
    matplotlib,
    matplotlib_available,
    scipy_available,
    networkx_available,
)
from pyomo.contrib.incidence_analysis.visualize import spy_dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_dynamic_model,
    make_degenerate_solid_phase_model,
)


@unittest.skipUnless(matplotlib_available, "Matplotlib is not available")
@unittest.skipUnless(scipy_available, "SciPy is not available")
@unittest.skipUnless(networkx_available, "NetworkX is not available")
class TestSpy(unittest.TestCase):
    def test_spy_dulmage_mendelsohn(self):
        models = [
            make_gas_expansion_model(),
            make_dynamic_model(),
            make_degenerate_solid_phase_model(),
        ]
        for m in models:
            fig, ax = spy_dulmage_mendelsohn(m)
            # Note that this is a weak test. We just test that we can call the
            # plot method, it doesn't raise an error, and gives us back the
            # types we expect. We don't attempt to validate the resulting plot.
            self.assertTrue(isinstance(fig, matplotlib.pyplot.Figure))
            self.assertTrue(isinstance(ax, matplotlib.pyplot.Axes))


if __name__ == "__main__":
    unittest.main()
