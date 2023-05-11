#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners:
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin,
#  University of Toledo, West Virginia University, et al. All rights reserved.
#
#  NOTICE. This Software was developed under funding from the
#  U.S. Department of Energy and the U.S. Government consequently retains
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable,
#  worldwide license in the Software to reproduce, distribute copies to the
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________


from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

import pyomo.common.unittest as unittest

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory('ipopt').available()


class TestReactorExample(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not scipy_available, "scipy is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_compute_FIM(self):
        from pyomo.contrib.doe.examples import reactor_compute_FIM

        reactor_compute_FIM.main()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_optimize_doe(self):
        from pyomo.contrib.doe.examples import reactor_optimize_doe

        reactor_optimize_doe.main()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_grid_search(self):
        from pyomo.contrib.doe.examples import reactor_grid_search

        reactor_grid_search.main()


if __name__ == "__main__":
    unittest.main()
