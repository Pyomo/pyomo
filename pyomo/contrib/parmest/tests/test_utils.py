#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import pandas as pd, pandas_available

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @unittest.pytest.mark.expensive
    def test_convert_param_to_var(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            reactor_design_model,
        )

        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        theta_names = ["k1", "k2", "k3"]

        instance = reactor_design_model(data.loc[0])
        solver = pyo.SolverFactory("ipopt")
        solver.solve(instance)

        instance_vars = parmest.utils.convert_params_to_vars(
            instance, theta_names, fix_vars=True
        )
        solver.solve(instance_vars)

        assert instance.k1() == instance_vars.k1()
        assert instance.k2() == instance_vars.k2()
        assert instance.k3() == instance_vars.k3()


if __name__ == "__main__":
    unittest.main()
