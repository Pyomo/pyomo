#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
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

    def test_convert_param_to_var(self):
        # TODO: Check that this works for different structured models (indexed, blocks, etc)

        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            ReactorDesignExperiment,
        )

        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        # make model
        exp = ReactorDesignExperiment(data, 0)
        instance = exp.get_labeled_model()

        theta_names = ['k1', 'k2', 'k3']
        m_vars = parmest.utils.convert_params_to_vars(
            instance, theta_names, fix_vars=True
        )

        for v in theta_names:
            self.assertTrue(hasattr(m_vars, v))
            c = m_vars.find_component(v)
            self.assertIsInstance(c, pyo.Var)
            self.assertTrue(c.fixed)
            c_old = instance.find_component(v)
            self.assertEqual(pyo.value(c), pyo.value(c_old))
            self.assertTrue(c in m_vars.unknown_parameters)


if __name__ == "__main__":
    unittest.main()
