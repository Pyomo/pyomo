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
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()
pynumero_ASL_available = AmplInterface.available()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestRooneyBieglerExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_model(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import rooney_biegler

        rooney_biegler.main()

    def test_model_with_constraint(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import (
            rooney_biegler_with_constraint,
        )

        rooney_biegler_with_constraint.main()

    @unittest.skipUnless(pynumero_ASL_available, "test requires libpynumero_ASL")
    @unittest.skipUnless(seaborn_available, "test requires seaborn")
    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import (
            parameter_estimation_example,
        )

        parameter_estimation_example.main()

    @unittest.skipUnless(seaborn_available, "test requires seaborn")
    def test_bootstrap_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import bootstrap_example

        bootstrap_example.main()

    @unittest.skipUnless(seaborn_available, "test requires seaborn")
    def test_likelihood_ratio_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import (
            likelihood_ratio_example,
        )

        likelihood_ratio_example.main()


@unittest.skipUnless(pynumero_ASL_available, "test requires libpynumero_ASL")
@unittest.skipUnless(ipopt_available, "The 'ipopt' solver is not available")
@unittest.skipUnless(
    parmest.parmest_available, "Cannot test parmest: required dependencies are missing"
)
class TestReactionKineticsExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_example(self):
        from pyomo.contrib.parmest.examples.reaction_kinetics import (
            simple_reaction_parmest_example,
        )

        simple_reaction_parmest_example.main()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestSemibatchExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_model(self):
        from pyomo.contrib.parmest.examples.semibatch import semibatch

        semibatch.main()

    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.semibatch import (
            parameter_estimation_example,
        )

        parameter_estimation_example.main()

    def test_scenario_example(self):
        from pyomo.contrib.parmest.examples.semibatch import scenario_example

        scenario_example.main()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesignExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @unittest.pytest.mark.expensive
    def test_model(self):
        from pyomo.contrib.parmest.examples.reactor_design import reactor_design

        reactor_design.main()

    @unittest.skipUnless(pynumero_ASL_available, "test requires libpynumero_ASL")
    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import (
            parameter_estimation_example,
        )

        parameter_estimation_example.main()

    @unittest.skipUnless(seaborn_available, "test requires seaborn")
    def test_bootstrap_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import bootstrap_example

        bootstrap_example.main()

    @unittest.pytest.mark.expensive
    def test_likelihood_ratio_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import (
            likelihood_ratio_example,
        )

        likelihood_ratio_example.main()

    @unittest.pytest.mark.expensive
    def test_leaveNout_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import leaveNout_example

        leaveNout_example.main()

    def test_timeseries_data_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import (
            timeseries_data_example,
        )

        timeseries_data_example.main()

    def test_multisensor_data_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import (
            multisensor_data_example,
        )

        multisensor_data_example.main()

    @unittest.skipUnless(
        matplotlib_available and seaborn_available,
        "test requires matplotlib and seaborn",
    )
    def test_datarec_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import datarec_example

        datarec_example.main()

    def test_update_suffix_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import update_suffix_example

        suffix_obj, new_vals, new_var_vals = update_suffix_example.main()

        # Check that the suffix object has been updated correctly
        for i, v in enumerate(new_var_vals):
            self.assertAlmostEqual(new_var_vals[i], new_vals[i], places=6)


if __name__ == "__main__":
    unittest.main()
