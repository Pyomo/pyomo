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
import pyomo.contrib.parmest.parmest as parmest
from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
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

    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import parameter_estimation_example
        parameter_estimation_example.main()

    def test_bootstrap_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import bootstrap_example
        bootstrap_example.main()

    def test_likelihood_ratio_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import likelihood_ratio_example
        likelihood_ratio_example.main()


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
class TestReactionKineticsExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_example(self):
        from pyomo.contrib.parmest.examples.reaction_kinetics import simple_reaction_parmest_example
        simple_reaction_parmest_example.main()


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
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
        from pyomo.contrib.parmest.examples.semibatch import parameter_estimation_example
        parameter_estimation_example.main()

    def test_scenario_example(self):
        from pyomo.contrib.parmest.examples.semibatch import scenario_example
        scenario_example.main()


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
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

    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import parameter_estimation_example
        parameter_estimation_example.main()

    def test_bootstrap_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import bootstrap_example
        bootstrap_example.main()

    @unittest.pytest.mark.expensive
    def test_likelihood_ratio_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import likelihood_ratio_example
        likelihood_ratio_example.main()

    @unittest.pytest.mark.expensive
    def test_leaveNout_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import leaveNout_example
        leaveNout_example.main()

    def test_timeseries_data_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import timeseries_data_example
        timeseries_data_example.main()

    def test_multisensor_data_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import multisensor_data_example
        multisensor_data_example.main()

    def test_datarec_example(self):
        from pyomo.contrib.parmest.examples.reactor_design import datarec_example
        datarec_example.main()


if __name__ == "__main__":
    unittest.main()
