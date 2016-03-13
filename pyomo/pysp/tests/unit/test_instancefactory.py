#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
import os
import sys

import pyutilib.th as unittest

from pyomo.pysp.scenariotree.instance_factory import ScenarioTreeInstanceFactory

pysp_examples_dir = os.path.abspath(__file__)
for i in range(5):
    pysp_examples_dir = os.path.dirname(pysp_examples_dir)
pysp_examples_dir = os.path.join(pysp_examples_dir, "examples", "pysp")

# farmer location
farmer_examples_dir = os.path.join(pysp_examples_dir, "farmer")
farmer_model_dir = os.path.join(farmer_examples_dir, "models")
farmer_scenariotree_dir = os.path.join(farmer_examples_dir, "scenariodata")

class TestInstanceFactory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def test_random_bundles(self):
        self.assertTrue("ReferenceModel" not in sys.modules)
        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(farmer_model_dir,
                                        farmer_scenariotree_dir)
        scenario_tree = \
            scenario_instance_factory.generate_scenario_tree(random_bundles=2)
        self.assertEqual(scenario_tree.contains_bundles(), True)
        self.assertEqual(len(scenario_tree._scenario_bundles), 2)
        self.assertTrue("ReferenceModel" in sys.modules)
        del sys.modules["ReferenceModel"]

TestInstanceFactory = unittest.category('smoke','nightly','expensive')(TestInstanceFactory)

if __name__ == "__main__":
    unittest.main()
