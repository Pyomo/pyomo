#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This is a test to ensure all of the parmest examples run.
# assert statements should be included in the example files

import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_rooney_biegler_model(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import rooney_biegler
        rooney_biegler.main()
    
    def test_rooney_biegler_parmest(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import parmest_example
        parmest_example.main()
    
    def test_reaction_kinetics(self):
        from pyomo.contrib.parmest.examples.reaction_kinetics import simple_reaction_parmest_example
        simple_reaction_parmest_example.main()
        
    def test_semibatch_model(self):
        from pyomo.contrib.parmest.examples.semibatch import semibatch
        semibatch.main()
    
    #def test_semibatch_parmest(self):
    #    from pyomo.contrib.parmest.examples.semibatch import parmest_example
    #    parmest_example.main()
        
    def test_reactor_design_model(self):
        from pyomo.contrib.parmest.examples.reactor_design import reactor_design
        reactor_design.main()
        
    #def test_reactor_design_parmest(self):
    #    from pyomo.contrib.parmest.examples.reactor_design import parmest_example
    #    parmest_example.main()
    
    

if __name__ == "__main__":
    unittest.main()
