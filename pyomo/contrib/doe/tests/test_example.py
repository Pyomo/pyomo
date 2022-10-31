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


from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy_available,
)

import pyomo.common.unittest as unittest
import pyomo.contrib.doe.fim_doe as doe

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()

class TestReactorExample(unittest.TestCase):
    
    #@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    #def test_reactor_kinetics(self):
        #from pyomo.contrib.doe.example import reactor_kinetics
        #reactor_kinetics.main()
    
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not scipy_available, "scipy is not available")
    def test_reactor_compute_FIM(self):
        from pyomo.contrib.doe.example import reactor_compute_FIM
        reactor_compute_FIM.main()
        
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_optimize_doe(self):
        from pyomo.contrib.doe.example import reactor_optimize_doe
        reactor_optimize_doe.main()
        
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search(self):
        from pyomo.contrib.doe.example import reactor_grid_search
        reactor_grid_search.main()
        
        
if __name__ == "__main__":
    unittest.main()
