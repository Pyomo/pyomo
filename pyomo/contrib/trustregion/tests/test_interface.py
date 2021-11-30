#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available
from pyomo.environ import (
    Var, ConcreteModel, Reals, ExternalFunction,
    Objective, Constraint, sqrt, sin, SolverFactory
    )

logger = logging.getLogger('pyomo.contrib.trustregion')

@unittest.skipIf(not numpy_available, "Cannot test the trustregion solver without numpy")
class TestTrustRegionInterface(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.)
        self.m.x = Var(range(2), initialize=2.)
        self.m.x[1] = 1.0

        def blackbox(a,b):
            return sin(a-b)

        self.m.bb = ExternalFunction(blackbox)

        self.m.obj = Objective(
            expr=(self.m.z[0]-1.0)**2 + (self.m.z[0]-self.m.z[1])**2
            + (self.m.z[2]-1.0)**2 + (self.m.x[0]-1.0)**4
            + (self.m.x[1]-1.0)**6
        )
        self.m.c1 = Constraint(
            expr=(self.m.x[0] * self.m.z[0]**2
                  + self.m.bb(self.m.x[0], self.m.x[1])
                  == 2*sqrt(2.0))
            )
        self.m.c2 = Constraint(
            expr=self.m.z[2]**4 * self.m.z[1]**2 + self.m.z[1] == 8+sqrt(2.0))

    def test_execute_TRF(self):
        model = self.m.clone()
        #SolverFactory('trustregion').solve(model)



if __name__ == '__main__':
    unittest.main()