#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#!/usr/bin/env python

import pyutilib.th as unittest
from pyomo.common.config import ConfigBlock

from pyomo.core.expr.current import identify_variables
from pyomo.environ import ConcreteModel, Var, Reals, Objective, Constraint, ExternalFunction, SolverFactory, value, sqrt, sin
from pyomo.opt import check_available_solvers

try:
    import numpy
    numpy_available = True
    from pyomo.contrib.trustregion.PyomoInterface import PyomoInterface
except:
    numpy_available = False


@unittest.skipIf(numpy_available==False, "Skipping pyomo.contrib.trustregion tests because numpy is not installed.")
class TestPyomoInterfaceInitialization(unittest.TestCase):
    def setUp(self):
        m = ConcreteModel()
        m.z = Var(range(3),domain=Reals)
        for i in range(3):
            m.z[i] = 2.0
        m.x = Var(range(2))
        for i in range(2):
            m.x[i] = 2.0
        m.obj = Objective(expr= (m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6)
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))
        self.m = m

    def test_1(self):
        '''
        The simplest case that the black box has only two inputs and there is only one black block involved
        '''
        def blackbox(a,b):
            return sin(a-b)

        m = self.m
        bb = ExternalFunction(blackbox)
        m.eflist = [bb]
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        pI = PyomoInterface(m, [bb], ConfigBlock())
        self.assertEqual(pI.lx,2)
        self.assertEqual(pI.ly,1)
        self.assertEqual(pI.lz,3)
        self.assertEqual(len(list(identify_variables(m.c1.body))),3)
        self.assertEqual(len(list(identify_variables(m.c2.body))),2)

    def test_2(self):
        '''
        The simplest case that the black box has only one inputs and there is only a formula
        '''
        def blackbox(a):
            return sin(a)

        m = self.m
        bb = ExternalFunction(blackbox)
        m.eflist = [bb]
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0]-m.x[1]) == 2*sqrt(2.0))
        pI = PyomoInterface(m, [bb], ConfigBlock())
        self.assertEqual(pI.lx,1)
        self.assertEqual(pI.ly,1)
        self.assertEqual(pI.lz,5)
        self.assertEqual(len(list(identify_variables(m.c1.body))),3)
        self.assertEqual(len(list(identify_variables(m.c2.body))),2)
        self.assertEqual(len(m.tR.conset),1)
        self.assertEqual(len(list(identify_variables(m.tR.conset[1].body))),3)

    def test_3(self):
        '''
        The simplest case that the black box has only two inputs and there is only one black block involved
        '''
        def blackbox(a,b):
            return sin(a-b)

        m = self.m
        bb = ExternalFunction(blackbox)
        m.eflist = [bb]
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0], m.x[1]) == 2*sqrt(2.0))
        m.c3 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0], m.z[1]) == 2*sqrt(2.0))
        pI = PyomoInterface(m, [bb], ConfigBlock())
        self.assertEqual(pI.lx,3)
        self.assertEqual(pI.ly,2)
        self.assertEqual(pI.lz,2)
        self.assertEqual(len(list(identify_variables(m.c1.body))),3)
        self.assertEqual(len(list(identify_variables(m.c2.body))),2)
        self.assertEqual(len(list(identify_variables(m.c3.body))),3)


    def test_4(self):
        '''
        The simplest case that the black box has only two inputs and there is only one black block involved
        '''
        def blackbox(a,b):
            return sin(a-b)

        m = self.m
        bb = ExternalFunction(blackbox)
        m.eflist = [bb]
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0], m.x[1]) == 2*sqrt(2.0))
        m.c3 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0], m.z[1]) == 2*sqrt(2.0))
        pI = PyomoInterface(m, [bb], ConfigBlock())
        self.assertEqual(pI.lx,3)
        self.assertEqual(pI.ly,2)
        self.assertEqual(pI.lz,2)
        self.assertEqual(len(list(identify_variables(m.c1.body))),3)
        self.assertEqual(len(list(identify_variables(m.c2.body))),2)
        self.assertEqual(len(list(identify_variables(m.c3.body))),3)

    @unittest.skipIf(not check_available_solvers('ipopt'),
                     "The 'ipopt' solver is not available")
    @unittest.skipIf(not check_available_solvers('gjh'),
                     "The 'gjh' solver is not available")
    def test_execute_TRF(self):
        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(2), initialize=2.)
        m.x[1] = 1.0

        def blackbox(a,b):
            return sin(a-b)
        bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
            + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
        )
        m.c1 = Constraint(
            expr=m.x[0] * m.z[0]**2 + bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

        SolverFactory('trustregion').solve(m, [bb])

        self.assertAlmostEqual(value(m.obj), 0.277044789315, places=4)
        self.assertAlmostEqual(value(m.x[0]), 1.32193855369, places=4)
        self.assertAlmostEqual(value(m.x[1]), 0.628744699822, places=4)


if __name__ == '__main__':
    unittest.main()
