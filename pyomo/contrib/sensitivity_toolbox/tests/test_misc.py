# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
Unit Tests for interfacing with sIPOPT 
"""

import os
from os.path import abspath, dirname, normpath, join

import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.environ import ConcreteModel, Var, Param, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.contrib.sensitivity_toolbox.sensitivity_toolbox import sipopt

currdir = dirname(abspath(__file__)) + os.sep
exdir = normpath(join(currdir,'examples'))

#try:
#    import ipopt_sens
#    ipopt_sens_available = True
#except ImportError:
#    ipopt_sens_available = False


solver = 'ipopt_sens'
solver_io = 'nl'
opt = SolverFactory(solver, solver_io=solver_io) 


class TestSensitivityToolbox(unittest.TestCase):
    
    
    #test arguments
    @unittest.skipIf(opt is None, "ipopt_sense is not available")
    def test_bad_arg(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0,1))

        m.a = Param(initialize=1)
        m.b = Param(initialize=2)
        m.c = Param(initialize=3)

        m.x = Var(m.t)

        list_one = [m.a,m.b]
        list_two = [m.a,m.b,m.c]
        list_three = [m.a, m.x]
        
        try:
            Result = sipopt(m,list_one,list_two)
            self.fail("Expected Exception")
        except Exception:
            pass

        try:
            Result = sipopt(m,list_three,list_two)
            self.fail("Expected Exception")
        except Exception:
            pass

        try:
            Result = sipopt(m,list_one,list_three)
            self.fail("Expected Exception")
        except Exception:
            pass


    #test feedbackController Solution
    @unittest.skipIf(opt is None, "ipopt_sens is not available")
    def test_soln(self):
        exmod = import_file(join(exdir,'m_sipopt_feedbackController.py'))
        m = exmod.create_model()

        m.perturbed_a = Param(initialize=-0.25)
        m.perturbed_H = Param(initialize=0.5)

        m_sipopt = sipopt(m,[m.a,m.H],
                            [m.perturbed_a,m.perturbed_H])
        
        self.assertTrue(hasattr(m_sipopt,'sens_sol_state_1'))
        self.assertAlmostEqual(value(m_sipopt.J),0.0048956761,8)
         

if __name__=="__main__":
    unittest.main()



