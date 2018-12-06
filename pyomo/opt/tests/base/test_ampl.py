#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.opt.problem.ampl
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
import pyutilib.common

import pyomo.opt

old_tempdir = pyutilib.services.TempfileManager.tempdir

def filter(text):
    return 'Problem:' in text or text.startswith('NAME')

def filter_nl(text):
    return '# problem'

solver = None
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = pyomo.opt.check_available_solvers('glpk')

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir

    def test3_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = pyomo.opt.AmplModel(currdir+'test3.mod')
        """ Convert from MOD+DAT to NL """
        try:
            self.model.write(currdir+'test3.nl')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.nl', currdir+'test3.baseline.nl', filter=filter_nl, tolerance=1e-6)

    def test3_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = pyomo.opt.AmplModel(currdir+'test3.mod')
        try:
            self.model.write(currdir+'test3.lp')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("glpsol") is None:
                self.fail("Unexpected ApplicationError - glpsol is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("glpsol") is None:
                self.fail("Unexpected ConverterError - glpsol is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.lp', currdir+'test3.baseline.lp', filter=filter, tolerance=1e-6)

    def test3_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyomo.common.registered_executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(currdir+'test3.mod')
        try:
            self.model.write(currdir+'test3.mps')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.mps', currdir+'test3.baseline.mps', filter=filter, tolerance=1e-6)

    def test3a_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = pyomo.opt.AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.nl')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.nl', currdir+'test3.baseline.nl', filter=filter_nl, tolerance=1e-6)

    def test3a_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = pyomo.opt.AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.lp')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("glpsol") is None:
                self.fail("Unexpected ApplicationError - glpsol is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("glpsol") is None:
                self.fail("Unexpected ConverterError - glpsol is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.lp', currdir+'test3.baseline.lp', filter=filter, tolerance=1e-6)

    def test3a_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyomo.common.registered_executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.mps')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.mps', currdir+'test3.baseline.mps', filter=filter, tolerance=1e-6)

    def test3_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(currdir+'test3.mod')
        opt = pyomo.opt.SolverFactory('glpk')
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=currdir+'test3.out', format='json')
        self.assertMatchesJsonBaseline(currdir+'test3.out', currdir+'test3.baseline.out', tolerance=1e-6)

    def test3a_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        opt = pyomo.opt.SolverFactory('glpk')
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=currdir+'test3a.out', format='json')
        self.assertMatchesJsonBaseline(currdir+'test3a.out', currdir+'test3.baseline.out', tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()

