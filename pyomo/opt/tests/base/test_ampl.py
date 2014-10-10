#
# Unit Tests for pyomo.opt.problem.ampl
#
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import re
from nose.tools import nottest
import xml
import filecmp
import pyutilib.th as unittest
import pyutilib.services
import pyutilib.common
import pyomo.misc.plugin
import pyomo.opt
from pyomo.opt import ProblemFormat, ConverterError, AmplModel, SolverFactory
import pyomo
import pyomo.environ

old_tempdir = pyutilib.services.TempfileManager.tempdir

solver = pyomo.opt.load_solvers('glpk')


def filter(text):
    return 'Problem:' in text or text.startswith('NAME')

def filter_nl(text):
    return '# problem'

class Test(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        #
        # Reset all options
        #
        #for ep in pyomo.misc.plugin.ExtensionPoint(pyomo.misc.plugin.IOption):
            #ep.reset()

    def test3_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = AmplModel(currdir+'test3.mod')
        """ Convert from MOD+DAT to NL """
        try:
            self.model.write(currdir+'test3.nl')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.nl', currdir+'test3.baseline.nl', filter=filter_nl, tolerance=1e-6)

    def test3_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = AmplModel(currdir+'test3.mod')
        try:
            self.model.write(currdir+'test3.lp')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("glpsol") is None:
                self.fail("Unexpected ApplicationError - glpsol is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("glpsol") is None:
                self.fail("Unexpected ConverterError - glpsol is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.lp', currdir+'test3.baseline.lp', filter=filter, tolerance=1e-6)

    def test3_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyutilib.services.registered_executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = AmplModel(currdir+'test3.mod')
        try:
            self.model.write(currdir+'test3.mps')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3.mps', currdir+'test3.baseline.mps', filter=filter, tolerance=1e-6)

    def test3a_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.nl')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.nl', currdir+'test3.baseline.nl', filter=filter_nl, tolerance=1e-6)

    def test3a_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.lp')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("glpsol") is None:
                self.fail("Unexpected ApplicationError - glpsol is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("glpsol") is None:
                self.fail("Unexpected ConverterError - glpsol is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.lp', currdir+'test3.baseline.lp', filter=filter, tolerance=1e-6)

    def test3a_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyutilib.services.registered_executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        try:
            self.model.write(currdir+'test3a.mps')
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("ampl") is None:
                self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
            return
        self.assertFileEqualsBaseline(currdir+'test3a.mps', currdir+'test3.baseline.mps', filter=filter, tolerance=1e-6)

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test3_solve(self):
        self.model = AmplModel(currdir+'test3.mod')
        opt = solver['glpk']
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=currdir+'test3.out', format='json')
        self.assertMatchesJsonBaseline(currdir+'test3.out', currdir+'test3.baseline.out', tolerance=1e-6)

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test3a_solve(self):
        self.model = AmplModel(currdir+'test3a.mod', currdir+'test3a.dat')
        opt = solver['glpk']
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=currdir+'test3a.out', format='json')
        self.assertMatchesJsonBaseline(currdir+'test3a.out', currdir+'test3.baseline.out', tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()

