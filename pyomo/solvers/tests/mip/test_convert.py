#
# Unit Tests for coopr.opt.base.convert
#
#

import os
import sys
from os.path import abspath, dirname
cooprdir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import re
import xml
import filecmp
import shutil
from nose.tools import nottest
import pyutilib.th as unittest
import pyutilib.services
import pyutilib.common
import coopr.core.plugin
from coopr.opt import ProblemFormat, ConverterError
import coopr

old_tempdir = pyutilib.services.TempfileManager.tempdir

def filter(line):
    return 'Problem' in line or line.startswith('NAME')

class MockArg(object):

    def __init__(self):
        pass

    def valid_problem_types(self):
        return [ProblemFormat.pyomo]

    def write(self,filename="", format=None, solver_capability=None, io_options={}):
        return (filename,None)

class MockArg2(MockArg):

    def valid_problem_types(self):
        return [ProblemFormat.nl]

    def write(self,filename="", format=None, solver_capability=None, io_options={}):
        OUTPUT=open(filename,"w")
        INPUT=open(currdir+"test4.nl")
        for line in INPUT:
            OUTPUT.write(line)
        OUTPUT.close()
        INPUT.close()
        return (filename,None)

class MockArg3(MockArg):

    def valid_problem_types(self):
        return [ProblemFormat.mod]

    def write(self,filename="", format=None, solver_capability=None, io_options={}):
        return (filename,None)

class MockArg4(MockArg):

    def write(self,filename="", format=None, solver_capability=None, io_options={}):
        OUTPUT=open(filename,"w")
        INPUT=open(currdir+"test4.nl")
        for line in INPUT:
            OUTPUT.write(line)
        OUTPUT.close()
        INPUT.close()
        return (filename,None)


class Test(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        #
        # Reset all options
        #
        #for ep in coopr.core.plugin.ExtensionPoint(coopre.core.plugin.IOption):
            #ep.reset()

    def test_nl_nl1(self):
        #""" Convert from NL to NL """
        ans = coopr.opt.convert_problem( ("test4.nl",), None, [ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl",))

    def test_nl_nl2(self):
        #""" Convert from NL to NL """
        ans = coopr.opt.convert_problem( ("test4.nl","tmp.nl"), None, [ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl","tmp.nl"))

    def test_nl_lp1(self):
        #""" Convert from NL to LP """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test4.nl",), None, [ProblemFormat.cpxlp])
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("pico_convert") is None:
                self.fail("Unexpected ApplicationError - pico_convert is enabled but not available: '%s'" % str(err))
            return
        except ConverterError:
            err = sys.exc_info()[1]
            if not pyutilib.services.registered_executable("pico_convert") is None:
                self.fail("Unexpected ConverterError - pico_convert is enabled but not available: '%s'" % str(err))
            return
        self.assertEqual(ans[0][0][-15:],"pico_convert.lp")
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test1_convert.lp")

    def test_mod_lp1(self):
        #""" Convert from MOD to LP """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test3.mod",), None, [ProblemFormat.cpxlp])
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
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test2_convert.lp", filter=filter)

    def test_mod_lp2(self):
        #""" Convert from MOD+DAT to LP """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test5.mod",currdir+"test5.dat"), None, [ProblemFormat.cpxlp])
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
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test3_convert.lp", filter=filter)

    def test_mod_nl1(self):
        #""" Convert from MOD to NL """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test3.mod",), None, [ProblemFormat.nl])
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
        self.assertTrue(ans[0][0].endswith('.nl'))
        #self.assertFileEqualsBinaryFile(ans[0][0], currdir+"test_mod_nl1.nl")

    def test_mod_nl2(self):
        #""" Convert from MOD+DAT to NL """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test5.mod",currdir+"test5.dat"), None, [ProblemFormat.nl])
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
        self.assertTrue(ans[0][0].endswith('.nl'))
        #self.assertFileEqualsBaseline(ans[0][0], currdir+"test_mod_nl2.nl")

    def test_mock_lp1(self):
        #""" Convert from Pyomo to LP """
        arg=MockArg()
        ans = coopr.opt.convert_problem( (arg,ProblemFormat.cpxlp,arg), None, [ProblemFormat.cpxlp])
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$",ans[0][0]), None)

    def test_pyomo_lp1(self):
        #""" Convert from Pyomo to LP with file"""
        ans = coopr.opt.convert_problem( (currdir+'model.py',ProblemFormat.cpxlp,), None, [ProblemFormat.cpxlp])
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$",ans[0][0]), None)

    def test_mock_lp2(self):
        #""" Convert from NL to LP """
        arg=MockArg2()
        try:
            ans = coopr.opt.convert_problem( (arg,), None, [ProblemFormat.cpxlp])
        except ConverterError:
            err = sys.exc_info()[1]
            if pyutilib.services.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-15:],"pico_convert.lp")
        os.remove(ans[0][0])

    def test_mock_mps1(self):
        #""" Convert from Pyomo to MPS """
        arg=MockArg4()
        try:
            ans = coopr.opt.convert_problem( (arg,ProblemFormat.mps,arg), None, [ProblemFormat.mps])
        except ConverterError:
            err = sys.exc_info()[1]
            if pyutilib.services.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-16:],"pico_convert.mps")
        os.remove(ans[0][0])

    def test_pyomo_mps1(self):
        #""" Convert from Pyomo to MPS with file"""
        try:
            ans = coopr.opt.convert_problem( (currdir+'model.py',ProblemFormat.mps,), None, [ProblemFormat.mps])
        except ConverterError:
            err = sys.exc_info()[1]
            if pyutilib.services.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-16:],"pico_convert.mps")
        os.remove(ans[0][0])

    def test_mock_nl1(self):
        #""" Convert from Pyomo to NL """
        arg=MockArg4()
        ans = coopr.opt.convert_problem( (arg,ProblemFormat.nl,arg), None, [ProblemFormat.nl])
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$",ans[0][0]), None)
        os.remove(ans[0][0])

    def test_pyomo_nl1(self):
        #""" Convert from Pyomo to NL with file"""
        ans = coopr.opt.convert_problem( (currdir+'model.py',ProblemFormat.nl,), None, [ProblemFormat.nl])
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$",ans[0][0]), None)
        os.remove(ans[0][0])

    def test_error1(self):
        #""" No valid problem types """
        try:
            coopr.opt.convert_problem( ("test4.nl","tmp.nl"), ProblemFormat.nl, [])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            err = sys.exc_info()[1]
            pass

    def test_error2(self):
        #""" Target problem type is not valid """
        try:
            coopr.opt.convert_problem( ("test4.nl","tmp.nl"), ProblemFormat.nl, [ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error3(self):
        #""" Empty argument list """
        try:
            coopr.opt.convert_problem( (), None, [ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error4(self):
        #""" Unknown source type """
        try:
            coopr.opt.convert_problem( ("prob.foo",), None, [ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error5(self):
        #""" Unknown source type """
        try:
            coopr.opt.convert_problem( ("prob.lp",), ProblemFormat.nl, [ProblemFormat.nl])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error6(self):
        #""" Cannot use pico_convert with more than one file """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test4.nl","foo"), None, [ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error8(self):
        #""" Error when source file cannot be found """
        try:
            ans = coopr.opt.convert_problem( (currdir+"unknown.nl",), None, [ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception")
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if pyutilib.services.registered_executable("pico_convert").enabled():
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
            return
        except ConverterError:
            pass

    def test_error9(self):
        #""" The Opt configuration has not been initialized """
        cmd = pyutilib.services.registered_executable("pico_convert")
        if not cmd is None:
            cmd.disable()
        try:
            ans = coopr.opt.convert_problem( (currdir+"test4.nl",), None, [ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but pico_convert should not be defined.")
        except ConverterError:
            pass
        if not cmd is None:
            cmd.enable()

    def test_error10(self):
        #""" GLPSOL can only convert file data """
        try:
            arg = MockArg3()
            ans = coopr.opt.convert_problem( (arg,ProblemFormat.cpxlp,arg), None, [ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but glpsol cannot handle objects.")
        except ConverterError:
            pass

    def test_error11(self):
        #""" Cannot convert MOD that contains data """
        try:
            ans = coopr.opt.convert_problem( (currdir+"test3.mod",currdir+"test5.dat"), None, [ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception because we provided a MOD file with a 'data;' declaration")
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if pyutilib.services.registered_executable("glpsol").enabled():
                self.fail("Expected ApplicationError because glpsol is not available: '%s'" % str(err))
            return
        except ConverterError:
            pass

if __name__ == "__main__":
    unittest.main()
