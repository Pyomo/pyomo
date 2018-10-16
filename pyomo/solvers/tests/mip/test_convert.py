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
# Unit Tests for pyomo.opt.base.convert
#

import re
import sys
import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
import pyutilib.common

import pyomo.opt

def filter(line):
    return 'Problem' in line or line.startswith('NAME')

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = pyutilib.services.TempfileManager.tempdir
    pyutilib.services.TempfileManager.tempdir = currdir

def tearDownModule():
    pyutilib.services.TempfileManager.tempdir = old_tempdir

class MockArg(object):

    def __init__(self):
        pass

    def valid_problem_types(self):
        return [pyomo.opt.ProblemFormat.pyomo]

    def write(self,filename="", format=None, solver_capability=None, io_options={}):
        return (filename,None)

class MockArg2(MockArg):

    def valid_problem_types(self):
        return [pyomo.opt.ProblemFormat.nl]

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
        return [pyomo.opt.ProblemFormat.mod]

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

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()

    def test_nl_nl1(self):
        #""" Convert from NL to NL """
        ans = pyomo.opt.convert_problem( ("test4.nl",), None, [pyomo.opt.ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl",))

    def test_nl_nl2(self):
        #""" Convert from NL to NL """
        ans = pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), None, [pyomo.opt.ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl","tmp.nl"))

    def test_nl_lp1(self):
        #""" Convert from NL to LP """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test4.nl",), None, [pyomo.opt.ProblemFormat.cpxlp])
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("pico_convert") is None:
                self.fail("Unexpected ApplicationError - pico_convert is enabled but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if not pyomo.common.registered_executable("pico_convert") is None:
                self.fail("Unexpected ConverterError - pico_convert is enabled but not available: '%s'" % str(err))
            return
        self.assertEqual(ans[0][0][-15:],"pico_convert.lp")
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test1_convert.lp")

    def test_mod_lp1(self):
        #""" Convert from MOD to LP """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test3.mod",), None, [pyomo.opt.ProblemFormat.cpxlp])
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
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test2_convert.lp", filter=filter)

    def test_mod_lp2(self):
        #""" Convert from MOD+DAT to LP """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test5.mod",currdir+"test5.dat"), None, [pyomo.opt.ProblemFormat.cpxlp])
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
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        self.assertFileEqualsBaseline(ans[0][0], currdir+"test3_convert.lp", filter=filter)

    def test_mod_nl1(self):
        #""" Convert from MOD to NL """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test3.mod",), None, [pyomo.opt.ProblemFormat.nl])
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
        self.assertTrue(ans[0][0].endswith('.nl'))
        #self.assertFileEqualsBinaryFile(ans[0][0], currdir+"test_mod_nl1.nl")

    def test_mod_nl2(self):
        #""" Convert from MOD+DAT to NL """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test5.mod",currdir+"test5.dat"), None, [pyomo.opt.ProblemFormat.nl])
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
        self.assertTrue(ans[0][0].endswith('.nl'))
        #self.assertFileEqualsBaseline(ans[0][0], currdir+"test_mod_nl2.nl")

    def test_mock_lp1(self):
        #""" Convert from Pyomo to LP """
        arg=MockArg()
        ans = pyomo.opt.convert_problem( (arg,pyomo.opt.ProblemFormat.cpxlp,arg), None, [pyomo.opt.ProblemFormat.cpxlp])
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$",ans[0][0]), None)

    def test_pyomo_lp1(self):
        #""" Convert from Pyomo to LP with file"""
        ans = pyomo.opt.convert_problem( (currdir+'model.py',pyomo.opt.ProblemFormat.cpxlp,), None, [pyomo.opt.ProblemFormat.cpxlp])
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$",ans[0][0]), None)

    def test_mock_lp2(self):
        #""" Convert from NL to LP """
        arg=MockArg2()
        try:
            ans = pyomo.opt.convert_problem( (arg,), None, [pyomo.opt.ProblemFormat.cpxlp])
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-15:],"pico_convert.lp")
        os.remove(ans[0][0])

    # Note sure what to do with this test now that we
    # have a native MPS converter
    def Xtest_mock_mps1(self):
        #""" Convert from Pyomo to MPS """
        arg=MockArg4()
        try:
            ans = pyomo.opt.convert_problem( (arg,pyomo.opt.ProblemFormat.mps,arg), None, [pyomo.opt.ProblemFormat.mps])
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-16:],"pico_convert.mps")
        os.remove(ans[0][0])

    def test_pyomo_mps1(self):
        #""" Convert from Pyomo to MPS with file"""
        try:
            ans = pyomo.opt.convert_problem( (currdir+'model.py',pyomo.opt.ProblemFormat.mps,), None, [pyomo.opt.ProblemFormat.mps])
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.registered_executable("pico_convert") is None:
                return
            else:
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
        self.assertEqual(ans[0][0][-16:],"pico_convert.mps")
        os.remove(ans[0][0])

    def test_mock_nl1(self):
        #""" Convert from Pyomo to NL """
        arg=MockArg4()
        ans = pyomo.opt.convert_problem( (arg,pyomo.opt.ProblemFormat.nl,arg), None, [pyomo.opt.ProblemFormat.nl])
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$",ans[0][0]), None)
        os.remove(ans[0][0])

    def test_pyomo_nl1(self):
        #""" Convert from Pyomo to NL with file"""
        ans = pyomo.opt.convert_problem( (currdir+'model.py',pyomo.opt.ProblemFormat.nl,), None, [pyomo.opt.ProblemFormat.nl])
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$",ans[0][0]), None)
        os.remove(ans[0][0])

    def test_error1(self):
        #""" No valid problem types """
        try:
            pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), pyomo.opt.ProblemFormat.nl, [])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            pass

    def test_error2(self):
        #""" Target problem type is not valid """
        try:
            pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), pyomo.opt.ProblemFormat.nl, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error3(self):
        #""" Empty argument list """
        try:
            pyomo.opt.convert_problem( (), None, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error4(self):
        #""" Unknown source type """
        try:
            pyomo.opt.convert_problem( ("prob.foo",), None, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error5(self):
        #""" Unknown source type """
        try:
            pyomo.opt.convert_problem( ("prob.lp",), pyomo.opt.ProblemFormat.nl, [pyomo.opt.ProblemFormat.nl])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error6(self):
        #""" Cannot use pico_convert with more than one file """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test4.nl","foo"), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error8(self):
        #""" Error when source file cannot be found """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"unknown.nl",), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception")
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.registered_executable("pico_convert").enabled():
                self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            pass

    def test_error9(self):
        #""" The Opt configuration has not been initialized """
        cmd = pyomo.common.unregister_executable("pico_convert")
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test4.nl",), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but pico_convert should not be defined.")
        except pyomo.opt.ConverterError:
            pass
        cmd = pyomo.common.register_executable("pico_convert")

    def test_error10(self):
        #""" GLPSOL can only convert file data """
        try:
            arg = MockArg3()
            ans = pyomo.opt.convert_problem( (arg,pyomo.opt.ProblemFormat.cpxlp,arg), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but glpsol cannot handle objects.")
        except pyomo.opt.ConverterError:
            pass

    def test_error11(self):
        #""" Cannot convert MOD that contains data """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test3.mod",currdir+"test5.dat"), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected ConverterError exception because we provided a MOD file with a 'data;' declaration")
        except pyutilib.common.ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.registered_executable("glpsol").enabled():
                self.fail("Expected ApplicationError because glpsol is not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            pass

if __name__ == "__main__":
    unittest.main()
