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

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager

import pyomo.opt

old_tempdir = TempfileManager.tempdir

class MockArg(object):

    def __init__(self):
        pass

    def valid_problem_types(self):
        return [pyomo.opt.ProblemFormat.pyomo]

    def write(self,filename="", format=None):
        pass

class MockArg2(MockArg):

    def valid_problem_types(self):
        return [pyomo.opt.ProblemFormat.nl]

    def write(self,filename="", format=None):
        OUTPUT=open(filename,"w")
        INPUT=open(currdir+"test4.nl")
        for line in INPUT:
            print >>OUTPUT, line,
        OUTPUT.close()
        INPUT.close()

class MockArg3(MockArg):

    def valid_problem_types(self):
        return [pyomo.opt.ProblemFormat.mod]

    def write(self,filename="", format=None):
        pass

class MockArg4(MockArg):

    def write(self,filename="", format=None):
        OUTPUT=open(filename,"w")
        INPUT=open(currdir+"test4.nl")
        for line in INPUT:
            print >>OUTPUT, line,
        OUTPUT.close()
        INPUT.close()


class OptConvertDebug(unittest.TestCase):

    def setUp(self):
        TempfileManager.tempdir = currdir

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test_nl_nl1(self):
        """ Convert from NL to NL """
        ans = pyomo.opt.convert_problem( ("test4.nl",), None, [pyomo.opt.ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl",))

    def test_nl_nl2(self):
        """ Convert from NL to NL """
        ans = pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), None, [pyomo.opt.ProblemFormat.nl])
        self.assertEqual(ans[0],("test4.nl","tmp.nl"))

    def test_error1(self):
        """ No valid problem types """
        try:
            pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), pyomo.opt.ProblemFormat.nl, [])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error2(self):
        """ Target problem type is not valid """
        try:
            pyomo.opt.convert_problem( ("test4.nl","tmp.nl"), pyomo.opt.ProblemFormat.nl, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error3(self):
        """ Empty argument list """
        try:
            pyomo.opt.convert_problem( (), None, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error4(self):
        """ Unknown source type """
        try:
            pyomo.opt.convert_problem( ("prob.foo",), None, [pyomo.opt.ProblemFormat.mps])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error5(self):
        """ Unknown source type """
        try:
            pyomo.opt.convert_problem( ("prob.lp",), pyomo.opt.ProblemFormat.nl, [pyomo.opt.ProblemFormat.nl])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error6(self):
        """ Cannot use pico_convert with more than one file """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test4.nl","foo"), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except pyomo.opt.ConverterError:
            pass

    def test_error8(self):
        """ Error when source file cannot be found """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"unknown.nl",), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected pyomo.opt.ConverterError exception")
        except ApplicationError:
            if pyomo.common.Executable("pico_convert"):
                self.fail("Expected ApplicationError because pico_convert is not available")
            return
        except pyomo.opt.ConverterError:
            pass

    def test_error9(self):
        """ The Opt configuration has not been initialized """
        cmd = pyomo.common.Executable("pico_convert")
        if cmd:
            cmd.disable()
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test4.nl",), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but pico_convert should not be defined.")
        except pyomo.opt.ConverterError:
            pass
        cmd.rehash()

    def test_error10(self):
        """ GLPSOL can only convert file data """
        try:
            arg = MockArg3()
            ans = pyomo.opt.convert_problem( (arg,pyomo.opt.ProblemFormat.cpxlp,arg), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("This test didn't fail, but glpsol cannot handle objects.")
        except pyomo.opt.ConverterError:
            pass

    def test_error11(self):
        """ Cannot convert MOD that contains data """
        try:
            ans = pyomo.opt.convert_problem( (currdir+"test3.mod",currdir+"test5.dat"), None, [pyomo.opt.ProblemFormat.cpxlp])
            self.fail("Expected pyomo.opt.ConverterError exception because we provided a MOD file with a 'data;' declaration")
        except ApplicationError:
            if pyomo.common.Executable("glpsol"):
                self.fail("Expected ApplicationError because glpsol is not available")
            return
        except pyomo.opt.ConverterError:
            pass

if __name__ == "__main__":
    unittest.main()
