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
#
# Unit Tests for pyomo.opt.base.convert
#

from itertools import zip_longest
import re
import sys
import os
from os.path import join
from filecmp import cmp

import pyomo.common.unittest as unittest

from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt import ProblemFormat, ConverterError, convert_problem
from pyomo.common import Executable


def filter(line):
    return 'Problem' in line or line.startswith('NAME')


currdir = this_file_dir()
deleteFiles = True


class MockArg(object):
    def __init__(self):
        pass

    def valid_problem_types(self):
        return [ProblemFormat.pyomo]

    def write(self, filename="", format=None, solver_capability=None, io_options={}):
        return (filename, None)


class MockArg2(MockArg):
    def valid_problem_types(self):
        return [ProblemFormat.nl]

    def write(self, filename="", format=None, solver_capability=None, io_options={}):
        OUTPUT = open(filename, "w")
        INPUT = open(join(currdir, "test4.nl"))
        for line in INPUT:
            OUTPUT.write(line)
        OUTPUT.close()
        INPUT.close()
        return (filename, None)


class MockArg3(MockArg):
    def valid_problem_types(self):
        return [ProblemFormat.mod]

    def write(self, filename="", format=None, solver_capability=None, io_options={}):
        return (filename, None)


class MockArg4(MockArg):
    def write(self, filename="", format=None, solver_capability=None, io_options={}):
        OUTPUT = open(filename, "w")
        INPUT = open(join(currdir, "test4.nl"))
        for line in INPUT:
            OUTPUT.write(line)
        OUTPUT.close()
        INPUT.close()
        return (filename, None)


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def test_nl_nl1(self):
        # """ Convert from NL to NL """
        ans = convert_problem(("test4.nl",), None, [ProblemFormat.nl])
        self.assertEqual(ans[0], ("test4.nl",))

    def test_nl_nl2(self):
        # """ Convert from NL to NL """
        ans = convert_problem(("test4.nl", "tmp.nl"), None, [ProblemFormat.nl])
        self.assertEqual(ans[0], ("test4.nl", "tmp.nl"))

    @unittest.skipUnless(
        Executable("pico_convert").available(), 'pico_convert required'
    )
    def test_nl_lp1(self):
        # """ Convert from NL to LP """
        ans = convert_problem((join(currdir, "test4.nl"),), None, [ProblemFormat.cpxlp])
        self.assertEqual(ans[0][0][-15:], "pico_convert.lp")
        _out, _log = ans[0][0], join(currdir, "test1_convert.lp")
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))

    @unittest.skipUnless(Executable("glpsol").available(), 'glpsol required')
    def test_mod_lp1(self):
        # """ Convert from MOD to LP """
        ans = convert_problem(
            (join(currdir, "test3.mod"),), None, [ProblemFormat.cpxlp]
        )
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        with open(ans[0][0], 'r') as f1, open(
            join(currdir, "test2_convert.lp"), 'r'
        ) as f2:
            for line1, line2 in zip_longest(f1, f2):
                if 'Problem' in line1:
                    continue
                self.assertEqual(line1, line2)

    @unittest.skipUnless(Executable("glpsol").available(), 'glpsol required')
    def test_mod_lp2(self):
        # """ Convert from MOD+DAT to LP """
        ans = convert_problem(
            (join(currdir, "test5.mod"), join(currdir, "test5.dat")),
            None,
            [ProblemFormat.cpxlp],
        )
        self.assertTrue(ans[0][0].endswith("glpsol.lp"))
        with open(ans[0][0], 'r') as f1, open(
            join(currdir, "test3_convert.lp"), 'r'
        ) as f2:
            for line1, line2 in zip_longest(f1, f2):
                if 'Problem' in line1:
                    continue
                self.assertEqual(line1, line2)

    @unittest.skipUnless(Executable("ampl").available(), 'ampl required')
    def test_mod_nl1(self):
        # """ Convert from MOD to NL """
        ans = convert_problem((join(currdir, "test3.mod"),), None, [ProblemFormat.nl])
        self.assertTrue(ans[0][0].endswith('.nl'))
        # self.assertFileEqualsBinaryFile(ans[0][0], join(currdir, "test_mod_nl1.nl")

    @unittest.skipUnless(Executable("ampl").available(), 'ampl required')
    def test_mod_nl2(self):
        # """ Convert from MOD+DAT to NL """
        ans = convert_problem(
            (join(currdir, "test5.mod"), join(currdir, "test5.dat")),
            None,
            [ProblemFormat.nl],
        )
        self.assertTrue(ans[0][0].endswith('.nl'))
        # self.assertTrue(cmp(ans[0][0], join(currdir, "test_mod_nl2.nl")

    def test_mock_lp1(self):
        # """ Convert from Pyomo to LP """
        arg = MockArg()
        ans = convert_problem(
            (arg, ProblemFormat.cpxlp, arg), None, [ProblemFormat.cpxlp]
        )
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$", ans[0][0]), None)

    def test_pyomo_lp1(self):
        # """ Convert from Pyomo to LP with file"""
        ans = convert_problem(
            (join(currdir, 'model.py'), ProblemFormat.cpxlp),
            None,
            [ProblemFormat.cpxlp],
        )
        self.assertNotEqual(re.match(".*tmp.*pyomo.lp$", ans[0][0]), None)

    def test_mock_lp2(self):
        # """ Convert from NL to LP """
        arg = MockArg2()
        try:
            ans = convert_problem((arg,), None, [ProblemFormat.cpxlp])
        except ConverterError:
            err = sys.exc_info()[1]
            if not Executable("pico_convert"):
                return
            else:
                self.fail(
                    "Expected ApplicationError because pico_convert "
                    "is not available: '%s'" % str(err)
                )
        self.assertEqual(ans[0][0][-15:], "pico_convert.lp")
        os.remove(ans[0][0])

    # Note sure what to do with this test now that we
    # have a native MPS converter
    def Xtest_mock_mps1(self):
        # """ Convert from Pyomo to MPS """
        arg = MockArg4()
        try:
            ans = convert_problem(
                (arg, ProblemFormat.mps, arg), None, [ProblemFormat.mps]
            )
        except ConverterError:
            err = sys.exc_info()[1]
            if not Executable("pico_convert"):
                return
            else:
                self.fail(
                    "Expected ApplicationError because pico_convert "
                    "is not available: '%s'" % str(err)
                )
        self.assertEqual(ans[0][0][-16:], "pico_convert.mps")
        os.remove(ans[0][0])

    def test_pyomo_mps1(self):
        # """ Convert from Pyomo to MPS with file"""
        try:
            ans = convert_problem(
                (join(currdir, 'model.py'), ProblemFormat.mps),
                None,
                [ProblemFormat.mps],
            )
        except ConverterError:
            err = sys.exc_info()[1]
            if not Executable("pico_convert"):
                return
            else:
                self.fail(
                    "Expected ApplicationError because pico_convert "
                    "is not available: '%s'" % str(err)
                )
        self.assertEqual(ans[0][0][-16:], "pico_convert.mps")
        os.remove(ans[0][0])

    def test_mock_nl1(self):
        # """ Convert from Pyomo to NL """
        arg = MockArg4()
        ans = convert_problem((arg, ProblemFormat.nl, arg), None, [ProblemFormat.nl])
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$", ans[0][0]), None)
        os.remove(ans[0][0])

    def test_pyomo_nl1(self):
        # """ Convert from Pyomo to NL with file"""
        ans = convert_problem(
            (join(currdir, 'model.py'), ProblemFormat.nl), None, [ProblemFormat.nl]
        )
        self.assertNotEqual(re.match(".*tmp.*pyomo.nl$", ans[0][0]), None)
        os.remove(ans[0][0])

    def test_error1(self):
        # """ No valid problem types """
        try:
            convert_problem(("test4.nl", "tmp.nl"), ProblemFormat.nl, [])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            err = sys.exc_info()[1]
            pass

    def test_error2(self):
        # """ Target problem type is not valid """
        try:
            convert_problem(
                ("test4.nl", "tmp.nl"), ProblemFormat.nl, [ProblemFormat.mps]
            )
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error3(self):
        # """ Empty argument list """
        try:
            convert_problem((), None, [ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error4(self):
        # """ Unknown source type """
        try:
            convert_problem(("prob.foo",), None, [ProblemFormat.mps])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error5(self):
        # """ Unknown source type """
        try:
            convert_problem(("prob.lp",), ProblemFormat.nl, [ProblemFormat.nl])
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error6(self):
        # """ Cannot use pico_convert with more than one file """
        try:
            ans = convert_problem(
                (join(currdir, "test4.nl"), "foo"), None, [ProblemFormat.cpxlp]
            )
            self.fail("Expected ConverterError exception")
        except ConverterError:
            pass

    def test_error8(self):
        # """ Error when source file cannot be found """
        try:
            ans = convert_problem(
                (join(currdir, "unknown.nl"),), None, [ProblemFormat.cpxlp]
            )
            self.fail("Expected ConverterError exception")
        except ApplicationError:
            err = sys.exc_info()[1]
            if not Executable("pico_convert"):
                self.fail(
                    "Expected ApplicationError because pico_convert "
                    "is not available: '%s'" % str(err)
                )
            return
        except ConverterError:
            pass

    def test_error9(self):
        # """ The Opt configuration has not been initialized """
        cmd = Executable("pico_convert").disable()
        try:
            ans = convert_problem(
                (join(currdir, "test4.nl"),), None, [ProblemFormat.cpxlp]
            )
            self.fail("This test didn't fail, but pico_convert should not be defined.")
        except ConverterError:
            pass
        cmd = Executable("pico_convert").rehash()

    def test_error10(self):
        # """ GLPSOL can only convert file data """
        try:
            arg = MockArg3()
            ans = convert_problem(
                (arg, ProblemFormat.cpxlp, arg), None, [ProblemFormat.cpxlp]
            )
            self.fail("This test didn't fail, but glpsol cannot handle objects.")
        except ConverterError:
            pass

    def test_error11(self):
        # """ Cannot convert MOD that contains data """
        try:
            ans = convert_problem(
                (join(currdir, "test3.mod"), join(currdir, "test5.dat")),
                None,
                [ProblemFormat.cpxlp],
            )
            self.fail(
                "Expected ConverterError exception because we provided a MOD file with a 'data;' declaration"
            )
        except ApplicationError:
            err = sys.exc_info()[1]
            if Executable("glpsol"):
                self.fail(
                    "Expected ApplicationError because glpsol "
                    "is not available: '%s'" % str(err)
                )
            return
        except ConverterError:
            pass


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
