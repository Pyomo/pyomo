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
# Unit Tests for pyomo.opt.problem.ampl
#

from itertools import zip_longest
import json
import os
import sys
from os.path import join
from filecmp import cmp

import pyomo.common.unittest as unittest

from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

import pyomo.opt

currdir = this_file_dir()
deleteFiles = True


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ

        solvers = pyomo.opt.check_available_solvers('glpk')

    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def test3_write_nl(self):
        """Convert from AMPL to NL"""
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        """ Convert from MOD+DAT to NL """
        _test = TempfileManager.create_tempfile(suffix='test3.nl')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ApplicationError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ConverterError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.nl')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('# problem',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3_write_lp(self):
        """Convert from AMPL to LP"""
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        _test = TempfileManager.create_tempfile(suffix='test3.lp')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail(
                    "Unexpected ApplicationError - glpsol is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail(
                    "Unexpected ConverterError - glpsol is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.lp')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('Problem:',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3_write_mps(self):
        """Convert from AMPL to MPS"""
        if not pyomo.common.Executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        _test = TempfileManager.create_tempfile(suffix='test3.mps')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ApplicationError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ConverterError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.mps')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('NAME',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3a_write_nl(self):
        """Convert from AMPL to NL"""
        self.model = pyomo.opt.AmplModel(
            join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat')
        )
        _test = TempfileManager.create_tempfile(suffix='test3a.nl')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ApplicationError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ConverterError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.nl')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('# problem',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3a_write_lp(self):
        """Convert from AMPL to LP"""
        self.model = pyomo.opt.AmplModel(
            join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat')
        )
        _test = TempfileManager.create_tempfile(suffix='test3a.lp')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail(
                    "Unexpected ApplicationError - glpsol is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail(
                    "Unexpected ConverterError - glpsol is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.lp')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('Problem:',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3a_write_mps(self):
        """Convert from AMPL to MPS"""
        if not pyomo.common.Executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(
            join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat')
        )
        _test = TempfileManager.create_tempfile(suffix='test3a.mps')
        try:
            self.model.write(_test)
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ApplicationError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail(
                    "Unexpected ConverterError - ampl is enabled "
                    "but not available: '%s'" % str(err)
                )
            return
        _base = join(currdir, 'test3.baseline.mps')
        with open(_test, 'r') as run, open(_base, 'r') as baseline:
            for line1, line2 in zip_longest(run, baseline):
                for _pattern in ('NAME',):
                    if line1.find(_pattern) >= 0:
                        line1 = line1[: line1.find(_pattern) + len(_pattern)]
                        line2 = line2[: line2.find(_pattern) + len(_pattern)]
                self.assertEqual(
                    line1, line2, msg="Files %s and %s differ" % (_test, _base)
                )

    def test3_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        opt = pyomo.opt.SolverFactory('glpk')
        _test = TempfileManager.create_tempfile(suffix='test3.out')
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=_test, format='json')
        with open(_test, 'r') as out, open(
            join(currdir, "test3.baseline.out"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-6, allow_second_superset=True
            )

    def test3a_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(
            join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat')
        )
        opt = pyomo.opt.SolverFactory('glpk')
        results = opt.solve(self.model, keepfiles=False)
        _test = TempfileManager.create_tempfile(suffix='test3a.out')
        results.write(filename=_test, format='json')
        with open(_test, 'r') as out, open(
            join(currdir, "test3.baseline.out"), 'r'
        ) as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-6, allow_second_superset=True
            )


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
