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
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________


import glob
import os
import shutil
import sys
from six import StringIO

from os.path import abspath, dirname
currdir = dirname(abspath(__file__)) + os.sep
tempdir = dirname(abspath(__file__)) + os.sep + 'tempdir' + os.sep

import pyutilib.th as unittest

import pyomo.common.tempfiles as tempfiles

from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager

try:
    from pyutilib.component.config.tempfiles import (
        TempfileManager as pyutilib_mngr
    )
except ImportError:
    pyutilib_mngr = None

old_tempdir = TempfileManager.tempdir

class Test(unittest.TestCase):

    def setUp(self):
        TempfileManager.tempdir = tempdir
        TempfileManager.push()
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)
        os.mkdir(tempdir)

    def tearDown(self):
        TempfileManager.pop()
        TempfileManager.tempdir = old_tempdir
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)

    def test_add1(self):
        """Test explicit adding of a file that is missing"""
        try:
            TempfileManager.add_tempfile(tempdir + 'add1')
            self.fail("Expected IOError because file 'add1' does not exist")
        except IOError:
            pass

    def test_add1_dir(self):
        """Test explicit adding of a directory that is missing"""
        try:
            TempfileManager.add_tempfile(tempdir + 'add1')
            self.fail(
                "Expected IOError because directory 'add1' does not exist")
        except IOError:
            pass

    def test_add2(self):
        """Test explicit adding of a file that is missing"""
        TempfileManager.add_tempfile(tempdir + 'add2', False)

    def test_add2_dir(self):
        """Test explicit adding of a directory that is missing"""
        TempfileManager.add_tempfile(tempdir + 'add2', False)

    def test_add3(self):
        """Test explicit adding of a file that already exists"""
        OUTPUT = open(tempdir + 'add3', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        TempfileManager.add_tempfile(tempdir + 'add3')

    def test_add3_dir(self):
        """Test explicit adding of a directory that already exists"""
        os.mkdir(tempdir + 'add3')
        TempfileManager.add_tempfile(tempdir + 'add3')

    def test_pushpop1(self):
        """Test pushpop logic"""
        TempfileManager.push()
        OUTPUT = open(tempdir + 'pushpop1', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        TempfileManager.add_tempfile(tempdir + 'pushpop1')
        TempfileManager.pop()
        if os.path.exists(tempdir + 'pushpop1'):
            self.fail("pop() failed to clean out files")

    def test_pushpop1_dir(self):
        """Test pushpop logic with directories"""
        TempfileManager.push()
        os.mkdir(tempdir + 'pushpop1')
        TempfileManager.add_tempfile(tempdir + 'pushpop1')
        TempfileManager.pop()
        if os.path.exists(tempdir + 'pushpop1'):
            self.fail("pop() failed to clean out directories")

    def test_pushpop2(self):
        """Test pushpop logic"""
        TempfileManager.push()
        OUTPUT = open(tempdir + 'pushpop2', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        TempfileManager.add_tempfile(tempdir + 'pushpop2')

        TempfileManager.push()
        OUTPUT = open(tempdir + 'pushpop2a', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        TempfileManager.add_tempfile(tempdir + 'pushpop2a')
        TempfileManager.pop()
        if not os.path.exists(tempdir + 'pushpop2'):
            self.fail("pop() clean out all files")
        if os.path.exists(tempdir + 'pushpop2a'):
            self.fail("pop() failed to clean out files")

        TempfileManager.pop()
        if os.path.exists(tempdir + 'pushpop2'):
            self.fail("pop() failed to clean out files")

    def test_pushpop2_dir(self):
        """Test pushpop logic with directories"""
        TempfileManager.push()
        os.mkdir(tempdir + 'pushpop2')
        TempfileManager.add_tempfile(tempdir + 'pushpop2')

        TempfileManager.push()
        os.mkdir(tempdir + 'pushpop2a')
        TempfileManager.add_tempfile(tempdir + 'pushpop2a')
        TempfileManager.pop()
        if not os.path.exists(tempdir + 'pushpop2'):
            self.fail("pop() clean out all files")
        if os.path.exists(tempdir + 'pushpop2a'):
            self.fail("pop() failed to clean out files")

        TempfileManager.pop()
        if os.path.exists(tempdir + 'pushpop2'):
            self.fail("pop() failed to clean out files")

    def test_clear(self):
        """Test clear logic"""
        TempfileManager.push()
        OUTPUT = open(tempdir + 'pushpop2', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        os.mkdir(tempdir + 'pushpopdir2')
        TempfileManager.add_tempfile(tempdir + 'pushpop2')
        TempfileManager.add_tempfile(tempdir + 'pushpopdir2')

        TempfileManager.push()
        OUTPUT = open(tempdir + 'pushpop2a', 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        os.mkdir(tempdir + 'pushpopdir2a')
        TempfileManager.add_tempfile(tempdir + 'pushpop2a')
        TempfileManager.add_tempfile(tempdir + 'pushpopdir2a')

        TempfileManager.clear_tempfiles()

        if os.path.exists(tempdir + 'pushpop2a'):
            self.fail("clear_tempfiles() failed to clean out files")
        if os.path.exists(tempdir + 'pushpopdir2a'):
            self.fail("clear_tempfiles() failed to clean out directories")
        if os.path.exists(tempdir + 'pushpop2'):
            self.fail("clear_tempfiles() failed to clean out files")
        if os.path.exists(tempdir + 'pushpopdir2'):
            self.fail("clear_tempfiles() failed to clean out directories")

    def test_create1(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempfile()
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('tmp'))

    def test_create1_dir(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempdir()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('tmp'))

    def test_create1a(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempfile(dir=tempdir)
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('tmp'))

    def test_create1a_dir(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempdir(dir=tempdir)
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('tmp'))

    def test_create2(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempfile(prefix='foo')
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('foo'))

    def test_create2_dir(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempdir(prefix='foo')
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.startswith('foo'))

    def test_create3(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempfile(suffix='bar')
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.endswith('bar'))

    def test_create3_dir(self):
        """Test create logic - no options"""
        fname = TempfileManager.create_tempdir(suffix='bar')
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertTrue(fname.endswith('bar'))

    def test_create4(self):
        """Test create logic - no options"""
        TempfileManager.sequential_files(2)
        fname = TempfileManager.create_tempfile()
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertEqual(fname, 'tmp2')
        #
        TempfileManager.unique_files()
        fname = TempfileManager.create_tempfile()
        OUTPUT = open(fname, 'w')
        OUTPUT.write('tempfile\n')
        OUTPUT.close()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 2)
        fname = os.path.basename(fname)
        self.assertNotEqual(fname, 'tmp3')
        self.assertTrue(fname.startswith('tmp'))

    def test_create4_dir(self):
        """Test create logic - no options"""
        TempfileManager.sequential_files(2)
        fname = TempfileManager.create_tempdir()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 1)
        fname = os.path.basename(fname)
        self.assertEqual(fname, 'tmp2')
        #
        TempfileManager.unique_files()
        fname = TempfileManager.create_tempdir()
        self.assertEqual(len(list(glob.glob(tempdir + '*'))), 2)
        fname = os.path.basename(fname)
        self.assertNotEqual(fname, 'tmp3')
        self.assertTrue(fname.startswith('tmp'))

    @unittest.skipIf(not sys.platform.lower().startswith('win'),
                     "test only applies to Windows platforms")
    def test_open_tempfile_windows(self):
        TempfileManager.push()
        fname = TempfileManager.create_tempfile()
        f = open(fname)
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = True
            with self.assertRaisesRegex(
                    WindowsError, ".*process cannot access the file"):
                TempfileManager.pop()
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)

        TempfileManager.push()
        fname = TempfileManager.create_tempfile()
        f = open(fname)
        log = StringIO()
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = False
            with LoggingIntercept(log, 'pyomo.common'):
                TempfileManager.pop()
            self.assertIn("Unable to delete temporary file", log.getvalue())
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)

    @unittest.skipIf(pyutilib_mngr is None,
                     "deprecation test requires pyutilib")
    def test_deprecated_tempdir(self):
        TempfileManager.push()
        try:
            tmpdir = TempfileManager.create_tempdir()
            _orig = pyutilib_mngr.tempdir
            pyutilib_mngr.tempdir = tmpdir
            TempfileManager.tempdir = None

            log = StringIO()
            with LoggingIntercept(log, 'pyomo.core'):
                fname = TempfileManager.create_tempfile()
            self.assertIn(
                "The use of the PyUtilib TempfileManager.tempdir "
                "to specify the default location for Pyomo "
                "temporary files", log.getvalue().replace("\n", " "))

            log = StringIO()
            with LoggingIntercept(log, 'pyomo.core'):
                dname = TempfileManager.create_tempdir()
            self.assertIn(
                "The use of the PyUtilib TempfileManager.tempdir "
                "to specify the default location for Pyomo "
                "temporary directories", log.getvalue().replace("\n", " "))
        finally:
            TempfileManager.pop()
            pyutilib_mngr.tempdir = _orig

if __name__ == "__main__":
    unittest.main()
