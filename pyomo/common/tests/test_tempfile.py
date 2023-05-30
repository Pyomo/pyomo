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
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO

from os.path import abspath, dirname

import pyomo.common.unittest as unittest

import pyomo.common.tempfiles as tempfiles

from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
    TempfileManager,
    TempfileManagerClass,
    TempfileContextError,
)

try:
    from pyutilib.component.config.tempfiles import TempfileManager as pyutilib_mngr
except ImportError:
    pyutilib_mngr = None

old_tempdir = TempfileManager.tempdir
tempdir = None


class Test_LegacyTestSuite(unittest.TestCase):
    def setUp(self):
        global tempdir
        tempdir = tempfile.mkdtemp() + os.sep
        TempfileManager.tempdir = tempdir
        TempfileManager.push()

    def tearDown(self):
        global tempdir
        TempfileManager.pop()
        TempfileManager.tempdir = old_tempdir
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)
        tempdir = None

    def test_add1(self):
        """Test explicit adding of a file that is missing"""
        try:
            TempfileManager.add_tempfile(tempdir + 'add1')
            self.fail("Expected IOError because file 'add1' does not exist")
        except IOError:
            pass

    def test_add2(self):
        """Test explicit adding of a file that is missing"""
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
        # Push a new context onto the stack so that the tearDown succeeds
        TempfileManager.push()

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


class Test_TempfileManager(unittest.TestCase):
    def setUp(self):
        self.TM = TempfileManagerClass()

    def tearDown(self):
        self.TM.shutdown()

    def test_create_tempfile(self):
        context = self.TM.push()
        fname = self.TM.create_tempfile("suffix", "prefix")
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isfile(fname))
        context.release()
        self.assertFalse(os.path.exists(fname))

    def test_mkstemp(self):
        context = self.TM.new_context()
        fd, fname = context.mkstemp("suffix", "prefix")
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isfile(fname))
        # calling fsync should run without error
        os.fsync(fd)
        context.release()
        self.assertFalse(os.path.exists(fname))
        # calling fsync should error, as the file descriptor was closed
        with self.assertRaises(OSError):
            os.fsync(fd)

        context = self.TM.new_context()
        fd, fname = context.mkstemp("suffix", "prefix")
        # closing the fd should not result in an error when the context
        # is released
        os.close(fd)
        context.release()

    def test_create_tempdir(self):
        context = self.TM.push()
        fname = self.TM.create_tempdir("suffix", "prefix")
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isdir(fname))
        context.release()
        self.assertFalse(os.path.exists(fname))

    def test_add_tempfile(self):
        context1 = self.TM.push()
        context2 = self.TM.push()
        fname = context1.create_tempfile()
        dname = context1.create_tempdir()
        sub_fname = os.path.join(dname, "testfile")
        self.TM.add_tempfile(fname)
        with self.assertRaisesRegex(
            IOError,
            "Temporary file does not exist: %s" % sub_fname.replace('\\', '\\\\'),
        ):
            self.TM.add_tempfile(sub_fname)
        self.TM.add_tempfile(sub_fname, exists=False)
        with open(sub_fname, "w") as FILE:
            FILE.write("\n")

        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(dname))
        self.assertTrue(os.path.exists(sub_fname))
        self.TM.pop()
        self.assertFalse(os.path.exists(fname))
        self.assertTrue(os.path.exists(dname))
        self.assertFalse(os.path.exists(sub_fname))

        # Releasing a context that is missing files should be OK (no error)
        self.TM.pop()
        self.assertFalse(os.path.exists(fname))
        self.assertFalse(os.path.exists(dname))
        self.assertFalse(os.path.exists(sub_fname))

    def test_sequential_files(self):
        with LoggingIntercept() as LOG:
            self.assertIsNone(self.TM.sequential_files())
        self.assertIn(
            "The TempfileManager.sequential_files() method has been removed",
            LOG.getvalue().replace('\n', ' '),
        )
        self.assertIsNone(self.TM.unique_files())

    def test_gettempprefix(self):
        ctx = self.TM.new_context()
        pre = ctx.gettempprefix()
        self.assertIsInstance(pre, str)
        self.assertEqual(pre, tempfile.gettempprefix())

        preb = ctx.gettempprefixb()
        self.assertIsInstance(preb, bytes)
        self.assertEqual(preb, tempfile.gettempprefixb())

    def test_gettempdir(self):
        context = self.TM.push()
        # Creating in the system TMP
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        system_tmpdir = os.path.dirname(fname)
        self.assertEqual(system_tmpdir, tempfile.gettempdir())
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, system_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)

        # Creating in a TMP specified on the TempfileManager
        manager_tmpdir = context.create_tempdir()
        self.assertNotEqual(manager_tmpdir, system_tmpdir)
        self.TM.tempdir = manager_tmpdir
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, manager_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)

        # Creating in a TMP specified on the context
        context_tmpdir = context.create_tempdir()
        self.assertNotEqual(context_tmpdir, system_tmpdir)
        self.assertNotEqual(context_tmpdir, manager_tmpdir)
        context.tempdir = context_tmpdir
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, context_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)

        # Creating in a TMP specified on the context ... but in bytes
        context.tempdir = context_tmpdir.encode()
        fname = context.create_tempfile()
        self.assertIsInstance(fname, bytes)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, context_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)
        # Cleanup
        self.TM.pop()

    def test_shutdown(self):
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(LOG.getvalue(), "")

        self.TM = TempfileManagerClass()
        ctx = self.TM.push()
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(
            LOG.getvalue().strip(),
            "TempfileManagerClass instance: un-popped tempfile "
            "contexts still exist during TempfileManager instance "
            "shutdown",
        )

        self.TM = TempfileManagerClass()
        ctx = self.TM.push()
        fname = ctx.create_tempfile()
        self.assertTrue(os.path.exists(fname))
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(
            LOG.getvalue().strip(),
            "Temporary files created through TempfileManager "
            "contexts have not been deleted (observed during "
            "TempfileManager instance shutdown).\n"
            "Undeleted entries:\n\t%s\n"
            "TempfileManagerClass instance: un-popped tempfile "
            "contexts still exist during TempfileManager instance "
            "shutdown" % fname,
        )

        # The TM is already shut down, so this should be a noop
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(LOG.getvalue(), "")

    def test_del_clears_contexts(self):
        TM = TempfileManagerClass()
        ctx = TM.push()
        fname = ctx.create_tempfile()
        self.assertTrue(os.path.exists(fname))
        with LoggingIntercept() as LOG:
            TM = None
            gc.collect()
            gc.collect()
            gc.collect()
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(
            LOG.getvalue().strip(),
            "Temporary files created through TempfileManager "
            "contexts have not been deleted (observed during "
            "TempfileManager instance shutdown).\n"
            "Undeleted entries:\n\t%s\n"
            "TempfileManagerClass instance: un-popped tempfile "
            "contexts still exist during TempfileManager instance "
            "shutdown" % fname,
        )

    def test_tempfilemanager_as_context_manager(self):
        with LoggingIntercept() as LOG:
            with self.TM:
                fname = self.TM.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(LOG.getvalue(), "")

            with self.TM:
                self.TM.push()
                fname = self.TM.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(
                LOG.getvalue().strip(),
                "TempfileManager: tempfile context was pushed onto "
                "the TempfileManager stack within a context manager "
                "(i.e., `with TempfileManager:`) but was not popped "
                "before the context manager exited.  Popping the "
                "context to preserve the stack integrity.",
            )

    def test_tempfilecontext_as_context_manager(self):
        with LoggingIntercept() as LOG:
            ctx = self.TM.new_context()
            with ctx:
                fname = ctx.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(LOG.getvalue(), "")

    @unittest.skipIf(
        not sys.platform.lower().startswith('win'),
        "test only applies to Windows platforms",
    )
    def test_open_tempfile_windows(self):
        self.TM.push()
        fname = self.TM.create_tempfile()
        f = open(fname)
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = True
            with self.assertRaisesRegex(
                WindowsError, ".*process cannot access the file"
            ):
                self.TM.pop()
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)

        self.TM.push()
        fname = self.TM.create_tempfile()
        f = open(fname)
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = False
            with LoggingIntercept(None, 'pyomo.common') as LOG:
                self.TM.pop()
            self.assertIn("Unable to delete temporary file", LOG.getvalue())
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)

    @unittest.skipIf(pyutilib_mngr is None, "deprecation test requires pyutilib")
    def test_deprecated_tempdir(self):
        self.TM.push()
        try:
            tmpdir = self.TM.create_tempdir()
            _orig = pyutilib_mngr.tempdir
            pyutilib_mngr.tempdir = tmpdir
            self.TM.tempdir = None

            with LoggingIntercept() as LOG:
                fname = self.TM.create_tempfile()
            self.assertIn(
                "The use of the PyUtilib TempfileManager.tempdir "
                "to specify the default location for Pyomo "
                "temporary files",
                LOG.getvalue().replace("\n", " "),
            )

            with LoggingIntercept() as LOG:
                dname = self.TM.create_tempdir()
            self.assertIn(
                "The use of the PyUtilib TempfileManager.tempdir "
                "to specify the default location for Pyomo "
                "temporary files",
                LOG.getvalue().replace("\n", " "),
            )
        finally:
            self.TM.pop()
            pyutilib_mngr.tempdir = _orig

    def test_context(self):
        with self.assertRaisesRegex(
            TempfileContextError, "TempfileManager has no currently active context"
        ):
            self.TM.context()
        ctx = self.TM.push()
        self.assertIs(ctx, self.TM.context())


if __name__ == "__main__":
    unittest.main()
