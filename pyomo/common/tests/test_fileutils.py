#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import platform
import shutil
import stat
import sys
import tempfile

import pyutilib.th as unittest
from pyutilib.subprocess import run

import pyomo.common.config as config
from pyomo.common.fileutils import (
    thisFile, find_file, find_library, find_executable,
    _system, _path, _libExt,
)

_thisFile = thisFile()

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = None
        self.basedir = os.path.abspath(os.path.curdir)
        self.config = config.PYOMO_CONFIG_DIR
        self.ld_library_path = os.environ.get('LD_LIBRARY_PATH', None)
        self.path = os.environ.get('PATH', None)

    def tearDown(self):
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)
            os.chdir(self.basedir)
        if self.ld_library_path is None:
            os.environ.pop('LD_LIBRARY_PATH', None)
        else:
            os.environ['LD_LIBRARY_PATH'] = self.ld_library_path
        if self.path is None:
            os.environ.pop('PATH', None)
        else:
            os.environ['PATH'] = self.path
        config.PYOMO_CONFIG_DIR = self.config

    def test_thisFile(self):
        self.assertEquals(_thisFile, __file__.replace('.pyc','.py'))
        self.assertEquals(run([
            sys.executable,'-c',
            'from pyomo.common.fileutils import thisFile;print(thisFile())'
        ])[1].strip(), '<string>')
        self.assertEquals(run(
            [sys.executable],
            stdin='from pyomo.common.fileutils import thisFile;'
            'print(thisFile())'
        )[1].strip(), '<stdin>')

    def test_system(self):
        self.assertTrue(platform.system().lower().startswith(_system()))
        self.assertNotIn('.', _system())
        self.assertNotIn('-', _system())
        self.assertNotIn('_', _system())

    def test_path(self):
        orig_path = os.environ.get('PATH', None)
        if orig_path:
            self.assertEqual(os.pathsep.join(_path()), os.environ['PATH'])
        os.environ.pop('PATH', None)
        self.assertEqual(os.pathsep.join(_path()), os.defpath)
        # PATH restored by teadDown()

    def test_findfile(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())
        subdir_name = 'aaa'
        subdir = os.path.join(self.tmpdir, subdir_name)
        os.mkdir(subdir)
        os.chdir(self.tmpdir)

        fname = 'foo.py'
        self.assertEqual(
            None,
            find_file(fname)
        )

        open(os.path.join(self.tmpdir,fname),'w').close()
        open(os.path.join(subdir,fname),'w').close()
        open(os.path.join(subdir,'aaa'),'w').close()
        # we can find files in the CWD
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname)
        )
        # unless we don't look in the cwd
        self.assertEqual(
            None,
            find_file(fname, cwd=False)
        )
        # cwd overrides pathlist
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname, pathlist=[subdir])
        )
        self.assertEqual(
            os.path.join(subdir,fname),
            find_file(fname, pathlist=[subdir], cwd=False)
        )
        # ...unless the CWD match fails the MODE check
        self.assertEqual(
            ( os.path.join(self.tmpdir,fname)
              if _system() in ('windiws','cygwin')
              else None ),
            find_file(fname, pathlist=[subdir], mode=os.X_OK)
        )
        mode = os.stat(os.path.join(subdir,fname)).st_mode
        os.chmod( os.path.join(subdir,fname),
                  mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH )
        self.assertEqual(
            os.path.join(subdir,fname),
            find_file(fname, pathlist=[subdir], mode=os.X_OK)
        )

        # implicit extensions work (even if they are not necessary)
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname, ext='.py')
        )
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname, ext=['.py'])
        )

        # implicit extensions work (and when they are not necessary)
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname[:-3], ext='.py')
        )
        self.assertEqual(
            os.path.join(self.tmpdir,fname),
            find_file(fname[:-3], ext=['.py'])
        )

        # only files are found
        self.assertEqual(
            os.path.join(subdir,subdir_name),
            find_file( subdir_name,
                       pathlist=[self.tmpdir, subdir], cwd=False )
        )

        # empty dirs are skipped
        self.assertEqual(
            os.path.join(subdir,subdir_name),
            find_file( subdir_name,
                       pathlist=['', self.tmpdir, subdir], cwd=False )
        )

    def test_find_library(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(self.tmpdir)

        config.PYOMO_CONFIG_DIR = self.tmpdir
        os.mkdir(os.path.join(self.tmpdir, 'lib'))
        os.mkdir(os.path.join(self.tmpdir, 'bin'))

        subdir_name = 'a_lib'
        subdir = os.path.join(self.tmpdir, subdir_name)
        os.mkdir(subdir)
        bindir_name = 'a_bin'
        bindir = os.path.join(self.tmpdir, bindir_name)
        os.mkdir(bindir)

        libExt = _libExt[_system()][0]

        fname1 = 'foo'
        open(os.path.join(self.tmpdir,fname1),'w').close()
        open(os.path.join(subdir,fname1),'w').close()
        open(os.path.join(bindir,fname1),'w').close()
        fname2 = 'bar'
        open(os.path.join(subdir,fname2 + libExt),'w').close()
        fname3 = 'baz'
        open(os.path.join(bindir,fname3),'w').close()

        fname4 = 'in_lib'
        open(os.path.join(self.tmpdir, 'lib', fname4),'w').close()
        fname5 = 'in_bin'
        open(os.path.join(self.tmpdir, 'bin', fname2),'w').close()
        open(os.path.join(self.tmpdir, 'bin', fname5),'w').close()

        os.environ['LD_LIBRARY_PATH'] = os.pathsep + subdir + os.pathsep
        os.environ['PATH'] = os.pathsep + bindir + os.pathsep

        self.assertEqual(
            os.path.join(self.tmpdir, fname1),
            find_library(fname1)
        )
        self.assertEqual(
            os.path.join(subdir, fname1),
            find_library(fname1, cwd=False)
        )
        self.assertEqual(
            os.path.join(subdir, fname2) + libExt,
            find_library(fname2)
        )
        self.assertEqual(
            os.path.join(bindir, fname3),
            find_library(fname3)
        )
        self.assertEqual(
            None,
            find_library(fname3, include_PATH=False)
        )
        self.assertEqual(
            os.path.join(bindir, fname3),
            find_library(fname3, pathlist=os.pathsep+bindir+os.pathsep)
        )
        # test an explicit pathlist overrides LD_LIBRARY_PATH
        self.assertEqual(
            os.path.join(bindir, fname1),
            find_library(fname1, cwd=False, pathlist=[bindir])
        )
        # test that the PYOMO_CONFIG_DIR is included
        self.assertEqual(
            os.path.join(self.tmpdir, 'lib', fname4),
            find_library(fname4)
        )
        # and the Bin dir
        self.assertEqual(
            os.path.join(self.tmpdir, 'bin', fname5),
            find_library(fname5)
        )
        # ... but only if include_PATH is true
        self.assertEqual(
            None,
            find_library(fname5, include_PATH=False)
        )
        # And none of them if the pathlist is specified
        self.assertEqual(
            None,
            find_library(fname4, pathlist=bindir)
        )
        self.assertEqual(
            None,
            find_library(fname5, pathlist=bindir)
        )
