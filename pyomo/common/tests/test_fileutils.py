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

from pyomo.common.fileutils import thisFile, find_file, _system

_thisFile = thisFile()

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = None
        self.basedir = os.path.abspath(os.path.curdir)

    def tearDown(self):
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)
            os.chdir(self.basedir)

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

    def test_findfile(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())
        subdir = os.path.join(self.tmpdir, 'aaa')
        os.mkdir(subdir)
        os.chdir(self.tmpdir)

        fname = 'foo.py'
        self.assertEqual(
            None,
            find_file(fname)
        )

        open(os.path.join(self.tmpdir,fname),'w').close()
        open(os.path.join(subdir,fname),'w').close()
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
            find_file(fname, pathlist=['',os.pathsep.join(['',subdir])])
        )
        self.assertEqual(
            os.path.join(subdir,fname),
            find_file(fname, pathlist=['',os.pathsep.join(['',subdir])],
                      cwd=False)
        )
        # ...unless the CWD match fails the MODE check
        self.assertEqual(
            ( os.path.join(self.tmpdir,fname)
              if _system() in ('windiws','cygwin')
              else None ),
            find_file(fname, pathlist=['',os.pathsep.join(['',subdir])],
                      mode=os.X_OK)
        )
        mode = os.stat(os.path.join(subdir,fname)).st_mode
        os.chmod( os.path.join(subdir,fname),
                  mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH )
        self.assertEqual(
            os.path.join(subdir,fname),
            find_file(fname, pathlist=['',os.pathsep.join(['',subdir])],
                      mode=os.X_OK)
        )

