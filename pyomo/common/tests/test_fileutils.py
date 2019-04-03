#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
import platform
import shutil
import stat
import sys
import tempfile

from six import StringIO

import pyutilib.th as unittest
from pyutilib.subprocess import run

import pyomo.common.config as config
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
    this_file, this_file_dir, find_file, find_library, find_executable, 
    ExecutableManager, _system, _path, _exeExt, _libExt,
)

_this_file = this_file()
_this_file_dir = this_file_dir()

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = None
        self.basedir = os.path.abspath(os.path.curdir)
        self.config = config.PYOMO_CONFIG_DIR
        self.ld_library_path = os.environ.get('LD_LIBRARY_PATH', None)
        self.path = os.environ.get('PATH', None)

    def tearDown(self):
        config.PYOMO_CONFIG_DIR = self.config
        os.chdir(self.basedir)
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)
        if self.ld_library_path is None:
            os.environ.pop('LD_LIBRARY_PATH', None)
        else:
            os.environ['LD_LIBRARY_PATH'] = self.ld_library_path
        if self.path is None:
            os.environ.pop('PATH', None)
        else:
            os.environ['PATH'] = self.path

    def _make_exec(self, fname):
        open(fname,'w').close()
        mode = os.stat(fname).st_mode
        os.chmod( fname, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH )

    def test_this_file(self):
        self.assertEquals(_this_file, __file__.replace('.pyc','.py'))
        # Note that in some versions of PyPy, this can return <module>
        # instead of the normal <string>
        self.assertIn(run([
            sys.executable,'-c',
            'from pyomo.common.fileutils import this_file;'
            'print(this_file())'
        ])[1].strip(), ['<string>','<module>'])
        self.assertEquals(run(
            [sys.executable],
            stdin='from pyomo.common.fileutils import this_file;'
            'print(this_file())'
        )[1].strip(), '<stdin>')

    def test_this_file_dir(self):
        expected_path = os.path.join('pyomo','common','tests')
        self.assertTrue(_this_file_dir.endswith(expected_path))

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
        # NOTE: PATH restored by tearDown()

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
        #  (except on Windows, where all files have X_OK)
        self.assertEqual(
            ( os.path.join(self.tmpdir,fname)
              if _system() in ('windows','cygwin')
              else None ),
            find_file(fname, pathlist=[subdir], mode=os.X_OK)
        )
        self._make_exec(os.path.join(subdir,fname))
        self.assertEqual(
            ( os.path.join(self.tmpdir,fname)
              if _system() in ('windows','cygwin')
              else os.path.join(subdir,fname) ),
            find_file(fname, pathlist=[subdir], mode=os.X_OK)
        )
        # pathlist may also be a string
        self.assertEqual(
            os.path.join(subdir,fname),
            find_file(fname, pathlist=os.pathsep+subdir+os.pathsep, cwd=False)
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
        config_libdir = os.path.join(self.tmpdir, 'lib')
        os.mkdir(config_libdir)
        config_bindir = os.path.join(self.tmpdir, 'bin')
        os.mkdir(config_bindir)

        ldlibdir_name = 'in_ld_lib'
        ldlibdir = os.path.join(self.tmpdir, ldlibdir_name)
        os.mkdir(ldlibdir)
        os.environ['LD_LIBRARY_PATH'] = os.pathsep + ldlibdir + os.pathsep

        pathdir_name = 'in_path'
        pathdir = os.path.join(self.tmpdir, pathdir_name)
        os.mkdir(pathdir)
        os.environ['PATH'] = os.pathsep + pathdir + os.pathsep

        libExt = _libExt[_system()][0]

        f_in_cwd_ldlib_path = 'f_in_cwd_ldlib_path'
        open(os.path.join(self.tmpdir,f_in_cwd_ldlib_path),'w').close()
        open(os.path.join(ldlibdir,f_in_cwd_ldlib_path),'w').close()
        open(os.path.join(pathdir,f_in_cwd_ldlib_path),'w').close()
        f_in_ldlib_extension = 'f_in_ldlib_extension'
        open(os.path.join(ldlibdir,f_in_ldlib_extension + libExt),'w').close()
        f_in_path = 'f_in_path'
        open(os.path.join(pathdir,f_in_path),'w').close()

        f_in_configlib = 'f_in_configlib'
        open(os.path.join(config_libdir, f_in_configlib),'w').close()
        f_in_configbin = 'f_in_configbin'
        open(os.path.join(config_bindir, f_in_ldlib_extension),'w').close()
        open(os.path.join(config_bindir, f_in_configbin),'w').close()


        self.assertEqual(
            os.path.join(self.tmpdir, f_in_cwd_ldlib_path),
            find_library(f_in_cwd_ldlib_path)
        )
        self.assertEqual(
            os.path.join(ldlibdir, f_in_cwd_ldlib_path),
            find_library(f_in_cwd_ldlib_path, cwd=False)
        )
        self.assertEqual(
            os.path.join(ldlibdir, f_in_ldlib_extension) + libExt,
            find_library(f_in_ldlib_extension)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_path),
            find_library(f_in_path)
        )
        self.assertEqual(
            None,
            find_library(f_in_path, include_PATH=False)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_path),
            find_library(f_in_path, pathlist=os.pathsep+pathdir+os.pathsep)
        )
        # test an explicit pathlist overrides LD_LIBRARY_PATH
        self.assertEqual(
            os.path.join(pathdir, f_in_cwd_ldlib_path),
            find_library(f_in_cwd_ldlib_path, cwd=False, pathlist=[pathdir])
        )
        # test that the PYOMO_CONFIG_DIR 'lib' dir is included
        self.assertEqual(
            os.path.join(config_libdir, f_in_configlib),
            find_library(f_in_configlib)
        )
        # and the Bin dir
        self.assertEqual(
            os.path.join(config_bindir, f_in_configbin),
            find_library(f_in_configbin)
        )
        # ... but only if include_PATH is true
        self.assertEqual(
            None,
            find_library(f_in_configbin, include_PATH=False)
        )
        # And none of them if the pathlist is specified
        self.assertEqual(
            None,
            find_library(f_in_configlib, pathlist=pathdir)
        )
        self.assertEqual(
            None,
            find_library(f_in_configbin, pathlist=pathdir)
        )


    def test_find_executable(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(self.tmpdir)

        config.PYOMO_CONFIG_DIR = self.tmpdir
        config_libdir = os.path.join(self.tmpdir, 'lib')
        os.mkdir(config_libdir)
        config_bindir = os.path.join(self.tmpdir, 'bin')
        os.mkdir(config_bindir)

        ldlibdir_name = 'in_ld_lib'
        ldlibdir = os.path.join(self.tmpdir, ldlibdir_name)
        os.mkdir(ldlibdir)
        os.environ['LD_LIBRARY_PATH'] = os.pathsep + ldlibdir + os.pathsep

        pathdir_name = 'in_path'
        pathdir = os.path.join(self.tmpdir, pathdir_name)
        os.mkdir(pathdir)
        os.environ['PATH'] = os.pathsep + pathdir + os.pathsep

        exeExt = _exeExt[_system()] or ''

        f_in_cwd_notexe = 'f_in_cwd_notexe'
        open(os.path.join(self.tmpdir,f_in_cwd_notexe), 'w').close()
        f_in_cwd_ldlib_path = 'f_in_cwd_ldlib_path'
        self._make_exec(os.path.join(self.tmpdir,f_in_cwd_ldlib_path))
        self._make_exec(os.path.join(ldlibdir,f_in_cwd_ldlib_path))
        self._make_exec(os.path.join(pathdir,f_in_cwd_ldlib_path))
        f_in_path_extension = 'f_in_path_extension'
        self._make_exec(os.path.join(pathdir,f_in_path_extension + exeExt))
        f_in_path = 'f_in_path'
        self._make_exec(os.path.join(pathdir,f_in_path))

        f_in_configlib = 'f_in_configlib'
        self._make_exec(os.path.join(config_libdir, f_in_configlib))
        f_in_configbin = 'f_in_configbin'
        self._make_exec(os.path.join(config_libdir, f_in_path_extension))
        self._make_exec(os.path.join(config_bindir, f_in_configbin))


        self.assertEqual(
            ( os.path.join(self.tmpdir,f_in_cwd_notexe)
              if _system() in ('windows','cygwin')
              else None ),
            find_executable(f_in_cwd_notexe)
        )
        self.assertEqual(
            os.path.join(self.tmpdir, f_in_cwd_ldlib_path),
            find_executable(f_in_cwd_ldlib_path)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_cwd_ldlib_path),
            find_executable(f_in_cwd_ldlib_path, cwd=False)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_path_extension) + exeExt,
            find_executable(f_in_path_extension)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_path),
            find_executable(f_in_path)
        )
        self.assertEqual(
            None,
            find_executable(f_in_path, include_PATH=False)
        )
        self.assertEqual(
            os.path.join(pathdir, f_in_path),
            find_executable(f_in_path, pathlist=os.pathsep+pathdir+os.pathsep)
        )
        # test an explicit pathlist overrides PATH
        self.assertEqual(
            os.path.join(ldlibdir, f_in_cwd_ldlib_path),
            find_executable(f_in_cwd_ldlib_path, cwd=False, pathlist=[ldlibdir])
        )
        # test that the PYOMO_CONFIG_DIR 'bin' dir is included
        self.assertEqual(
            os.path.join(config_bindir, f_in_configbin),
            find_executable(f_in_configbin)
        )
        # ... but only if the pathlist is not specified
        self.assertEqual(
            None,
            find_executable(f_in_configbin, pathlist=pathdir)
        )


    def test_ExecutableManager(self):
        Executable = ExecutableManager()
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())

        config.PYOMO_CONFIG_DIR = self.tmpdir
        config_bindir = os.path.join(self.tmpdir, 'bin')
        os.mkdir(config_bindir)

        pathdir_name = 'in_path'
        pathdir = os.path.join(self.tmpdir, pathdir_name)
        os.mkdir(pathdir)
        os.environ['PATH'] = os.pathsep + pathdir + os.pathsep

        f_in_tmp = 'f_in_tmp'
        self._make_exec(os.path.join(self.tmpdir,f_in_tmp))
        f_in_path = 'f_in_path'
        self._make_exec(os.path.join(pathdir,f_in_path))
        f_in_cfg = 'f_in_configbin'
        self._make_exec(os.path.join(config_bindir, f_in_cfg))

        # Test availability
        self.assertTrue( Executable(f_in_path).available() )
        if not Executable(f_in_path):
            self.fail("Expected casting Executable(f_in_path) to bool=True")

        # Test getting the path to the executable
        self.assertEqual( Executable(f_in_path).path(),
                          os.path.join(pathdir, f_in_path) )
        self.assertEqual( "%s" % Executable(f_in_path),
                          os.path.join(pathdir, f_in_path) )
        self.assertEqual( Executable(f_in_path).executable,
                          os.path.join(pathdir, f_in_path) )

        # Test the above for a nonexistant file
        self.assertFalse( Executable(f_in_tmp).available() )
        if Executable(f_in_tmp):
            self.fail("Expected casting Executable(f_in_tmp) to bool=False")
        self.assertIsNone( Executable(f_in_tmp).path() )
        self.assertEqual( "%s" % Executable(f_in_tmp), "" )
        self.assertIsNone( Executable(f_in_tmp).executable )

        # While the local CONFIG is set up with Pyomo, it will not be
        # reflected in the Executable pathlist, as that was set up when
        # Pyomo was first imported
        self.assertFalse( Executable(f_in_cfg).available() )
        Executable.pathlist.append(config_bindir)
        # and adding it won't change things (status is cached)
        self.assertFalse( Executable(f_in_cfg).available() )
        # until we tell the manager to rehash the executables
        Executable.rehash()
        self.assertTrue( Executable(f_in_cfg).available() )
        self.assertEqual( Executable(f_in_cfg).path(),
                          os.path.join(config_bindir, f_in_cfg) )

        # Another file that doesn't exist
        f_in_path2 = 'f_in_path2'
        f_loc = os.path.join(pathdir, f_in_path2)
        self.assertFalse( Executable(f_in_path2).available() )
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.common', logging.WARNING):
            Executable(f_in_path2).executable = f_loc
            self.assertIn(
                "explicitly setting the path for executable '%s' to a "
                "non-executable file or nonexistent location ('%s')"
                % (f_in_path2, f_loc), output.getvalue())
        self.assertFalse( Executable(f_in_path2).available() )
        self._make_exec(os.path.join(pathdir,f_in_path2))
        self.assertFalse( Executable(f_in_path2).available() )
        Executable(f_in_path2).rehash()
        self.assertTrue( Executable(f_in_path2).available() )

        # And disabling it will "remove" it
        Executable(f_in_path2).disable()
        self.assertFalse( Executable(f_in_path2).available() )
        self.assertIsNone( Executable(f_in_path2).path() )
        Executable(f_in_path2).rehash()
        self.assertTrue( Executable(f_in_path2).available() )
        self.assertEqual( Executable(f_in_path2).path(), f_loc )
