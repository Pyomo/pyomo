#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import ctypes
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
    PathManager, _system, _path, _exeExt, _libExt, _ExecutableData,
)
from pyomo.common.download import FileDownloader

try:
    samefile = os.path.samefile
except AttributeError:
    # os.path.samefile is not available in Python 2.7 under Windows.
    # Mock up a dummy function for that platform.
    def samefile(a,b):
        return True

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

    def _check_file(self, found, ref):
        #
        # On OSX, the identified path is (sometimes) prepended with
        # '/private/'. These two paths are in fact equivalent on OSX,
        # but we will explicitly verify that fact.  We check the string
        # first so that we can generate a more informative error in the
        # case of "gross" failure.
        #
        self.assertTrue(
            found.endswith(ref), "%s does not end with %s" % (found, ref))
        self.assertTrue(samefile(ref, found))

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
        self._check_file(find_file(fname),os.path.join(self.tmpdir,fname))

        # unless we don't look in the cwd
        self.assertIsNone(find_file(fname, cwd=False))

        # cwd overrides pathlist
        self._check_file(find_file(fname, pathlist=[subdir]),
                         os.path.join(self.tmpdir,fname))
        
        self._check_file(find_file(fname, pathlist=[subdir], cwd=False),
                         os.path.join(subdir,fname))

        # ...unless the CWD match fails the MODE check
        #  (except on Windows, where all files have X_OK)
        found = find_file(fname, pathlist=[subdir], mode=os.X_OK)
        if _system() in ('windows','cygwin'):
            self._check_file(found, os.path.join(self.tmpdir,fname))
        else:
            self.assertIsNone(found)

        self._make_exec(os.path.join(subdir,fname))
        found = find_file(fname, pathlist=[subdir], mode=os.X_OK)
        if _system() in ('windows','cygwin'):
            ref = os.path.join(self.tmpdir,fname)
        else:
            ref = os.path.join(subdir,fname)
        self._check_file(found, ref)

        # pathlist may also be a string
        self._check_file(
            find_file(fname, pathlist=os.pathsep+subdir+os.pathsep, cwd=False),
            os.path.join(subdir,fname)
        )

        # implicit extensions work (even if they are not necessary)
        self._check_file(find_file(fname, ext='.py'),
                         os.path.join(self.tmpdir,fname))
        self._check_file(find_file(fname, ext=['.py']),
                         os.path.join(self.tmpdir,fname))

        # implicit extensions work (when they are necessary)
        self._check_file(find_file(fname[:-3], ext='.py'),
                         os.path.join(self.tmpdir,fname))

        self._check_file(find_file(fname[:-3], ext=['.py']),
                         os.path.join(self.tmpdir,fname))

        # only files are found
        self._check_file(find_file( subdir_name,
                                    pathlist=[self.tmpdir, subdir], cwd=False ),
                         os.path.join(subdir,subdir_name))

        # empty dirs are skipped
        self._check_file(
            find_file( subdir_name,
                       pathlist=['', self.tmpdir, subdir], cwd=False ),
            os.path.join(subdir,subdir_name)
        )

    def test_find_library(self):
        self.tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(self.tmpdir)

        # Find a system library (before we muck with the PATH)
        _args = {'cwd':False, 'include_PATH':False, 'pathlist':[]}
        if FileDownloader.get_sysinfo()[0] == 'windows':
            a = find_library('ntdll', **_args)
            b = find_library('ntdll.dll', **_args)
            c = find_library('foo\\bar\\ntdll.dll', **_args)
        else:
            a = find_library('c', **_args)
            b = find_library('libc.so', **_args)
            c = find_library('foo/bar/libc.so', **_args)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNotNone(c)
        self.assertEqual(a,b)
        self.assertEqual(a,c)
        # Verify that the library is loadable (they are all the same
        # file, so only check one)
        _lib = ctypes.cdll.LoadLibrary(a)
        self.assertIsNotNone(_lib)

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


        self._check_file(
            find_library(f_in_cwd_ldlib_path),
            os.path.join(self.tmpdir, f_in_cwd_ldlib_path)
        )
        self._check_file(
            os.path.join(ldlibdir, f_in_cwd_ldlib_path),
            find_library(f_in_cwd_ldlib_path, cwd=False)
        )
        self._check_file(
            os.path.join(ldlibdir, f_in_ldlib_extension) + libExt,
            find_library(f_in_ldlib_extension)
        )
        self._check_file(
            os.path.join(pathdir, f_in_path),
            find_library(f_in_path)
        )
        if _system() == 'windows':
            self._check_file(
                os.path.join(pathdir, f_in_path),
                find_library(f_in_path, include_PATH=False)
            )
        else:
            # Note that on Windows, ctypes.util.find_library *always*
            # searches the PATH
            self.assertIsNone(
                find_library(f_in_path, include_PATH=False)
            )
        self._check_file(
            os.path.join(pathdir, f_in_path),
            find_library(f_in_path, pathlist=os.pathsep+pathdir+os.pathsep)
        )
        # test an explicit pathlist overrides LD_LIBRARY_PATH
        self._check_file(
            os.path.join(pathdir, f_in_cwd_ldlib_path),
            find_library(f_in_cwd_ldlib_path, cwd=False, pathlist=[pathdir])
        )
        # test that the PYOMO_CONFIG_DIR 'lib' dir is included
        self._check_file(
            os.path.join(config_libdir, f_in_configlib),
            find_library(f_in_configlib)
        )
        # and the Bin dir
        self._check_file(
            os.path.join(config_bindir, f_in_configbin),
            find_library(f_in_configbin)
        )
        # ... but only if include_PATH is true
        self.assertIsNone(
            find_library(f_in_configbin, include_PATH=False)
        )
        # And none of them if the pathlist is specified
        self.assertIsNone(
            find_library(f_in_configlib, pathlist=pathdir)
        )
        self.assertIsNone(
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


        found = find_executable(f_in_cwd_notexe)
        if _system() in ('windows','cygwin'):
            self._check_file(found, os.path.join(self.tmpdir,f_in_cwd_notexe))
        else:
            self.assertIsNone(found)

        self._check_file(
            find_executable(f_in_cwd_ldlib_path),
            os.path.join(self.tmpdir, f_in_cwd_ldlib_path)
        )
        self._check_file(
            os.path.join(pathdir, f_in_cwd_ldlib_path),
            find_executable(f_in_cwd_ldlib_path, cwd=False)
        )
        self._check_file(
            find_executable(f_in_path_extension),
            os.path.join(pathdir, f_in_path_extension) + exeExt
        )
        self._check_file(
            find_executable(f_in_path),
            os.path.join(pathdir, f_in_path)
        )
        self.assertIsNone(
            find_executable(f_in_path, include_PATH=False)
        )
        self._check_file(
            find_executable(f_in_path, pathlist=os.pathsep+pathdir+os.pathsep),
            os.path.join(pathdir, f_in_path)
        )
        
        # test an explicit pathlist overrides PATH
        self._check_file(
            os.path.join(ldlibdir, f_in_cwd_ldlib_path),
            find_executable(f_in_cwd_ldlib_path, cwd=False, pathlist=[ldlibdir])
        )
        # test that the PYOMO_CONFIG_DIR 'bin' dir is included
        self._check_file(
            os.path.join(config_bindir, f_in_configbin),
            find_executable(f_in_configbin)
        )
        # ... but only if the pathlist is not specified
        self.assertIsNone(
            find_executable(f_in_configbin, pathlist=pathdir)
        )


    def test_PathManager(self):
        Executable = PathManager(find_executable, _ExecutableData)
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
        self._check_file( Executable(f_in_path).path(),
                          os.path.join(pathdir, f_in_path) )
        self._check_file( "%s" % Executable(f_in_path),
                          os.path.join(pathdir, f_in_path) )
        self._check_file( Executable(f_in_path).executable,
                          os.path.join(pathdir, f_in_path) )

        # Test the above for a nonexistant file
        self.assertFalse( Executable(f_in_tmp).available() )
        if Executable(f_in_tmp):
            self.fail("Expected casting Executable(f_in_tmp) to bool=False")
        self.assertIsNone( Executable(f_in_tmp).path() )
        self.assertEqual( "%s" % Executable(f_in_tmp), "" )
        self.assertIsNone( Executable(f_in_tmp).executable )

        # If we override the pathlist, then we will not find the CONFIGDIR
        Executable.pathlist = []
        self.assertFalse( Executable(f_in_cfg).available() )
        Executable.pathlist.append(config_bindir)
        # and adding it won't change things (status is cached)
        self.assertFalse( Executable(f_in_cfg).available() )
        # until we tell the manager to rehash the executables
        Executable.rehash()
        self.assertTrue( Executable(f_in_cfg).available() )
        self.assertEqual( Executable(f_in_cfg).path(),
                          os.path.join(config_bindir, f_in_cfg) )
        # Note that if we clear the pathlist, then the current value of
        # CONFIGDIR will be honored
        Executable.pathlist = None
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
                "explicitly setting the path for '%s' to an "
                "invalid object or nonexistent location ('%s')"
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
