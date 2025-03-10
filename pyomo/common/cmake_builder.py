#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import errno
import os
import shutil
import stat
import sys
import tempfile

import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_executable


def handleReadonly(function, path, excinfo):
    excvalue = excinfo[1]
    if excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        function(path)
    else:
        raise


def build_cmake_project(
    targets, package_name=None, description=None, user_args=[], parallel=None
):
    from setuptools import Extension, Distribution
    from setuptools.command.build_ext import build_ext

    class _CMakeBuild(build_ext, object):
        def run(self):
            for cmake_ext in self.extensions:
                self._cmake_build_target(cmake_ext)

        def _cmake_build_target(self, cmake_ext):
            cmake_config = 'Debug' if self.debug else 'Release'
            cmake_args = [
                '-DCMAKE_INSTALL_PREFIX=' + envvar.PYOMO_CONFIG_DIR,
                #'-DCMAKE_BUILD_TYPE=' + cmake_config,
            ] + cmake_ext.user_args

            try:
                # Redirect all stderr to stdout (to prevent powershell
                # from inadvertently failing builds)
                sys.stderr.flush()
                sys.stdout.flush()
                old_stderr = os.dup(sys.stderr.fileno())
                os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
                old_environ = dict(os.environ)
                if cmake_ext.parallel:
                    # --parallel was only added in cmake 3.12.  Use an
                    # environment variable so that we don't have to bump
                    # the minimum cmake version.
                    os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = str(cmake_ext.parallel)

                cmake = find_executable('cmake')
                if cmake is None:
                    raise IOError("cmake not found in the system PATH")
                self.spawn([cmake, cmake_ext.target_dir] + cmake_args)
                if not self.dry_run:
                    # Skip build and go straight to install: the build
                    # harness should take care of dependencies and this
                    # will prevent repeated builds in MSVS
                    #
                    # self.spawn(['cmake', '--build', '.',
                    #            '--config', cmake_config])
                    self.spawn(
                        [
                            cmake,
                            '--build',
                            '.',
                            '--target',
                            'install',
                            '--config',
                            cmake_config,
                        ]
                    )
            finally:
                # Restore stderr
                sys.stderr.flush()
                sys.stdout.flush()
                os.dup2(old_stderr, sys.stderr.fileno())
                os.environ = old_environ

    class CMakeExtension(Extension, object):
        def __init__(self, target_dir, user_args, parallel):
            # don't invoke the original build_ext for this special extension
            super(CMakeExtension, self).__init__(
                self.__class__.__qualname__, sources=[]
            )
            self.target_dir = target_dir
            self.user_args = user_args
            self.parallel = parallel

    if package_name is None:
        package_name = 'build_cmake'
    if description is None:
        description = package_name
    # Get the source directory from the caller's frame for use in
    # resolving any relative target paths
    caller_dir = this_file_dir(2)
    ext_modules = [
        CMakeExtension(os.path.join(caller_dir, target), user_args, parallel)
        for target in targets
    ]

    sys.stdout.write(f"\n**** Building {description} ****\n")
    package_config = {
        'name': package_name,
        'packages': [],
        'ext_modules': ext_modules,
        'cmdclass': {'build_ext': _CMakeBuild},
    }
    dist = Distribution(package_config)
    basedir = os.path.abspath(os.path.curdir)
    try:
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(tmpdir)
        dist.run_command('build_ext')
        install_dir = envvar.PYOMO_CONFIG_DIR
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir, onerror=handleReadonly)
    sys.stdout.write(f"Installed {description} to {install_dir}\n")
