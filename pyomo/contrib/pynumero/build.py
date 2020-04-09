#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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

from pyomo.common import config
from pyomo.common.fileutils import this_file_dir

def handleReadonly(function, path, excinfo):
    excvalue = excinfo[1]
    if excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        function(path)
    else:
        raise

def build_pynumero(user_args=[]):
    import distutils.core
    from setuptools import Extension
    from distutils.command.build_ext import build_ext

    class _CMakeBuild(build_ext, object):
        def run(self):
            project_dir = self.extensions[0].project_dir

            cmake_config = 'Debug' if self.debug else 'Release'
            cmake_args = [
                '-DCMAKE_INSTALL_PREFIX=' + config.PYOMO_CONFIG_DIR,
                '-DBUILD_AMPLMP_IF_NEEDED=ON',
                #'-DCMAKE_BUILD_TYPE=' + cmake_config,
            ] + user_args

            build_args = [
                '--config', cmake_config,
            ]

            try:
                # Redirect all stderr to stdout (to prevent powershell
                # from inadvertently failing builds)
                oldstderr = os.dup(sys.stderr.fileno())
                os.dup2(sys.stdout.fileno(), sys.stderr.fileno())

                self.spawn(['cmake', project_dir] + cmake_args)
                if not self.dry_run:
                    self.spawn(['cmake', '--build', '.'] + build_args)
                    self.spawn(['cmake', '--build', '.',
                                '--target', 'install'] + build_args)
            finally:
                # Restore stderr
                os.dup2(oldstderr, sys.stderr.fileno())

    class CMakeExtension(Extension, object):
        def __init__(self, name):
            # don't invoke the original build_ext for this special extension
            super(CMakeExtension, self).__init__(name, sources=[])
            self.project_dir = os.path.join(this_file_dir(), name)

    sys.stdout.write("\n**** Building PyNumero libraries ****\n")
    package_config = {
        'name': 'pynumero_libraries',
        'packages': [],
        'ext_modules': [CMakeExtension("src")],
        'cmdclass': {'build_ext': _CMakeBuild},
    }
    dist = distutils.core.Distribution(package_config)
    # install_dir = os.path.join(config.PYOMO_CONFIG_DIR, 'lib')
    # dist.get_command_obj('install_lib').install_dir = install_dir
    try:
        basedir = os.path.abspath(os.path.curdir)
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(tmpdir)
        dist.run_command('build_ext')
        install_dir = os.path.join(config.PYOMO_CONFIG_DIR, 'lib')
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir, onerror=handleReadonly)
      sys.stdout.write("Installed PyNumero libraries to %s\n" % ( install_dir, ))

if __name__ == "__main__":
    build_pynumero(sys.argv[1:])

