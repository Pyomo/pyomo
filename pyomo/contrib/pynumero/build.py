#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import sys
import os
import shutil
import tempfile

from pyomo.common import config
from pyomo.common.fileutils import this_file_dir

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
                '-DCMAKE_BUILD_TYPE=' + cmake_config,
                '-DBUILD_AMPLMP_IF_NEEDED=ON',
            ] + user_args

            build_args = [
                '--config', cmake_config,
            ]

            self.spawn(['cmake', project_dir] + cmake_args)
            if not self.dry_run:
                self.spawn(['cmake', '--build', '.'] + build_args)
                self.spawn(['cmake', '--build', '.',
                            '--target', 'install'] + build_args)

    class CMakeExtension(Extension, object):
        def __init__(self, name):
            # don't invoke the original build_ext for this special extension
            super(CMakeExtension, self).__init__(name, sources=[])
            self.project_dir = os.path.join(this_file_dir(), name)

    package_config = {
        'name': 'pynumero_libraries',
        'packages': [],
        'ext_modules': [CMakeExtension("src")],
        'cmdclass': {'build_ext': _CMakeBuild},
    }
    dist = distutils.core.Distribution(package_config)
    # install_dir = os.path.join(config.PYOMO_CONFIG_DIR, 'lib')
    # dist.get_command_obj('install_lib').install_dir = install_dir
    print("\n**** Building PyNumero libraries ****")
    try:
        basedir = os.path.abspath(os.path.curdir)
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(tmpdir)
        dist.run_command('build_ext')
        install_dir = os.path.join(config.PYOMO_CONFIG_DIR, 'lib')
        print("Installed PyNumero libraries to %s" % ( install_dir, ))
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    build_pynumero(sys.argv[1:])

