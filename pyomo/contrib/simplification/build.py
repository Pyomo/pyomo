#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import glob
import logging
import os
import shutil
import sys
import subprocess

from pyomo.common.download import FileDownloader
from pyomo.common.envvar import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import find_library, this_file_dir
from pyomo.common.tempfiles import TempfileManager


logger = logging.getLogger(__name__)


def build_ginac_library(parallel=None, argv=None):
    print("\n**** Building GiNaC library ****")

    configure_cmd = ['configure', '--prefix=' + PYOMO_CONFIG_DIR, '--disable-static']
    make_cmd = ['make']
    if parallel:
        make_cmd.append(f'-j{parallel}')
    install_cmd = ['make', 'install']

    with TempfileManager.new_context() as tempfile:
        tmpdir = tempfile.mkdtemp()

        downloader = FileDownloader()
        if argv:
            downloader.parse_args(argv)

        url = 'https://www.ginac.de/CLN/cln-1.3.7.tar.bz2'
        cln_dir = os.path.join(tmpdir, 'cln')
        downloader.set_destination_filename(cln_dir)
        logger.info(
            "Fetching CLN from %s and installing it to %s"
            % (url, downloader.destination())
        )
        downloader.get_tar_archive(url, dirOffset=1)
        assert subprocess.run(configure_cmd, cwd=cln_dir).returncode == 0
        logger.info("\nBuilding CLN\n")
        assert subprocess.run(make_cmd, cwd=cln_dir).returncode == 0
        assert subprocess.run(install_cmd, cwd=cln_dir).returncode == 0

        url = 'https://www.ginac.de/ginac-1.8.7.tar.bz2'
        ginac_dir = os.path.join(tmpdir, 'ginac')
        downloader.set_destination_filename(ginac_dir)
        logger.info(
            "Fetching GiNaC from %s and installing it to %s"
            % (url, downloader.destination())
        )
        downloader.get_tar_archive(url, dirOffset=1)
        assert subprocess.run(configure_cmd, cwd=ginac_dir).returncode == 0
        logger.info("\nBuilding GiNaC\n")
        assert subprocess.run(make_cmd, cwd=ginac_dir).returncode == 0
        assert subprocess.run(install_cmd, cwd=ginac_dir).returncode == 0


def build_ginac_interface(parallel=None, args=None):
    from distutils.dist import Distribution
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pyomo.common.cmake_builder import handleReadonly

    print("\n**** Building GiNaC interface ****")

    if args is None:
        args = list()
    dname = this_file_dir()
    _sources = ['ginac_interface.cpp']
    sources = list()
    for fname in _sources:
        sources.append(os.path.join(dname, fname))

    ginac_lib = find_library('ginac')
    if ginac_lib is None:
        raise RuntimeError(
            'could not find the GiNaC library; please make sure either to install '
            'the library and development headers system-wide, or include the '
            'path tt the library in the LD_LIBRARY_PATH environment variable'
        )
    ginac_lib_dir = os.path.dirname(ginac_lib)
    ginac_build_dir = os.path.dirname(ginac_lib_dir)
    ginac_include_dir = os.path.join(ginac_build_dir, 'include')
    if not os.path.exists(os.path.join(ginac_include_dir, 'ginac', 'ginac.h')):
        raise RuntimeError('could not find GiNaC include directory')

    cln_lib = find_library('cln')
    if cln_lib is None:
        raise RuntimeError(
            'could not find the CLN library; please make sure either to install '
            'the library and development headers system-wide, or include the '
            'path tt the library in the LD_LIBRARY_PATH environment variable'
        )
    cln_lib_dir = os.path.dirname(cln_lib)
    cln_build_dir = os.path.dirname(cln_lib_dir)
    cln_include_dir = os.path.join(cln_build_dir, 'include')
    if not os.path.exists(os.path.join(cln_include_dir, 'cln', 'cln.h')):
        raise RuntimeError('could not find CLN include directory')

    extra_args = ['-std=c++11']
    ext = Pybind11Extension(
        'ginac_interface',
        sources=sources,
        language='c++',
        include_dirs=[cln_include_dir, ginac_include_dir],
        library_dirs=[cln_lib_dir, ginac_lib_dir],
        libraries=['cln', 'ginac'],
        extra_compile_args=extra_args,
    )

    class ginacBuildExt(build_ext):
        def run(self):
            basedir = os.path.abspath(os.path.curdir)
            with TempfileManager.new_context() as tempfile:
                if self.inplace:
                    tmpdir = this_file_dir()
                else:
                    tmpdir = os.path.abspath(tempfile.mkdtemp())
                print("Building in '%s'" % tmpdir)
                os.chdir(tmpdir)
                super(ginacBuildExt, self).run()
                if not self.inplace:
                    library = glob.glob("build/*/ginac_interface.*")[0]
                    target = os.path.join(
                        PYOMO_CONFIG_DIR,
                        'lib',
                        'python%s.%s' % sys.version_info[:2],
                        'site-packages',
                        '.',
                    )
                    if not os.path.exists(target):
                        os.makedirs(target)
                    shutil.copy(library, target)

    package_config = {
        'name': 'ginac_interface',
        'packages': [],
        'ext_modules': [ext],
        'cmdclass': {"build_ext": ginacBuildExt},
    }

    dist = Distribution(package_config)
    dist.script_args = ['build_ext'] + args
    dist.parse_command_line()
    dist.run_command('build_ext')


class GiNaCInterfaceBuilder(object):
    def __call__(self, parallel):
        return build_ginac_interface(parallel)

    def skip(self):
        return not find_library('ginac')


if __name__ == '__main__':
    logging.getLogger('pyomo').setLevel(logging.DEBUG)
    parallel = None
    for i, arg in enumerate(sys.argv):
        if arg == '-j':
            parallel = int(sys.argv.pop(i + 1))
            sys.argv.pop(i)
            break
        if arg.startswith('-j'):
            if '=' in arg:
                parallel = int(arg.split('=')[1])
            else:
                parallel = int(arg[2:])
            sys.argv.pop(i)
            break
    if '--build-deps' in sys.argv:
        sys.argv.remove('--build-deps')
        build_ginac_library(parallel, [])
    build_ginac_interface(parallel, sys.argv[1:])
