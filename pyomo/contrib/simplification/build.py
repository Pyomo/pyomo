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

logger = logging.getLogger(__name__ if __name__ != '__main__' else 'pyomo')


def build_ginac_library(parallel=None, argv=None, env=None):
    sys.stdout.write("\n**** Building GiNaC library ****\n")

    configure_cmd = [
        os.path.join('.', 'configure'),
        '--prefix=' + PYOMO_CONFIG_DIR,
        '--disable-static',
    ]
    make_cmd = ['make']
    if parallel:
        make_cmd.append(f'-j{parallel}')
    install_cmd = ['make', 'install']

    env = dict(os.environ)
    pcdir = os.path.join(PYOMO_CONFIG_DIR, 'lib', 'pkgconfig')
    if 'PKG_CONFIG_PATH' in env:
        pcdir += os.pathsep + env['PKG_CONFIG_PATH']
    env['PKG_CONFIG_PATH'] = pcdir

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
        assert subprocess.run(configure_cmd, cwd=cln_dir, env=env).returncode == 0
        logger.info("\nBuilding CLN\n")
        assert subprocess.run(make_cmd, cwd=cln_dir, env=env).returncode == 0
        assert subprocess.run(install_cmd, cwd=cln_dir, env=env).returncode == 0

        url = 'https://www.ginac.de/ginac-1.8.9.tar.bz2'
        ginac_dir = os.path.join(tmpdir, 'ginac')
        downloader.set_destination_filename(ginac_dir)
        logger.info(
            "Fetching GiNaC from %s and installing it to %s"
            % (url, downloader.destination())
        )
        downloader.get_tar_archive(url, dirOffset=1)
        assert subprocess.run(configure_cmd, cwd=ginac_dir, env=env).returncode == 0
        logger.info("\nBuilding GiNaC\n")
        assert subprocess.run(make_cmd, cwd=ginac_dir, env=env).returncode == 0
        assert subprocess.run(install_cmd, cwd=ginac_dir, env=env).returncode == 0
        print("Installed GiNaC to %s" % (ginac_dir,))


def _find_include(libdir, incpaths):
    rel_path = ('include',) + incpaths
    while 1:
        basedir = os.path.dirname(libdir)
        if not basedir or basedir == libdir:
            return None
        if os.path.exists(os.path.join(basedir, *rel_path)):
            return os.path.join(basedir, *(rel_path[: -len(incpaths)]))
        libdir = basedir


def build_ginac_interface(parallel=None, args=None):
    from distutils.dist import Distribution
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pyomo.common.cmake_builder import handleReadonly

    sys.stdout.write("\n**** Building GiNaC interface ****\n")

    if args is None:
        args = []
    sources = [
        os.path.join(this_file_dir(), 'ginac', 'src', fname)
        for fname in ['ginac_interface.cpp']
    ]

    ginac_lib = find_library('ginac')
    if not ginac_lib:
        raise RuntimeError(
            'could not find the GiNaC library; please make sure either to install '
            'the library and development headers system-wide, or include the '
            'path to the library in the LD_LIBRARY_PATH environment variable'
        )
    ginac_lib_dir = os.path.dirname(ginac_lib)
    ginac_include_dir = _find_include(ginac_lib_dir, ('ginac', 'ginac.h'))
    if not ginac_include_dir:
        raise RuntimeError('could not find GiNaC include directory')

    cln_lib = find_library('cln')
    if not cln_lib:
        raise RuntimeError(
            'could not find the CLN library; please make sure either to install '
            'the library and development headers system-wide, or include the '
            'path to the library in the LD_LIBRARY_PATH environment variable'
        )
    cln_lib_dir = os.path.dirname(cln_lib)
    cln_include_dir = _find_include(cln_lib_dir, ('cln', 'cln.h'))
    if not cln_include_dir:
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
                    tmpdir = os.path.join(this_file_dir(), 'ginac')
                else:
                    tmpdir = os.path.abspath(tempfile.mkdtemp())
                sys.stdout.write("Building in '%s'\n" % tmpdir)
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
                    sys.stdout.write(f"Installing {library} in {target}\n")
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        dest='parallel',
        type=int,
        default=None,
        help="Enable parallel build with PARALLEL cores",
    )
    parser.add_argument(
        "--build-deps",
        dest='build_deps',
        action='store_true',
        default=False,
        help="Download and build the CLN/GiNaC libraries",
    )
    options, argv = parser.parse_known_args(sys.argv)
    logging.getLogger('pyomo').setLevel(logging.INFO)
    if options.build_deps:
        build_ginac_library(options.parallel, [])
    build_ginac_interface(options.parallel, argv[1:])
