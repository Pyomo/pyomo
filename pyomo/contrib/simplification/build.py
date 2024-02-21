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

import glob
import os
import shutil
import sys
import tempfile
from distutils.dist import Distribution

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pyomo.common.envvar import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import find_library, this_file_dir


def build_ginac_interface(args=None):
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
            'could not find GiNaC library; please make sure it is in the LD_LIBRARY_PATH environment variable'
        )
    ginac_lib_dir = os.path.dirname(ginac_lib)
    ginac_build_dir = os.path.dirname(ginac_lib_dir)
    ginac_include_dir = os.path.join(ginac_build_dir, 'include')
    if not os.path.exists(os.path.join(ginac_include_dir, 'ginac', 'ginac.h')):
        raise RuntimeError('could not find GiNaC include directory')

    cln_lib = find_library('cln')
    if cln_lib is None:
        raise RuntimeError(
            'could not find CLN library; please make sure it is in the LD_LIBRARY_PATH environment variable'
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
            if self.inplace:
                tmpdir = this_file_dir()
            else:
                tmpdir = os.path.abspath(tempfile.mkdtemp())
            print("Building in '%s'" % tmpdir)
            os.chdir(tmpdir)
            try:
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
            finally:
                os.chdir(basedir)
                if not self.inplace:
                    shutil.rmtree(tmpdir, onerror=handleReadonly)

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


if __name__ == '__main__':
    build_ginac_interface(sys.argv[1:])
