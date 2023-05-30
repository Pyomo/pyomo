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

import shutil
import glob
import os
import sys
import tempfile


def handleReadonly(function, path, excinfo):
    excvalue = excinfo[1]
    if excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        function(path)
    else:
        raise


def get_appsi_extension(in_setup=False, appsi_root=None):
    from pybind11.setup_helpers import Pybind11Extension

    if appsi_root is None:
        from pyomo.common.fileutils import this_file_dir

        appsi_root = this_file_dir()

    sources = [
        os.path.join(appsi_root, 'cmodel', 'src', file_)
        for file_ in (
            'interval.cpp',
            'expression.cpp',
            'common.cpp',
            'nl_writer.cpp',
            'lp_writer.cpp',
            'model_base.cpp',
            'fbbt_model.cpp',
            'cmodel_bindings.cpp',
        )
    ]

    if in_setup:
        package_name = 'pyomo.contrib.appsi.cmodel.appsi_cmodel'
    else:
        package_name = 'appsi_cmodel'
    if sys.platform.startswith('win'):
        # Assume that builds on Windows will use MSVC
        # MSVC doesn't have a flag for c++11, use c++14
        extra_args = ['/std:c++14']
    else:
        # Assume all other platforms are GCC-like
        extra_args = ['-std=c++11']
    return Pybind11Extension(package_name, sources, extra_compile_args=extra_args)


def build_appsi(args=[]):
    print('\n\n**** Building APPSI ****')
    import setuptools
    from distutils.dist import Distribution
    from pybind11.setup_helpers import build_ext
    import pybind11.setup_helpers
    from pyomo.common.envvar import PYOMO_CONFIG_DIR
    from pyomo.common.fileutils import this_file_dir

    class appsi_build_ext(build_ext):
        def run(self):
            basedir = os.path.abspath(os.path.curdir)
            if self.inplace:
                tmpdir = os.path.join(this_file_dir(), 'cmodel')
            else:
                tmpdir = os.path.abspath(tempfile.mkdtemp())
            print("Building in '%s'" % tmpdir)
            os.chdir(tmpdir)
            try:
                super(appsi_build_ext, self).run()
                if not self.inplace:
                    library = glob.glob("build/*/appsi_cmodel.*")[0]
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

    try:
        original_pybind11_setup_helpers_macos = pybind11.setup_helpers.MACOS
        pybind11.setup_helpers.MACOS = False

        package_config = {
            'name': 'appsi_cmodel',
            'packages': [],
            'ext_modules': [get_appsi_extension(False)],
            'cmdclass': {"build_ext": appsi_build_ext},
        }

        dist = Distribution(package_config)
        dist.script_args = ['build_ext'] + args
        dist.parse_command_line()
        dist.run_command('build_ext')
    finally:
        pybind11.setup_helpers.MACOS = original_pybind11_setup_helpers_macos


class AppsiBuilder(object):
    def __call__(self, parallel):
        return build_appsi()


if __name__ == '__main__':
    # Note: this recognizes the "--inplace" command line argument: build
    # directory will be put in the source tree (and preserved), and the
    # SO will be left in appsi/cmodel.
    build_appsi(sys.argv[1:])
