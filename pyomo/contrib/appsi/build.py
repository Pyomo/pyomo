#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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
import warnings
from pyomo.common.envvar import PYOMO_CONFIG_DIR
from pyomo.common.fileutils import this_file_dir, find_library


def handleReadonly(function, path, excinfo):
    excvalue = excinfo[1]
    if excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
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
                #'interval.cpp',
                'expression.cpp',
                'common.cpp',
                'nl_writer.cpp',
                'lp_writer.cpp',
                #'model.cpp',
                'cmodel_bindings.cpp',
        )
    ]

    if in_setup:
        package_name = 'pyomo.contrib.appsi.cmodel.appsi_cmodel'
    else:
        package_name = 'appsi_cmodel'
    return Pybind11Extension(package_name, sources, extra_compile_args=['-std=c++11'])

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
                    target = os.path.join(
                        PYOMO_CONFIG_DIR, 'lib',
                        'python%s.%s' % sys.version_info[:2],
                        'site-packages', '.')
                    if not os.path.exists(target):
                        os.makedirs(target)
                    for ext in self.extensions:
                        library = glob.glob(f'build/*/{ext.name}.*')[0]
                        shutil.copy(library, target)
            finally:
                os.chdir(basedir)
                if not self.inplace:
                    shutil.rmtree(tmpdir, onerror=handleReadonly)

    try:
        original_pybind11_setup_helpers_macos = pybind11.setup_helpers.MACOS
        pybind11.setup_helpers.MACOS = False

        extensions = list()
        extensions.append(Pybind11Extension("appsi_cmodel", sources))

        # search for Highs; if found, add an extension
        highs_lib = find_library('highs', include_PATH=False)
        if highs_lib is not None:
            print(f'found Highs library: {highs_lib}')
            highs_lib_dir = os.path.dirname(highs_lib)
            highs_build_dir = os.path.dirname(highs_lib_dir)
            highs_include_dir = os.path.join(highs_build_dir, 'include')
            if os.path.exists(os.path.join(highs_include_dir, 'Highs.h')):
                print('found Highs include dir')
                highs_ext = Pybind11Extension('appsi_highs',
                                              sources=[os.path.join(appsi_root, 'highs_bindings', 'highs_bindings.cpp')],
                                              language='c++',
                                              include_dirs=[highs_include_dir],
                                              library_dirs=[highs_lib_dir],
                                              libraries=['highs'])
                extensions.append(highs_ext)
            else:
                print(f'Found HiGHS library, but could not find HiGHS include directory ({highs_include_dir}). Skipping HiGHS Python bindings.')
        else:
            print('Could not find HiGHS library; Skipping HiGHS Python bindings')

        package_config = {
            'name': 'appsi_cmodel',
            'packages': [],
            'ext_modules': extensions,
            'cmdclass': {
                "build_ext": appsi_build_ext,
            },
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
