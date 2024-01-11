from pybind11.setup_helpers import Pybind11Extension, build_ext
from pyomo.common.fileutils import this_file_dir, find_library
import os
from distutils.dist import Distribution
import sys
import shutil
import glob
import tempfile
from pyomo.common.envvar import PYOMO_CONFIG_DIR


def build_ginac_interface(args=[]):
    dname = this_file_dir()
    _sources = ['ginac_interface.cpp']
    sources = list()
    for fname in _sources:
        sources.append(os.path.join(dname, fname))

    ginac_lib = find_library('ginac')
    print(ginac_lib)
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

    class ginac_build_ext(build_ext):
        def run(self):
            basedir = os.path.abspath(os.path.curdir)
            if self.inplace:
                tmpdir = this_file_dir()
            else:
                tmpdir = os.path.abspath(tempfile.mkdtemp())
            print("Building in '%s'" % tmpdir)
            os.chdir(tmpdir)
            try:
                super(ginac_build_ext, self).run()
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
        'cmdclass': {"build_ext": ginac_build_ext},
    }

    dist = Distribution(package_config)
    dist.script_args = ['build_ext'] + args
    dist.parse_command_line()
    dist.run_command('build_ext')


if __name__ == '__main__':
    build_ginac_interface(sys.argv[1:])
