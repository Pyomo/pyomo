from pybind11.setup_helpers import Pybind11Extension, build_ext
from pyomo.common.fileutils import this_file_dir
import os
from distutils.dist import Distribution
import sys


def build_ginac_interface(args=[]):
    dname = this_file_dir()
    _sources = [
        'ginac_interface.cpp',
    ]
    sources = list()
    for fname in _sources:
        sources.append(os.path.join(dname, fname))
    extra_args = ['-std=c++11']
    ext = Pybind11Extension('ginac_interface', sources, extra_compile_args=extra_args)

    package_config = {
        'name': 'ginac_interface',
        'packages': [],
        'ext_modules': [ext],
        'cmdclass': {"build_ext": build_ext},
    }

    dist = Distribution(package_config)
    dist.script_args = ['build_ext'] + args
    dist.parse_command_line()
    dist.run_command('build_ext')


if __name__ == '__main__':
    build_ginac_interface(sys.argv[1:])
