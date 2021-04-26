

def build_appsi():
    print('\n\n**** Building APPSI ****')
    import setuptools
    from distutils.dist import Distribution
    import shutil
    import glob
    import os
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    import pybind11.setup_helpers

    original_pybind11_setup_helpers_macos = pybind11.setup_helpers.MACOS
    pybind11.setup_helpers.MACOS = False

    ext_modules = [Pybind11Extension("cmodel.cmodel",
                                     ['cmodel/src/expression.cpp',
                                      'cmodel/src/common.cpp',
                                      'cmodel/src/nl_writer.cpp',
                                      'cmodel/src/lp_writer.cpp',
                                      'cmodel/src/cmodel_bindings.cpp'])]

    package_config = {'name': 'appsi',
                      'packages': list(),
                      'ext_modules': ext_modules,
                      'cmdclass': {"build_ext": build_ext}}
    dist = Distribution(package_config)
    try:
        basedir = os.path.abspath(os.path.curdir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)
        dist.run_command('build_ext')
        library = glob.glob("build/*/cmodel/cmodel.*")[0]
        shutil.copy(library, 'cmodel/')
    finally:
        os.chdir(basedir)
        pybind11.setup_helpers.MACOS = original_pybind11_setup_helpers_macos


class AppsiBuilder(object):
    def __call__(self, parallel):
        return build_appsi()


if __name__ == '__main__':
    build_appsi()
