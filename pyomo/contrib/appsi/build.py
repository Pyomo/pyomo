from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11.setup_helpers
import shutil
import glob

pybind11.setup_helpers.MACOS = False

ext_modules = [Pybind11Extension("cmodel.cmodel",
                                 ['cmodel/src/expression.cpp',
                                  'cmodel/src/common.cpp',
                                  'cmodel/src/nl_writer.cpp',
                                  'cmodel/src/lp_writer.cpp',
                                  'cmodel/src/cmodel_bindings.cpp'])]

setup(name='appsi',
      version='0.1.0',
      packages=find_packages(),
      description='APPSI: Auto Persistent Pyomo Solver Interfaces',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      author='Michael Bynum',
      maintainer_email='mlbynum@sandia.gov',
      license='Revised BSD',
      url='TBD',
      install_requires=['pyomo'],
      include_package_data=True,
      ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext},
      python_requires='>=3.6',
      classifiers=["Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent"]
      )

library = glob.glob("build/*/cmodel/cmodel.*")[0]
shutil.copy(library, 'cmodel/')
