#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Script to generate the installer for pyomo.
"""

import sys
import os


def _find_packages(path):
    """
    Generate a list of nested packages
    """
    pkg_list = []
    if not os.path.exists(path):
        return []
    if not os.path.exists(path+os.sep+"__init__.py"):
        return []
    else:
        pkg_list.append(path)
    for root, dirs, files in os.walk(path, topdown=True):
        if root in pkg_list and "__init__.py" in files:
            for name in dirs:
                if os.path.exists(root+os.sep+name+os.sep+"__init__.py"):
                    pkg_list.append(root+os.sep+name)
    return [pkg for pkg in map(lambda x:x.replace(os.sep, "."), pkg_list)]


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

requires = [
    'PyUtilib>=5.6.3',
    'appdirs',
    'ply',
    'six>=1.4',
    ]
if sys.version_info < (2, 7):
    requires.append('argparse')
    requires.append('unittest2')
    requires.append('ordereddict')

from setuptools import setup
import sys

if 'develop' in sys.argv:
    using_cython = False
else:
    using_cython = True
if '--with-cython' in sys.argv:
    using_cython = True
    sys.argv.remove('--with-cython')

ext_modules = []
if using_cython:
    try:
        import platform
        if not platform.python_implementation() == "CPython":
            raise RuntimeError()
        from Cython.Build import cythonize
        #
        # Note: The Cython developers recommend that you destribute C source
        # files to users.  But this is fine for evaluating the utility of Cython
        #
        import shutil
        files = ["pyomo/core/expr/expr_pyomo5.pyx", "pyomo/core/expr/numvalue.pyx", "pyomo/core/util.pyx", "pyomo/repn/standard_repn.pyx", "pyomo/repn/plugins/cpxlp.pyx", "pyomo/repn/plugins/gams_writer.pyx", "pyomo/repn/plugins/baron_writer.pyx", "pyomo/repn/plugins/ampl/ampl_.pyx"]
        for f in files:
            shutil.copyfile(f[:-1], f)
        ext_modules = cythonize(files)
    except:
        using_cython = False

packages = _find_packages('pyomo')

setup(name='Pyomo',
      #
      # Note: trunk should have *next* major.minor
      #     VOTD and Final releases will have major.minor.revnum
      #
      # When cutting a release, ALSO update _major/_minor/_revnum in
      #
      #     pyomo/pyomo/version/__init__.py
      #     pyomo/RELEASE.txt
      #
      version='5.5.1',
      maintainer='William E. Hart',
      maintainer_email='wehart@sandia.gov',
      url='http://pyomo.org',
      license='BSD',
      platforms=["any"],
      description='Pyomo: Python Optimization Modeling Objects',
      long_description=read('README.txt'),
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: Jython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules' ],
      packages=packages,
      keywords=['optimization'],
      install_requires=requires,
      ext_modules = ext_modules,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
      entry_points="""
        [console_scripts]
        runbenders=pyomo.pysp.benders:Benders_main
        evaluate_xhat=pyomo.pysp.evaluate_xhat:EvaluateXhat_main
        runph=pyomo.pysp.phinit:PH_main
        runef=pyomo.pysp.ef_writer_script:main
        phsolverserver=pyomo.pysp.phsolverserver:main
        scenariotreeserver=pyomo.pysp.scenariotree.server_pyro:main
        computeconf=pyomo.pysp.computeconf:main

        results_schema=pyomo.scripting.commands:results_schema
        pyro_mip_server = pyomo.scripting.pyro_mip_server:main
        test.pyomo = pyomo.scripting.runtests:runPyomoTests
        pyomo = pyomo.scripting.pyomo_main:main
        pyomo_ns = pyomo.scripting.commands:pyomo_ns
        pyomo_nsc = pyomo.scripting.commands:pyomo_nsc
        kill_pyro_mip_servers = pyomo.scripting.commands:kill_pyro_mip_servers
        launch_pyro_mip_servers = pyomo.scripting.commands:launch_pyro_mip_servers
        readsol = pyomo.scripting.commands:readsol
        OSSolverService = pyomo.scripting.commands:OSSolverService
        pyomo_python = pyomo.scripting.commands:pyomo_python
        pyomo_old=pyomo.scripting.pyomo_command:main
        get_pyomo_extras = scripts.get_pyomo_extras:main

        [pyomo.command]
        pyomo.runbenders=pyomo.pysp.benders
        pyomo.evaluate_xhat=pyomo.pysp.evaluate_xhat
        pyomo.runph=pyomo.pysp.phinit
        pyomo.runef=pyomo.pysp.ef_writer_script
        pyomo.phsolverserver=pyomo.pysp.phsolverserver
        pyomo.scenariotreeserver=pyomo.pysp.scenariotree.server_pyro
        pyomo.computeconf=pyomo.pysp.computeconf

        pyomo.help = pyomo.scripting.driver_help
        pyomo.test.pyomo = pyomo.scripting.runtests
        pyomo.pyro_mip_server = pyomo.scripting.pyro_mip_server
        pyomo.results_schema=pyomo.scripting.commands
      """
      )
