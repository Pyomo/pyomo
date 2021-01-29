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

def read(*rnames):
    with open(os.path.join(os.path.dirname(__file__), *rnames)) as README:
        # Strip all leading badges up to, but not including the COIN-OR
        # badge so that they do not appear in the PyPI description
        while True:
            line = README.readline()
            if 'COIN-OR' in line:
                break
            if line.strip() and '[![' not in line:
                break
        return line + README.read()

def get_version():
    # Source pyomo/version/info.py to get the version number
    _verInfo = dict(globals())
    _verFile = os.path.join(os.path.dirname(__file__),
                            'pyomo','version','info.py')
    with open(_verFile) as _FILE:
        exec(_FILE.read(), _verInfo)
    return _verInfo['__version__']

from setuptools import setup, find_packages

CYTHON_REQUIRED = "required"
if 'develop' in sys.argv:
    using_cython = False
else:
    using_cython = "automatic"
if '--with-cython' in sys.argv:
    using_cython = CYTHON_REQUIRED
    sys.argv.remove('--with-cython')
if '--without-cython' in sys.argv:
    using_cython = False
    sys.argv.remove('--without-cython')

ext_modules = []
if using_cython:
    try:
        import platform
        if platform.python_implementation() != "CPython":
            # break out of this try-except (disable Cython)
            raise RuntimeError("Cython is only supported under CPython")
        from Cython.Build import cythonize
        #
        # Note: The Cython developers recommend that you destribute C source
        # files to users.  But this is fine for evaluating the utility of Cython
        #
        import shutil
        files = [
            "pyomo/core/expr/numvalue.pyx",
            "pyomo/core/expr/numeric_expr.pyx",
            "pyomo/core/expr/logical_expr.pyx",
            #"pyomo/core/expr/visitor.pyx",
            "pyomo/core/util.pyx",
            "pyomo/repn/standard_repn.pyx",
            "pyomo/repn/plugins/cpxlp.pyx",
            "pyomo/repn/plugins/gams_writer.pyx",
            "pyomo/repn/plugins/baron_writer.pyx",
            "pyomo/repn/plugins/ampl/ampl_.pyx",
        ]
        for f in files:
            shutil.copyfile(f[:-1], f)
        ext_modules = cythonize(files, compiler_directives={
            "language_level": 3 if sys.version_info >= (3, ) else 2})
    except:
        if using_cython == CYTHON_REQUIRED:
            print("""
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
""")
            raise
        using_cython = False

def run_setup():
   setup(name='Pyomo',
      #
      # Note: the release number is set in pyomo/version/info.py
      #
      version=get_version(),
      maintainer='Pyomo Developer Team',
      maintainer_email='pyomo-developers@googlegroups.com',
      url='http://pyomo.org',
      license='BSD',
      platforms=["any"],
      description='Pyomo: Python Optimization Modeling Objects',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      keywords=['optimization'],
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: Jython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules' ],
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
      install_requires=[
          'PyUtilib>=6.0.0',
          'enum34;python_version<"3.4"',
          'ply',
          'six>=1.4',
      ],
      packages=find_packages(exclude=("scripts",)),
      package_data={"pyomo.contrib.viewer":["*.ui"]},
      ext_modules = ext_modules,
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
        pyomo = pyomo.scripting.pyomo_main:main_console_script
        pyomo_ns = pyomo.scripting.commands:pyomo_ns
        pyomo_nsc = pyomo.scripting.commands:pyomo_nsc
        kill_pyro_mip_servers = pyomo.scripting.commands:kill_pyro_mip_servers
        launch_pyro_mip_servers = pyomo.scripting.commands:launch_pyro_mip_servers
        readsol = pyomo.scripting.commands:readsol
        OSSolverService = pyomo.scripting.commands:OSSolverService
        pyomo_python = pyomo.scripting.commands:pyomo_python
        pyomo_old=pyomo.scripting.pyomo_command:main

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
        pyomo.viewer=pyomo.contrib.viewer.pyomo_viewer
      """
      )

try:
    run_setup()
except SystemExit as e_info:
    # Cython can generate a SystemExit exception on Windows if the
    # environment is missing / has an incorrect Microsoft compiler.
    # Since Cython is not strictly required, we will disable Cython and
    # try re-running setup(), but only for this very specific situation.
    if 'Microsoft Visual C++' not in str(e_info):
        raise
    elif using_cython == CYTHON_REQUIRED:
        print("""
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
""")
        raise
    else:
        print("""
ERROR: setup() failed:
    %s
Re-running setup() without the Cython modules
""" % (str(e_info),))
        ext_modules = []
        run_setup()
        print("""
WARNING: Installation completed successfully, but the attempt to cythonize
         core Pyomo modules failed.  Cython provides performance
         optimizations and is not required for any Pyomo functionality.
         Cython returned the following error:
   "%s"
""" % (str(e_info),))
