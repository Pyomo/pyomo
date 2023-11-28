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

"""
Script to generate the installer for pyomo.
"""

import os
import platform
import sys
from setuptools import setup, find_packages, Command

try:
    from setuptools import DistutilsOptionError
except ImportError:
    from distutils.errors import DistutilsOptionError


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


def import_pyomo_module(*path):
    _module_globals = dict(globals())
    _module_globals['__name__'] = None
    _source = os.path.join(os.path.dirname(__file__), *path)
    with open(_source) as _FILE:
        exec(_FILE.read(), _module_globals)
    return _module_globals


def get_version():
    # Source pyomo/version/info.py to get the version number
    return import_pyomo_module('pyomo', 'version', 'info.py')['__version__']


CYTHON_REQUIRED = "required"
if not any(
    arg.startswith(cmd) for cmd in ('build', 'install', 'bdist') for arg in sys.argv
):
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
        if platform.python_implementation() != "CPython":
            # break out of this try-except (disable Cython)
            raise RuntimeError("Cython is only supported under CPython")
        from Cython.Build import cythonize

        #
        # Note: The Cython developers recommend that you distribute C source
        # files to users.  But this is fine for evaluating the utility of Cython
        #
        import shutil

        files = [
            "pyomo/core/expr/numvalue.pyx",
            "pyomo/core/expr/numeric_expr.pyx",
            "pyomo/core/expr/logical_expr.pyx",
            # "pyomo/core/expr/visitor.pyx",
            "pyomo/core/util.pyx",
            "pyomo/repn/standard_repn.pyx",
            "pyomo/repn/plugins/cpxlp.pyx",
            "pyomo/repn/plugins/gams_writer.pyx",
            "pyomo/repn/plugins/baron_writer.pyx",
            "pyomo/repn/plugins/ampl/ampl_.pyx",
        ]
        for f in files:
            shutil.copyfile(f[:-1], f)
        ext_modules = cythonize(files, compiler_directives={"language_level": 3})
    except:
        if using_cython == CYTHON_REQUIRED:
            print(
                """
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
"""
            )
            raise
        using_cython = False

if ('--with-distributable-extensions' in sys.argv) or (
    os.getenv('PYOMO_SETUP_ARGS') is not None
    and '--with-distributable-extensions' in os.getenv('PYOMO_SETUP_ARGS')
):
    try:
        sys.argv.remove('--with-distributable-extensions')
    except:
        pass
    #
    # Import the APPSI extension builder
    # NOTE: There is inconsistent behavior in Windows for APPSI.
    # As a result, we will NOT include these extensions in Windows.
    if not sys.platform.startswith('win'):
        appsi_extension = import_pyomo_module('pyomo', 'contrib', 'appsi', 'build.py')[
            'get_appsi_extension'
        ](
            in_setup=True,
            appsi_root=os.path.join(
                os.path.dirname(__file__), 'pyomo', 'contrib', 'appsi'
            ),
        )
        ext_modules.append(appsi_extension)


class DependenciesCommand(Command):
    """Custom setuptools command

    This will output the list of dependencies, including any optional
    dependencies for 'extras_require` targets.  This is needed so that
    we can (relatively) easily extract what `pip install '.[optional]'`
    would have done so that we can pass it on to a 'conda install'
    command when setting up Pyomo testing in a conda environment
    (because conda for all intents does not acknowledge
    `extras_require`).

    """

    description = "list the dependencies for this package"
    user_options = [('extras=', None, 'extra targets to include')]

    def initialize_options(self):
        self.extras = None

    def finalize_options(self):
        if self.extras is not None:
            self.extras = [e for e in (_.strip() for _ in self.extras.split(',')) if e]
            for e in self.extras:
                if e not in setup_kwargs['extras_require']:
                    raise DistutilsOptionError(
                        "extras can only include {%s}"
                        % (', '.join(setup_kwargs['extras_require']))
                    )

    def run(self):
        deps = list(self._print_deps(setup_kwargs['install_requires']))
        if self.extras is not None:
            for e in self.extras:
                deps.extend(self._print_deps(setup_kwargs['extras_require'][e]))
        print(' '.join(deps))

    def _print_deps(self, deplist):
        class version_cmp(object):
            ver = tuple(map(int, platform.python_version_tuple()[:2]))

            def __lt__(self, other):
                return self.ver < tuple(map(int, other.split('.')))

            def __le__(self, other):
                return self.ver <= tuple(map(int, other.split('.')))

            def __gt__(self, other):
                return not self.__le__(other)

            def __ge__(self, other):
                return not self.__lt__(other)

            def __eq__(self, other):
                return self.ver == tuple(map(int, other.split('.')))

            def __ne__(self, other):
                return not self.__eq__(other)

        implementation_name = sys.implementation.name
        platform_system = platform.system()
        python_version = version_cmp()
        for entry in deplist:
            dep, _, condition = (_.strip() for _ in entry.partition(';'))
            if condition and not eval(condition):
                continue
            yield dep


setup_kwargs = dict(
    name='Pyomo',
    #
    # Note: the release number is set in pyomo/version/info.py
    #
    cmdclass={'dependencies': DependenciesCommand},
    version=get_version(),
    maintainer='Pyomo Developer Team',
    maintainer_email='pyomo-developers@googlegroups.com',
    url='http://pyomo.org',
    project_urls={
        'Documentation': 'https://pyomo.readthedocs.io/',
        'Source': 'https://github.com/Pyomo/pyomo',
    },
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=['ply'],
    extras_require={
        'tests': [
            #'codecov', # useful for testing infrastructures, but not required
            'coverage',
            'pytest',
            'pytest-parallel',
            'parameterized',
            'pybind11',
        ],
        'docs': [
            'Sphinx>4',
            'sphinx-copybutton',
            'sphinx_rtd_theme>0.5',
            'sphinxcontrib-jsmath',
            'sphinxcontrib-napoleon',
            'numpy',  # Needed by autodoc for pynumero
            'scipy',  # Needed by autodoc for pynumero
        ],
        'optional': [
            'dill',  # No direct use, but improves lambda pickle
            'ipython',  # contrib.viewer
            # Note: matplotlib 3.6.1 has bug #24127, which breaks
            # seaborn's histplot (triggering parmest failures)
            'matplotlib!=3.6.1',
            # network, incidence_analysis, community_detection
            # Note: networkx 3.2 is Python>-3.9, but there is a broken
            # 3.2 package on conda-forgethat will get implicitly
            # installed on python 3.8
            'networkx<3.2; python_version<"3.9"',
            'networkx; python_version>="3.9"',
            'numpy',
            'openpyxl',  # dataportals
            #'pathos',   # requested for #963, but PR currently closed
            'pint',  # units
            'plotly',  # incidence_analysis
            'python-louvain',  # community_detection
            'pyyaml',  # core
            'scipy',
            'sympy',  # differentiation
            'xlrd',  # dataportals
            'z3-solver',  # community_detection
            #
            # subprocess output is merged more reliably if
            # 'PeekNamedPipe' is available from pywin32
            'pywin32; platform_system=="Windows"',
            #
            # The following optional dependencies are difficult to
            # install on PyPy (binary wheels are not available), so we
            # will only "require" them on other (CPython) platforms:
            #
            # DAE can use casadi
            'casadi; implementation_name!="pypy"',
            'numdifftools; implementation_name!="pypy"',  # pynumero
            'pandas; implementation_name!="pypy"',
            'seaborn; implementation_name!="pypy"',  # parmest.graphics
        ],
    },
    packages=find_packages(exclude=("scripts",)),
    package_data={
        "pyomo.contrib.ampl_function_demo": ["src/*"],
        "pyomo.contrib.appsi.cmodel": ["src/*"],
        "pyomo.contrib.mcpp": ["*.cpp"],
        "pyomo.contrib.pynumero": ['src/*', 'src/tests/*'],
        "pyomo.contrib.viewer": ["*.ui"],
    },
    ext_modules=ext_modules,
    entry_points="""
    [console_scripts]
    pyomo = pyomo.scripting.pyomo_main:main_console_script

    [pyomo.command]
    pyomo.help = pyomo.scripting.driver_help
    pyomo.viewer=pyomo.contrib.viewer.pyomo_viewer
    """,
)


try:
    setup(**setup_kwargs)
except SystemExit as e_info:
    # Cython can generate a SystemExit exception on Windows if the
    # environment is missing / has an incorrect Microsoft compiler.
    # Since Cython is not strictly required, we will disable Cython and
    # try re-running setup(), but only for this very specific situation.
    if 'Microsoft Visual C++' not in str(e_info):
        raise
    elif using_cython == CYTHON_REQUIRED:
        print(
            """
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
"""
        )
        raise
    else:
        print(
            """
ERROR: setup() failed:
    %s
Re-running setup() without the Cython modules
"""
            % (str(e_info),)
        )
        setup_kwargs['ext_modules'] = []
        setup(**setup_kwargs)
        print(
            """
WARNING: Installation completed successfully, but the attempt to cythonize
         core Pyomo modules failed.  Cython provides performance
         optimizations and is not required for any Pyomo functionality.
         Cython returned the following error:
   "%s"
"""
            % (str(e_info),)
        )
