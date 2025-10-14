#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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
from pathlib import Path
from setuptools import setup, find_packages, Command

try:
    # This works beginning in setuptools 40.7.0 (27 Jan 2019)
    from setuptools import DistutilsOptionError
except ImportError:
    # Needed for setuptools prior to 40.7.0
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


def check_config_arg(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    if name in os.getenv('PYOMO_SETUP_ARGS', '').split():
        return True
    return False


CYTHON_REQUIRED = "required"
if not any(
    arg.startswith(cmd)
    for cmd in ('build', 'install', 'bdist', 'wheel')
    for arg in sys.argv
):
    using_cython = False
elif sys.version_info[:2] < (3, 11):
    using_cython = "automatic"
else:
    using_cython = False
if check_config_arg('--with-cython'):
    using_cython = CYTHON_REQUIRED
if check_config_arg('--without-cython'):
    using_cython = False

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

if check_config_arg('--with-distributable-extensions'):
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
        class version_cmp:
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
    cmdclass={'dependencies': DependenciesCommand},
    version=get_version(),
    install_requires=['ply'],
    extras_require={
        # There are certain tests that also require pytest-qt, but because those
        # tests are so environment/machine specific, we are leaving these out of
        # the dependencies.
        'tests': ['coverage', 'parameterized', 'pybind11', 'pytest', 'pytest-parallel'],
        'docs': [
            'Sphinx>4,!=8.2.0',
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
            'linear-tree; python_version<"3.14"',  # contrib.piecewise
            # FIXME: This is a temporary pin that should be removed
            # when the linear-tree dependency is replaced
            'scikit-learn<1.7.0; implementation_name!="pypy" and python_version<"3.14"',
            'scikit-learn; implementation_name!="pypy" and python_version>="3.14"',
            # Note: matplotlib 3.6.1 has bug #24127, which breaks
            # seaborn's histplot (triggering parmest failures)
            # Note: minimum version from community_detection use of
            # matplotlib.pyplot.get_cmap()
            'matplotlib>=3.6.0,!=3.6.1',
            'networkx',  # network, incidence_analysis, community_detection
            'numpy',
            'openpyxl',  # dataportals
            'packaging',  # for checking other dependency versions
            #'pathos',   # requested for #963, but PR currently closed
            'pint',  # units
            'plotly',  # incidence_analysis
            'python-louvain',  # community_detection
            'pyyaml',  # core
            # qtconsole also requires a supported Qt version (PyQt5 or PySide6).
            # Because those are environment specific, we have left that out here.
            'qtconsole',  # contrib.viewer
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
            'casadi; implementation_name!="pypy"',  # dae
            'numdifftools; implementation_name!="pypy"',  # pynumero
            'pandas; implementation_name!="pypy"',
            'seaborn; implementation_name!="pypy"',  # parmest.graphics
        ],
    },
    packages=find_packages(exclude=("scripts",)),
    package_data={
        "pyomo.contrib.ampl_function_demo": ["src/*"],
        "pyomo.contrib.appsi.cmodel": ["src/*"],
        "pyomo.contrib.cspline_external": ["src/*"],
        "pyomo.contrib.aslfunctions": ["src/*"],
        "pyomo.contrib.mcpp": ["*.cpp"],
        "pyomo.contrib.pynumero": ['src/*', 'src/tests/*'],
        "pyomo.contrib.viewer": ["*.ui"],
        "pyomo.contrib.simplification.ginac": ["src/*.cpp", "src/*.hpp"],
    },
    ext_modules=ext_modules,
)


try:
    # setuptools.build_meta (>=68) forbids absolute paths in the `sources=` list.
    # This resets the extensions (only for those items that are absolute paths)
    # to use relative paths
    ROOT = Path(__file__).parent.resolve()
    for ext in ext_modules:
        rel_sources = []
        for src in ext.sources:
            p = Path(src)
            if p.is_absolute():
                p = p.relative_to(ROOT)
            rel_sources.append(p.as_posix())
        ext.sources[:] = rel_sources
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
