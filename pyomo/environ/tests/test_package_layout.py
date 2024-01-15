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
#
# Unit Tests for pyomo.base.misc
#

import glob
import importlib
import os
import subprocess
import sys

from itertools import filterfalse
from os.path import join

import pyomo.common.dependencies as dependencies
from pyomo.common.fileutils import PYOMO_ROOT_DIR

import pyomo.common.unittest as unittest

parameterized, param_available = dependencies.attempt_import('parameterized')
parameterized = parameterized.parameterized

_FAST_TEST = False

# List of directories under `pyomo` that intentionally do NOT have
# __init__.py files (because they either contain no Python files - or
# contain Python files that are only used in testing and explicitly NOT
# part of the "Pyomo package")
_NON_MODULE_DIRS = {
    join('contrib', 'ampl_function_demo', 'src'),
    join('contrib', 'appsi', 'cmodel', 'src'),
    join('contrib', 'pynumero', 'src'),
    join('core', 'tests', 'data', 'baselines'),
    join('core', 'tests', 'diet', 'baselines'),
    join('opt', 'tests', 'solver', 'exe_dir'),
    join('solvers', 'tests', 'checks', 'data'),
    join('solvers', 'tests', 'mip', 'asl'),
    join('solvers', 'tests', 'mip', 'cbc'),
    join('solvers', 'tests', 'mip', 'cplex'),
    join('solvers', 'tests', 'mip', 'glpk'),
    join('solvers', 'tests', 'piecewise_linear', 'baselines'),
    join('solvers', 'tests', 'piecewise_linear', 'kernel_baselines'),
    join('solvers', 'tests', 'piecewise_linear', 'kernel_problems'),
    join('solvers', 'tests', 'piecewise_linear', 'problems'),
}

_DO_NOT_IMPORT_MODULES = {
    'pyomo.common.tests.dep_mod_except',
    'pyomo.contrib.interior_point.examples.ex1',
}

try:
    _cwd = os.getcwd()
    os.chdir(os.path.join(PYOMO_ROOT_DIR, 'pyomo'))
    modules = sorted(
        os.path.join('pyomo', os.path.splitext(fname)[0]).replace(os.path.sep, '.')
        for fname in glob.glob(os.path.join('**', '*.py'), recursive=True)
        if not fname.endswith('__init__.py')
    )
    modules = list(filterfalse(_DO_NOT_IMPORT_MODULES.__contains__, modules))
finally:
    os.chdir(_cwd)


import_test = """
import pyomo.common.dependencies
pyomo.common.dependencies.SUPPRESS_DEPENDENCY_WARNINGS = True
import unittest
try:
    import %s
except unittest.case.SkipTest as e:
    # suppress the exception, but print the message
    print(e)
"""


class TestPackageLayout(unittest.TestCase):
    def test_for_init_files(self):
        _NMD = set(_NON_MODULE_DIRS)
        fail = []
        module_dir = os.path.join(PYOMO_ROOT_DIR, 'pyomo')
        for path, subdirs, files in os.walk(module_dir):
            assert path.startswith(module_dir)
            relpath = path[1 + len(module_dir) :]
            # Skip all __pycache__ directories
            try:
                subdirs.remove('__pycache__')
            except ValueError:
                pass

            if '__init__.py' in files:
                continue

            if relpath in _NMD:
                _NMD.remove(relpath)
                # Skip checking all subdirectories
                subdirs[:] = []
                continue

            fail.append(relpath)

        if fail:
            self.fail(
                "Directories are missing __init__.py files:\n\t"
                + "\n\t".join(sorted(fail))
            )
        if _NMD:
            self.fail(
                "_NON_MODULE_DIRS contains entries not found in package "
                "or unexpectedly contain a __init__.py file:\n\t"
                + "\n\t".join(sorted(_NMD))
            )

    @parameterized.expand(modules)
    @unittest.pytest.mark.importtest
    def test_module_import(self, module):
        # We will go through the entire package and ensure that all the
        # python modules are a least importable.  This is important to
        # be tested on the newest Python version (in part to catch
        # deprecation warnings before they become fatal parse errors).
        module_file = (
            os.path.join(PYOMO_ROOT_DIR, module.replace('.', os.path.sep)) + '.py'
        )
        # we need to delete the .pyc file, because some things (like
        # invalid docstrings) only toss the warning when the module is
        # initially byte-compiled.
        pyc = importlib.util.cache_from_source(module_file)
        if os.path.isfile(pyc):
            os.remove(pyc)
        test_code = import_test % module
        if _FAST_TEST:
            # This is much faster, as it only reloads each module once
            # (no subprocess, and no reloading and dependent modules).
            # However, it will generate false positives when reimporting
            # a single module creates side effects (this happens in some
            # of the testing harness for auto-registered test cases)
            from pyomo.common.fileutils import import_file
            import warnings

            try:
                _dep_warn = dependencies.SUPPRESS_DEPENDENCY_WARNINGS
                dependencies.SUPPRESS_DEPENDENCY_WARNINGS = True

                with warnings.catch_warnings():
                    warnings.resetwarnings()
                    warnings.filterwarnings('error')
                    import_file(module_file, clear_cache=True)
            except unittest.SkipTest as e:
                # suppress the exception, but print the message
                print(e)
            finally:
                dependencies.SUPPRESS_DEPENDENCY_WARNINGS = _dep_warn
        else:
            subprocess.run([sys.executable, '-Werror', '-c', test_code], check=True)
