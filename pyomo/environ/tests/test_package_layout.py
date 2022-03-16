#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.base.misc
#
import os
from os.path import join

from pyomo.common.fileutils import PYOMO_ROOT_DIR

import pyomo.common.unittest as unittest


# List of directories under `pyomo` that intentionally do NOT have
# __init__.py files (because they either contain no Python files - or
# contain Python files that are only used in testing and explicitly NOT
# part of the "Pyomo package")
_NON_MODULE_DIRS = {
    join('checker', 'doc'),
    join('checker', 'tests', 'examples'),
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

class TestPackageLayout(unittest.TestCase):
    def test_for_init_files(self):
        _NMD = set(_NON_MODULE_DIRS)
        fail = []
        module_dir = os.path.join(PYOMO_ROOT_DIR, 'pyomo')
        for path, subdirs, files in os.walk(module_dir):
            assert path.startswith(module_dir)
            relpath = path[1+len(module_dir):]
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


