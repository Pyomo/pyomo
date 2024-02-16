#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import glob
import os
from pyomo.common.dependencies import attempt_import, matplotlib_available
from pyomo.common.fileutils import this_file_dir
import pyomo.environ as pyo


currdir = this_file_dir()

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

# Needed for testing (triggers matplotlib import and switches its backend):
bool(matplotlib_available)


class TestOnlineDocExamples(unittest.BaselineTestDriver, unittest.TestCase):
    # Only test files in directories ending in -ch. These directories
    # contain the updated python and scripting files corresponding to
    # each chapter in the book.
    py_tests, sh_tests = unittest.BaselineTestDriver.gather_tests(
        list(filter(os.path.isdir, glob.glob(os.path.join(currdir, '*'))))
    )

    solver_dependencies = {
        'test_data_pyomo_diet1': ['glpk'],
        'test_data_pyomo_diet2': ['glpk'],
        'test_kernel_examples': ['glpk'],
    }
    # Note on package dependencies: two tests actually need
    # pyutilib.excel.spreadsheet; however, the pyutilib importer is
    # broken on Python>=3.12, so instead of checking for spreadsheet, we
    # will check for pyutilib.component, which triggers the importer
    # (and catches the error on 3.12)
    package_dependencies = {
        # data
        'test_data_ABCD9': ['pyodbc'],
        'test_data_ABCD8': ['pyodbc'],
        'test_data_ABCD7': ['win32com', 'pyutilib.component'],
        # dataportal
        'test_dataportal_dataportal_tab': ['xlrd', 'pyutilib.component'],
        'test_dataportal_set_initialization': ['numpy'],
        'test_dataportal_param_initialization': ['numpy'],
        # kernel
        'test_kernel_examples': ['pympler'],
    }

    @parameterized.parameterized.expand(
        sh_tests, name_func=unittest.BaselineTestDriver.custom_name_func
    )
    def test_sh(self, tname, test_file, base_file):
        self.shell_test_driver(tname, test_file, base_file)

    @parameterized.parameterized.expand(
        py_tests, name_func=unittest.BaselineTestDriver.custom_name_func
    )
    def test_py(self, tname, test_file, base_file):
        self.python_test_driver(tname, test_file, base_file)


# Execute the tests
if __name__ == '__main__':
    unittest.main()
