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
#
# Unit Tests for Pyomo tutorials
#

import runpy
import sys
import os

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.tee import capture_output

tutorial_dir = os.path.join(PYOMO_ROOT_DIR, "examples", "pyomo", "tutorials")

try:
    from win32com.client.dynamic import Dispatch

    _win32com = True
except:
    _win32com = False  # pragma:nocover

from pyomo.common.dependencies import pyutilib, pyutilib_available

_excel_available = False
if _win32com and pyutilib_available:
    from pyutilib.excel.spreadsheet_win32com import ExcelSpreadsheet_win32com

    tmp = ExcelSpreadsheet_win32com()
    try:
        tmp._excel_dispatch()
        tmp._excel_quit()
        _excel_available = True
    except:
        pass

try:
    import xlrd

    _xlrd = True
except:
    _xlrd = False
try:
    import openpyxl

    _openpyxl = True
except:
    _openpyxl = False


class PyomoTutorials(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp_path = list(sys.path)
        os.chdir(tutorial_dir)
        sys.path = [tutorial_dir] + sys.path

    def tearDown(self):
        os.chdir(self.cwd)
        sys.path = self.tmp_path

    def driver(self, name):
        with capture_output() as OUT:
            runpy.run_module(name, None, "__main__")
        self.assertIn(
            open(os.path.join(tutorial_dir, name + ".out"), 'r').read(), OUT.getvalue()
        )

    def test_data(self):
        self.driver('data')

    @unittest.skipUnless(_xlrd or _openpyxl, "Cannot read excel file.")
    @unittest.skipUnless(
        _win32com and _excel_available and pyutilib_available, "Cannot read excel file."
    )
    def test_excel(self):
        self.driver('excel')

    def test_set(self):
        self.driver('set')

    def test_table(self):
        self.driver('table')

    def test_param(self):
        self.driver('param')


if __name__ == "__main__":
    unittest.main()
