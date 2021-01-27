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
# Unit Tests for Pyomo tutorials
#

import runpy
import sys
import os
from os.path import abspath, dirname
topdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
currdir = dirname(abspath(__file__))+os.sep
tutorial_dir=topdir+os.sep+"examples"+os.sep+"pyomo"+os.sep+"tutorials"+os.sep

import pyutilib.th as unittest

try:
    from win32com.client.dynamic import Dispatch
    _win32com=True
except:
    _win32com=False #pragma:nocover

if _win32com:
    from pyutilib.excel.spreadsheet_win32com import ExcelSpreadsheet_win32com
    tmp = ExcelSpreadsheet_win32com()
    try:
        tmp._excel_dispatch()
        tmp._excel_quit()
        _excel_available = True
    except:
        _excel_available = False

try:
    import xlrd
    _xlrd=True
except:
    _xlrd=False
try:
    import openpyxl
    _openpyxl=True
except:
    _openpyxl=False


class PyomoTutorials(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp_path = list(sys.path)
        os.chdir(tutorial_dir)
        sys.path = [os.path.dirname(tutorial_dir)] + sys.path
        sys.path.append(os.path.dirname(tutorial_dir))
        sys.stderr.flush()
        sys.stdout.flush()
        self.save_stdout = sys.stdout
        self.save_stderr = sys.stderr

    def tearDown(self):
        os.chdir(self.cwd)
        sys.path = self.tmp_path
        sys.stdout = self.save_stdout
        sys.stderr = self.save_stderr

    def construct(self,filename):
        pass

    def test_data(self):
        OUTPUT = open(currdir+"data.log", 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module('data', None, "__main__")
        OUTPUT.close()
        self.assertIn(open(tutorial_dir+"data.out", 'r').read(),
                      open(currdir+"data.log", 'r').read())
        os.remove(currdir+"data.log")

    @unittest.skipIf(not ((_win32com and _excel_available) or _xlrd or _openpyxl), "Cannot read excel file.")
    def test_excel(self):
        OUTPUT = open(currdir+"excel.log", 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module('excel', None, "__main__")
        OUTPUT.close()
        self.assertIn(open(tutorial_dir+"excel.out", 'r').read(),
                      open(currdir+"excel.log", 'r').read())
        os.remove(currdir+"excel.log")

    def test_set(self):
        OUTPUT = open(currdir+"set.log", 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module('set', None, "__main__")
        OUTPUT.close()
        self.assertIn(open(tutorial_dir+"set.out", 'r').read(),
                      open(currdir+"set.log", 'r').read())
        os.remove(currdir+"set.log")

    def test_table(self):
        OUTPUT = open(currdir+"table.log", 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module('table', None, "__main__")
        OUTPUT.close()
        self.assertIn(open(tutorial_dir+"table.out", 'r').read(),
                      open(currdir+"table.log", 'r').read())
        os.remove(currdir+"table.log")

    def test_param(self):
        OUTPUT = open(currdir+"param.log", 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module('param', None, "__main__")
        OUTPUT.close()
        self.assertIn(open(tutorial_dir+"param.out", 'r').read(),
                      open(currdir+"param.log", 'r').read())
        os.remove(currdir+"param.log")

if __name__ == "__main__":
    unittest.main()
