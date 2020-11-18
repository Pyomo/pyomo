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

import os
from os.path import abspath, dirname
topdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
currdir = dirname(abspath(__file__))+os.sep
tutorial_dir=topdir+os.sep+"examples"+os.sep+"pyomo"+os.sep+"tutorials"+os.sep
import subprocess

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
        pass

    def construct(self, filename):
        pass

    def test_data(self):
        with open(currdir+"data.log", 'w') as f:
            os.chdir(tutorial_dir)
            subprocess.call(['python', tutorial_dir+"data.py"], stdout=f)
            os.chdir(currdir)
        self.assertFileEqualsBaseline(currdir+"data.log", tutorial_dir+"data.out")

    @unittest.skipIf(not ((_win32com and _excel_available) or _xlrd or _openpyxl), "Cannot read excel file.")
    def test_excel(self):
        with open(currdir+"excel.log", 'w') as f:
            os.chdir(tutorial_dir)
            subprocess.call(['python', tutorial_dir+"excel.py"], stdout=f)
            os.chdir(currdir)
        self.assertFileEqualsBaseline(currdir+"excel.log", tutorial_dir+"excel.out")

    def test_set(self):
        with open(currdir+"set.log", 'w') as f:
            os.chdir(tutorial_dir)
            subprocess.call(['python', tutorial_dir+"set.py"], stdout=f)
            os.chdir(currdir)
        self.assertFileEqualsBaseline(currdir+"set.log", tutorial_dir+"set.out")

    def test_table(self):
        with open(currdir+"table.log", 'w') as f:
            os.chdir(tutorial_dir)
            subprocess.call(['python', tutorial_dir+"table.py"], stdout=f)
            os.chdir(currdir)
        self.assertFileEqualsBaseline(currdir+"table.log", tutorial_dir+"table.out")

    def test_param(self):
        with open(currdir+"param.log", 'w') as f:
            os.chdir(tutorial_dir)
            subprocess.call(['python', tutorial_dir+"param.py"], stdout=f)
            os.chdir(currdir)
        self.assertFileEqualsBaseline(currdir+"param.log", tutorial_dir+"param.out")


if __name__ == "__main__":
    unittest.main()
