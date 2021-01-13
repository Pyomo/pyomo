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

import subprocess
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
        os.chdir(tutorial_dir)

    def tearDown(self):
        os.chdir(self.cwd)

    def construct(self,filename):
        pass

    def test_data(self):
        with open(currdir+"data.log", 'w') as f:
            subprocess.run([sys.executable, tutorial_dir+'data.py'], stdout=f)
        self.assertIn(open(tutorial_dir+"data.out", 'r').read(),
                      open(currdir+"data.log", 'r').read())

    @unittest.skipIf(not ((_win32com and _excel_available) or _xlrd or _openpyxl), "Cannot read excel file.")
    def test_excel(self):
        with open(currdir+'excel.log', 'w') as f:
            subprocess.run([sys.executable, tutorial_dir+'excel.py'], stdout=f)
        self.assertIn(open(tutorial_dir+"excel.out", 'r').read(),
                      open(currdir+"excel.log", 'r').read())

    def test_set(self):
        with open(currdir+'set.log', 'w') as f:
            subprocess.run([sys.executable, tutorial_dir+'set.py'], stdout=f)
        self.assertIn(open(tutorial_dir+"set.out", 'r').read(),
                      open(currdir+"set.log", 'r').read())

    def test_table(self):
        with open(currdir+'table.log', 'w') as f:
            subprocess.run([sys.executable, tutorial_dir+'table.py'], stdout=f)
        self.assertIn(open(tutorial_dir+"table.out", 'r').read(),
                      open(currdir+"table.log", 'r').read())

    def test_param(self):
        with open(currdir+'param.log', 'w') as f:
            subprocess.run([sys.executable, tutorial_dir+'param.py'], stdout=f)
        self.assertIn(open(tutorial_dir+"param.out", 'r').read(),
                      open(currdir+"param.log", 'r').read())

if __name__ == "__main__":
    unittest.main()
