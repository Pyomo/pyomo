#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Pyomo tutorials
#

import os
from os.path import abspath, dirname
topdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
currdir = dirname(abspath(__file__))+os.sep
tutorial_dir=topdir+os.sep+"examples"+os.sep+"pyomo"+os.sep+"tutorials"+os.sep

import pyutilib.misc
import pyutilib.th as unittest

from pyomo.environ import *

try:
    from win32com.client.dynamic import Dispatch
    _win32com=True
except:
    _win32com=False #pragma:nocover
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

    def construct(self,filename):
        pass

    def test_data(self):
        pyutilib.misc.run_file(tutorial_dir+"data.py", logfile=currdir+"data.log", execdir=tutorial_dir)
        self.assertFileEqualsBaseline(currdir+"data.log", tutorial_dir+"data.out")

    @unittest.skipIf(not (_win32com or _xlrd or _openpyxl), "Cannot real excel file.")
    def test_excel(self):
        pyutilib.misc.run_file(tutorial_dir+"excel.py", logfile=currdir+"excel.log", execdir=tutorial_dir)
        self.assertFileEqualsBaseline(currdir+"excel.log", tutorial_dir+"excel.out")

    def test_set(self):
        pyutilib.misc.run_file(tutorial_dir+"set.py", logfile=currdir+"set.log", execdir=tutorial_dir)
        self.assertFileEqualsBaseline(currdir+"set.log", tutorial_dir+"set.out")

    def test_table(self):
        pyutilib.misc.run_file(tutorial_dir+"table.py", logfile=currdir+"table.log", execdir=tutorial_dir)
        self.assertFileEqualsBaseline(currdir+"table.log", tutorial_dir+"table.out")

    def test_param(self):
        pyutilib.misc.run_file(tutorial_dir+"param.py", logfile=currdir+"param.log", execdir=tutorial_dir)
        self.assertFileEqualsBaseline(currdir+"param.log", tutorial_dir+"param.out")

if __name__ == "__main__":
    unittest.main()
