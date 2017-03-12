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
test_dir=topdir+os.sep+"examples"+os.sep+"doc"+os.sep+"pyomobook"+os.sep

import unittest
import sys

os.chdir(test_dir)
sys.path.append(test_dir)
from test_book_examples import *

if __name__ == "__main__":
    unittest.main()
