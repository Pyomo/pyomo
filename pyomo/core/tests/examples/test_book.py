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
test_dir=topdir+os.sep+"examples"+os.sep+"doc"+os.sep+"pyomobook"+os.sep

import unittest
import sys

os.chdir(test_dir)
sys.path.append(test_dir)

if __name__ == "__main__":
    unittest.main()
