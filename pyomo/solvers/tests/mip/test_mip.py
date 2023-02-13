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
# Tests driven by test_mip.yml
#

import os
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest

if __name__ == "__main__":
    unittest.main()
