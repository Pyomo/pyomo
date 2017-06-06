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
# Unit Tests for complete examples
#

import os
from os.path import abspath, dirname

topdir = dirname(dirname(abspath(__file__)))+os.sep+".."+os.sep+".."
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *

class TestAmplbook2(unittest.TestCase): pass

#
# DISABLED FOR NOW ... Revisit these when the ipconvert utility is stable...
#

#data_dir=topdir+os.sep+"examples"+os.sep+"pyomo"+os.sep+"amplbook2"+os.sep
#files = glob.glob(data_dir+"*.py")
#for file in files:
#    bname=os.path.basename(file)
#    name=bname.split('.')[0]
#    TestAmplbook2.add_commandline_test(cmd="cd "+data_dir+"; "+topdir+os.sep+"scripts/pyomo "+bname+" "+name+".dat", baseline=data_dir+name+".log", name=name)

if __name__ == "__main__":
    unittest.main()
