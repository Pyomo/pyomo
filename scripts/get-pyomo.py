#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# A script to install Pyomo and its dependencies (PyUtilib, ply, etc)
#

import sys
try:
    import pip
except ImportError:
    print("You must have 'pip' installed to run this script.")
    raise SystemExit


print("Installing Pyomo ...")
pip.main(['install']+sys.argv[1:]+['Pyomo'])
