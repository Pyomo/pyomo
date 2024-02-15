#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# pythonconditional.py

# @all:
x = 6
y = False

if x == 5:
    print("x happens to be 5")
    print("for what that is worth")
elif y:
    print("x is not 5, but at least y is True")
else:
    print("This program cannot tell us much.")
# @:all
