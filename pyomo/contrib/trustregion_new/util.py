#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from numpy import array

def copyVector(x, y, z):
    """
    This function is to create a hard copy of vector x, y, z.
    """
    return array(x), array(y), array(z)

def minIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return a
    return b

def maxIgnoreNone(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return b
    return a
