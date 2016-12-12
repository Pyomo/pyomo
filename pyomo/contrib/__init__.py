#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Conditionally adding sub-packages for Pyomo contributions that may
# be installed alongside of Pyomo.
#

try:
    import pyomo_simplemodel as simplemodel
except:
    pass

