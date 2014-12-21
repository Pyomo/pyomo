#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys

try:
    import pyomo
    import pyomo.environ
    import pyomo.core
    print("OK")
except Exception:
    e = sys.exc_info()[1]
    print("Pyomo package error: "+str(e))
