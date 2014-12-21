#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# billups.py
#
# Problem adapted from billups.gms
# http://ftp.cs.wisc.edu/pub/mcplib/gams/
#

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import Complementarity


def pyomo_create_model(**kwargs):
    M = ConcreteModel()
    M.x = Var()
    M.c = Complementarity(expr=((M.x - 1.0)*(M.x - 1.0) - 1.01 >= 0, M.x >= 0))
    return M

