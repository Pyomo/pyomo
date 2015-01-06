#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *

# Trivial model
def define_model():

    model = ConcreteModel()
    model.x = Var()
    model.obj = Objective(expr=model.x)
    model.con = Constraint(expr=model.x >= 1)

    return model
