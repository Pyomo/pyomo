#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()
model.X = Var()

if model.X >= 10.0:
    pass
if value(model.X) >= 10.0:
    pass
if model.X >= 10.0:
    pass

if model.X >= 10.0:
    if model.X >= 10.0:
        pass
