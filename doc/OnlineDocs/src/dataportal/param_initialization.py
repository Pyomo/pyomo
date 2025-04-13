#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import numpy

model = pyo.ConcreteModel()

# @decl1
model.a = pyo.Param(initialize=1.1)
# @decl1

# Initialize with a dictionary
# @decl2
model.b = pyo.Param([1, 2, 3], initialize={1: 1, 2: 2, 3: 3})
# @decl2


# Initialize with a function that returns native Python data
# @decl3
def c(model):
    return {1: 1, 2: 2, 3: 3}


model.c = pyo.Param([1, 2, 3], initialize=c)
# @decl3

model.pprint(verbose=True)
