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

# Initialize with a list, tuple or set
# @decl2
model.A = pyo.Set(initialize=[2, 3, 5])
model.B = pyo.Set(initialize=set([2, 3, 5]))
model.C = pyo.Set(initialize=(2, 3, 5))
# @decl2

# Initialize with a generator
# @decl3
model.D = pyo.Set(initialize=range(9))
model.E = pyo.Set(initialize=(i for i in model.B if i % 2 == 0))
# @decl3

# Initialize with a numpy
# @decl4
f = numpy.array([2, 3, 5])
model.F = pyo.Set(initialize=f)
# @decl4


# Initialize with a function that returns native Python data
# @decl5
def g(model):
    return [2, 3, 5]


model.G = pyo.Set(initialize=g)
# @decl5

# Initialize an indexed set with a dictionary
# @decl6
H_init = {}
H_init[2] = [1, 3, 5]
H_init[3] = [2, 4, 6]
H_init[4] = [3, 5, 7]
model.H = pyo.Set([2, 3, 4], initialize=H_init)
# @decl6

model.pprint(verbose=True)
