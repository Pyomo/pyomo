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

#
# Imports
#
from pyomo.environ import *

##
## Using a Model
##
#
# Pyomo makes a fundamental distinction between an abstract model and a
# problem instance.  The Pyomo AbstractModel() class is used to manage the
# declaration of model components (e.g. sets and variables), and to
# generate a problem instance.
#
model = AbstractModel()

##
## Declaring Sets
##
#
# An unordered set of arbitrary objects
#
model.A = Set()
#
# An unordered set of numeric values
#
model.B = Set()
#
# A simple cross-product
#
model.C = model.A * model.B
#
# A simple cross-product loaded with a tabular data format
#
model.D = Set(within=model.A * model.B)
#
# A multiple cross-product
#
model.E = Set(within=model.A * model.B * model.A)

#
# An indexed set
#
model.F = Set(model.A)
#
# An indexed set
#
model.G = Set(model.A, model.B)
#
# A simple set
#
model.H = Set()
#
# A simple set
#
model.I = Set()
#
# A two-dimensional set
#
model.J = Set(dimen=2)
#
# A two-dimensional set
#
model.K = Set(dimen=2)

##
## Declaring Params
##
#
#
# A simple parameter
#
model.Z = Param()
model.ZZ = Param()
#
# A single-dimension parameter
#
model.Y = Param(model.A)
#
# An example of initializing two single-dimension parameters together
#
model.X = Param(model.A)
model.W = Param(model.A)
#
# Initializing a parameter with two indices
#
model.U = Param(model.I, model.A)
model.T = Param(model.A, model.I)
#
# Initializing a parameter with missing data
#
model.S = Param(model.A)
#
# An example of initializing two single-dimension parameters together with
# an index set
#
model.R = Param(model.H, within=Reals)
model.Q = Param(model.H, within=Reals)
#
# An example of initializing parameters with a two-dimensional index set
#
model.P = Param(model.J, within=Reals)
model.PP = Param(model.J, within=Reals)
model.O = Param(model.J, within=Reals)

model.z = Set(dimen=2)
model.y = Set()
model.x = Set()

model.M = Param(model.K, within=Reals)
model.N = Param(model.y, within=Reals)

model.MM = Param(model.z)
model.MMM = Param(model.z)
model.NNN = Param(model.x)

##
## Process an input file and confirm that we get appropriate
## set instances.
##
instance = model.create_instance("data.dat")
instance.pprint()
