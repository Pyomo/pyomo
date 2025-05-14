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

#
# Imports
#
import pyomo.environ as pyo

##
## Using a Model
##
#
# Pyomo makes a fundamental distinction between an abstract model and a
# problem instance.  The Pyomo AbstractModel() class is used to manage the
# declaration of model components (e.g. sets and variables), and to
# generate a problem instance.
#
model = pyo.AbstractModel()

##
## Declaring Sets
##
#
# An unordered set of arbitrary objects
#
model.A = pyo.Set()
#
# An unordered set of numeric values
#
model.B = pyo.Set()
#
# A simple cross-product
#
model.C = model.A * model.B
#
# A simple cross-product loaded with a tabular data format
#
model.D = pyo.Set(within=model.A * model.B)
#
# A multiple cross-product
#
model.E = pyo.Set(within=model.A * model.B * model.A)

#
# An indexed set
#
model.F = pyo.Set(model.A)
#
# An indexed set
#
model.G = pyo.Set(model.A, model.B)
#
# A simple set
#
model.H = pyo.Set()
#
# A simple set
#
model.I = pyo.Set()
#
# A two-dimensional set
#
model.J = pyo.Set(dimen=2)
#
# A two-dimensional set
#
model.K = pyo.Set(dimen=2)

##
## Declaring Params
##
#
#
# A simple parameter
#
model.Z = pyo.Param()
model.ZZ = pyo.Param()
#
# A single-dimension parameter
#
model.Y = pyo.Param(model.A)
#
# An example of initializing two single-dimension parameters together
#
model.X = pyo.Param(model.A)
model.W = pyo.Param(model.A)
#
# Initializing a parameter with two indices
#
model.U = pyo.Param(model.I, model.A)
model.T = pyo.Param(model.A, model.I)
#
# Initializing a parameter with missing data
#
model.S = pyo.Param(model.A)
#
# An example of initializing two single-dimension parameters together with
# an index set
#
model.R = pyo.Param(model.H, within=pyo.Reals)
model.Q = pyo.Param(model.H, within=pyo.Reals)
#
# An example of initializing parameters with a two-dimensional index set
#
model.P = pyo.Param(model.J, within=pyo.Reals)
model.PP = pyo.Param(model.J, within=pyo.Reals)
model.O = pyo.Param(model.J, within=pyo.Reals)

model.z = pyo.Set(dimen=2)
model.y = pyo.Set()
model.x = pyo.Set()

model.M = pyo.Param(model.K, within=pyo.Reals)
model.N = pyo.Param(model.y, within=pyo.Reals)

model.MM = pyo.Param(model.z)
model.MMM = pyo.Param(model.z)
model.NNN = pyo.Param(model.x)

##
## Process an input file and confirm that we get appropriate
## set instances.
##
instance = model.create_instance("data.dat")
instance.pprint()
