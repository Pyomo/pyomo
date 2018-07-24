#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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
## Setting up a Model
##
#
# Create the model
#
model = AbstractModel()
#
# Create sets used to define parameters
#
model.A = Set()
model.B = Set()

##
## Declaring Params
##
#
#
# A simple parameter
#
model.Z = Param()
#
# A single-dimension parameter
#
model.Y = Param(model.A)
#
# Initializing a parameter with two indices
#
model.X = Param(model.A,model.B)
   
##
## Parameter Data
##
#
# A parameter can be constructed with the _initialize_ option, which is a 
# function that accepts the parameter indices and model and returns the value
# of that parameter element:
#
def W_init(model, i, j):
    #
    # Create the value of model.W[i,j]
    #
    return i*j
model.W = Param(model.A, model.B, initialize=W_init)
#
# Note that the parameter model.W is not created when this object is
# constructed.  Instead, W_init() is called during the construction of a
# problem instance.
#
# The _initialize_ option can also be used to specify the values in
# a parameter.  These default values may be overriden by later construction
# steps, or by data in an input file:
#
V_init={}
V_init[1]=1
V_init[2]=2
V_init[3]=9
model.V = Param(model.B, initialize=V_init)
#
# Note that parameter V is initialized with a dictionary, which maps 
# tuples from parameter indices to parameter values.  Simple, unindexed
# parameters can be initialized with a scalar value.
#
model.U = Param(initialize=9.9)
#
# Validation of parameter data is supported in two different ways.  First, 
# the domain of feasible parameter values can be specified with the _within_
# option:
#
model.T = Param(within=model.B)
#
# Note that the default domain for parameters is Reals, the set of floating
# point values.
#
# Validation of parameter data can also be performed with the _validate_ 
# option, which is a function that returns True if a parameter value is valid:
#
def S_validate(model, value):
    return value in model.A
model.S = Param(validate=S_validate)

##
## Default Values
##
#
# Pyomo assumes that parameter values are specified in a sparse manner.  For
# example, the instance Param(model.A,model.B) declares a parameter indexed
# over sets A and B.  However, not all of these values are necessarily
# declared in a model.  The default value for all parameters not declared
# is zero. This default can be overriden with the _default_ option.
#
# The following example illustrates how a parameter can be declared where
# every parameter value is nonzero, but the parameter is stored with a sparse
# representation.
#
R_init={}
R_init[2,1]=1
R_init[2,2]=1
R_init[2,3]=1
model.R = Param(model.A, model.B, default=99.0, initialize=R_init)
#
# Note that the parameter default value can also be specified in an input 
# file.  See data.dat for an example.
#
# Note that the explicit specification of a zero default changes Pyomo
# behavior.  For example, consider:
#
#   model.a = Param(model.A, default=0.0)
#   model.b = Param(model.A)
#
# When model.a[x] is accessed and the index has not been explicitly initialized,
# the value zero is returned.  This is true whether or not the parameter has
# been initialized with data.  Thus, the specification of a default value
# makes the parameter seem to be densely initialized.
#
# However, when model.b[x] is accessed and the
# index has not been initialized, an error occurs (and a Python exception is
# thrown).  Since the user did not explicitly declare a default, Pyomo 
# treats the reference to model.b[x] as an error.
#

##
## Process an input file and confirm that we get appropriate 
## parameter instances.
##
instance = model.create_instance("param.dat")
instance.pprint()
