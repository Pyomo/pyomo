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
## Creating a model
##
model = AbstractModel()

##
## Declaring Sets
##
#
# An unordered set of arbitrary objects can be defined by creating a Set()
# object:
#
model.A = Set()
#
# An index set of sets can also be specified by providing sets as options
# to the Set() object:
# 
model.B = Set()
model.C = Set(model.A,model.B)
#
# Set declarations can also use standard set operations to declare
# a set in a constructive fashion:
#
model.D = model.A | model.B
model.E = model.B & model.A
model.F = model.A - model.B
model.G = model.A ^ model.B
#
# Also, set cross-products can be specified as A*B
#
model.H = model.A * model.B
#
# Note that this is different from the following, which specifies that Hsub
# is a subset of this cross-product.
#
model.Hsub = Set(within=model.A * model.B)

##
## Data for Simple Sets
##
#
# A set can be constructed with the _initialize_ option, which is a function 
# that accepts the set indices and model and returns the value of that set 
# element:
#
def I_init(model):
    ans=[]
    for a in model.A:
        for b in model.B:
            ans.append( (a,b) )
    return ans
model.I = model.A*model.B
model.I.initialize = I_init
#
# Note that the set model.I is not created when this set object is
# constructed.  Instead, I_init() is called during the construction of a
# problem instance.
#
# A set can also be explicitly constructed by add set elements:
#
model.J = Set()
model.J.add(1,4,9)
#
# The _initialize_ option can also be used to specify the values in
# a set.  These default values may be overriden by later construction
# steps, or by data in an input file:
#
model.K = Set(initialize=[1,4,9])
model.K_2 = Set(initialize=[(1,4),(9,16)],dimen=2)
#
# Validation of set data is supported in two different ways.  First, a
# superset can be specified with the _within_ option:
#
model.L = Set(within=model.A)
#
# Validation of set data can also be performed with the _validate_ option,
# which is a function that returns True if a data belongs in this set:
#
def M_validate(model, value):
    return value in model.A
model.M = Set(validate=M_validate)
#
# Although the _within_ option is convenient, it can force the creation of
# a temporary set.  For example, consider the declaration
#
model.N = Set(within=model.A*model.B)
#
# In this example, the cross-product of sets A and B is needed to validate
# the members of set C.  Pyomo creates this set implicitly and uses
# it for validation.  By contrast, a simple validation function could be used
# in this example, though with a less intuitive syntax:
#
def O_validate(model, value):
    return value[0] in model.A and value[1] in model.B
model.O = Set(validate=O_validate)

##
## Data for Set Arrays
##
#
# A set array can be constructed with the _initialize_ option, which is a 
# function that accepts the set indices and model and returns the set for that 
# array index:
#
def P_init(model, i, j):
    return range(0,i*j)
model.P = Set(model.B,model.B)
model.P.initialize = P_init
#
# A set array CANNOT be explicitly constructed by adding set elements
# to individual arrays.  For example, the following is invalid:
#
#   model.Q = Set(model.B)
#   model.Q[2].add(4)
#   model.Q[4].add(16)
#
# The reason is that the line
#
#   model.Q = Set(model.B)
#
# declares set Q with an abstract index set B.  However, B is not initialized
# until the 'model.create_instance()' call is executed at the end of this file.  We
# could, however, execute 
#
#   model.Q[2].add(4)
#   model.Q[4].add(16)
#
# after the execution of 'model.create_instance()'.
#
# The _initialize_ option can also be used to specify the values in
# a set array.  These default values are defined in a dictionary, which 
# specifies how each array element is initialized:
#
R_init={}
R_init[2] = [1,3,5]
R_init[3] = [2,4,6]
R_init[4] = [3,5,7]
model.R = Set(model.B,initialize=R_init)
#
# Validation of a set array is supported with the _within_ option.  The
# elements of all sets in the array must be in this set:
#
model.S = Set(model.B, within=model.A)
#
# Validation of set arrays can also be performed with the _validate_ option.
# This is applied to all sets in the array:
#
def T_validate(model, value):
    return value in model.A
model.T = Set(model.B, validate=M_validate)

##
## Set options
##
#
# By default, sets are unordered.  That is, the internal representation
# may place the set elements in any order.  In some cases, we need to know
# the order in which set elements are declared.  In such cases, we can declare
# a set to be ordered with an additional constructor option.  
#
# An ordered set can take a initialization function with an additional option 
# that specifies the index into the ordered set.  In this case, the function is 
# called repeatedly to construct each element in the set:
# 
def U_init(model, z):
    if z==6:
        return Set.End
    if z==1:
        return 1
    else:
        return model.U[z-1]*z
model.U = Set(ordered=True, initialize=U_init)
#
# This example can be generalized to array sets.  Note that in this case
# we can use ordered sets to to index the array, thereby guaranteeing that
# data has been filled.  The following example illustrates the use of the
# RangeSet(a,b) object, which generates an ordered set from 'a' to 'b'
# (inclusive).
#
def V_init(model, z, i):
    if z==6:
        return Set.End
    if i==1:
        return z
    return model.V[i-1][z]+z-1
model.V = Set(RangeSet(1,4), initialize=V_init, ordered=True)

##
## Process an input file and confirm that we get appropriate 
## set instances.
##
instance = model.create_instance("set.dat")
instance.pprint()
