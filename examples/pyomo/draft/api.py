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

##
## General description of the API for Pyomo objects
##
## Note: this tutorial was not designed to run as a script
##

#
# Imports
#
from pyomo.core import *


## -------------------------------------------------------------------------
##
## Model objects
##
## -------------------------------------------------------------------------
#
#  The Model object is a fundamental to the use of Pyomo.  Elements of an
#  application are designed with sets, parameters, and variables; these
#  objects are contained in a model, which supports the generation of
#  problem instances that can be solved.
#
model = AbstractModel()
#
#  Once a model has been setup, a problem instance can be created with the
#  create() method:
#
instance = model.create("foo.dat")


## -------------------------------------------------------------------------
##
## Set objects
##
## -------------------------------------------------------------------------
#
#  The Set object is a container for an unordered set of numeric or
#  symbolic values.
#
####
#### Set Constructor Syntax
####
#
#  Simple constructor of a single set
#
model.A = Set()
#
#  Creating an array of sets, which are indexed with another set
#
model.C = Set(model.A, model.A)
#
#  Option 'initialize' indicates how values in the set will be constructed.
#  This option behaves differently depending on the type of data provided:
#  (1) a built-in set or list object can be used to initialize a single set,
#  (2) a dictionary can be used to initialize an array of sets, and
#  (3) a function can be used to initial a set (perhaps using model
#      information.
#
model.A = Set(initialize=[1, 4, 9])
model.B = Set(model.A, initialize={1: [1, 4, 9], 2: [2, 5, 10]})


def f(model):
    return range(0, 10)


model.A = Set(initialize=f)


#
#  Option 'ordered' specifies whether the set elements are ordered
#  This option allows for more sophisticated construction rules
#
def f(model, i):
    if i == 10:
        return Set.End
    if i == 0:
        return 1
    else:
        return model.A[i - 1] * (i + 1)


model.A = Set(ordered=True, initialize=f)
#
#  Option 'within' specifies a set that is used to validate set elements
#
model.B = Set(within=model.A)


#
#  Option 'validate' specifies a function that is used to validate set elements
#
def f(model, value):
    return value in model.A


model.B = Set(validate=f)
#
#  Option 'dimen' specifies the arity of the data in the set
#
model.A = Set(dimen=3)
#
####
#### Set Methods
####
#
# len() - returns the number of elements of a set
#
len(model.A)
len(model.A[i])
#
# data() - returns the underlying set object used by Set()
#
instance.A.data()
instance.A[i].data()
#
# dim() - returns the number of dimensions of the set index
#
instance.A.dim()
#
# clear() - emptys all sets
#
instance.A.clear()
#
# add() - adds data to a set
#
instance.A.add(1, 3, 5)
instance.A[i].add(1, 3, 5)
#
# remove() - removes data from a set, throwing an exception if the data does
#                not exist
#
instance.A.remove(1)
instance.A[i].remove(1)
#
# discard() - removes data from a set without throwing an exception if the
#                data does not exist
#
instance.A.discard(2)
instance.A[i].discard(2)
#
# Set iteration
#
for val in instance.A:
    print(val)
#
# Set comparisons
#
instance.A < instance.B  # True if A is strict subset of B
instance.A <= instance.B  # True if A is a subset of B
instance.A == instance.B  # True if A equals B
instance.A >= instance.B  # True if A is a superset of B
instance.A > instance.B  # True if A is a strict superset of B
#
# Set membership
#
val in instance.A  # True if 'val' is in A
#
# Set operations
#
instance.A | instance.B  # Set union
instance.A & instance.B  # Set intersection
instance.A ^ instance.B  # Set symmetric difference
instance.A - instance.B  # Set difference
#
# Set cross product - define a new set that is the cross-product of
#       two or more sets
#
instance.A * instance.B
#
# Ordered set operations
#
instance.A[j]  # returns the j'th member of ordered set A
instance.A[i][j]  # returns the j'th member of ordered set A[i]
#
# keys() - returns the indices of the set array
#
instance.A.keys()
#
####
#### Set attributes
####
#
# dimen - Specifies the arity of the data in the set
#
model.A.dimen = 3
#
# virtual - is True if the set contains no concrete data
#
instance.A.virtual


## -------------------------------------------------------------------------
##
## Param objects
##
## -------------------------------------------------------------------------
#
#  The Param object defines a scalar or array of numeric parameters.
#
####
#### Param Constructor Syntax
####
#
#  Scalar parameter
#
model.Z = Param()
#
#  Array of parameters
#
model.Z = Param(model.A, model.B)
#
#  Option 'initialize' specifies values used to construct the parameter
#
model.Z = Param(initialize=9)
model.Z = Param(model.A, initialize={1: 1, 2: 4, 3: 9})
model.Z = Param(model.A, initialize=2)


#
#  Option 'initialize' can also specify a function used to construct the
#       parameter
#
def f(model, i):
    return 3 * i


model.Z = Param(model.A, initialize=f)
#
#  Option 'default' specifies values used for a parameter if no value
#   has been set.  Note that for scalar parameters this has the same
#   role as the 'initialize' option.  For parameter arrays this
#   'fills in' parameter values that have not been initialized.
#
model.Z = Param(default=9.0)
model.Z = Param(model.A, default=9.0)
#
#  Option 'within' specifies a set that is used to validate parameters
#
model.Z = Param(within=model.A)


#
#  Option 'validate' specifies a function that is used to validate parameters
#
def f(model, value):
    return value in model.A


model.Z = Param(validate=f)
#
####
#### Param Methods
####
#
# len() - returns the number of parameters
#
len(instance.Z)
#
# dim() - returns the number of dimensions of the parameter index
#
instance.Z.dim()
#
# keys() - returns the indices of the parameter array
#
instance.Z.keys()
#
# Coercing parameter values explicitly
#
tmp = float(instance.Z)
tmp = float(instance.Z[i])
#
# Getting parameter values (which may be integer or floats)
#
tmp = value(instance.Z)
tmp = value(instance.Z[i])
tmp = instance.Z.value
tmp = instance.Z[i].value
#
# Setting parameter values
#
instance.Z = tmp
instance.Z[i] = tmp
#
# NOTE: if Z is an array parameter, then
#
#  instance.Z = tmp
#
# will initialize all of the parameter values to the value of 'tmp'.
#


## -------------------------------------------------------------------------
##
## Var objects
##
## -------------------------------------------------------------------------
#
#  The Var object defines a scalar or array of numeric variables.  These
#  variables define the search space for optimization.  Variables can
#  have initial values, and the value of variable can be retrieved and set.
#
####
#### Var Constructor Syntax
####
#
#  Scalar variable
#
model.x = Var()
#
#  Array of variables
#
model.x = Var(model.A, model.B)
#
#  Option 'initialize' specifies the initial values of variables
#
model.x = Var(initialize=9)
model.x = Var(model.A, initialize={1: 1, 2: 4, 3: 9})
model.x = Var(model.A, initialize=2)


#
#  Option 'initialize' can specify a function used to construct the initial
#   variable values
#
def f(model, i):
    return 3 * i


model.x = Var(model.A, initialize=f)
#
#  Option 'within' specifies a set that is used to constrain variables
#
model.x = Var(within=model.A)
#
#  Option 'bounds' specifies upper and lower bounds for variables.
#  Simple bounds can be specified, or a function that defines bounds for
#  different variables.
#
model.x = Var(bounds=(0.0, 1.0))


def f(model, i):
    return (model.x_low[i], model._x_high[i])


model.x = Var(bounds=f)
#
####
#### Var Methods
####
#
# len() - returns the number of variables
#
len(instance.x)
#
# dim() - returns the number of dimensions of the variable index
#
instance.x.dim()
#
# keys() - returns the indices of the variable array
#
instance.x.keys()
#
# Coercing variable values explicitly
#
tmp = float(instance.x)
tmp = float(instance.x[i])
#
# Getting variable values (which may be integer or floats)
#
tmp = value(instance.x)
tmp = value(instance.x[i])
tmp = instance.x.value
tmp = instance.x[i].value
#
# Setting variable values
#
instance.x = tmp
instance.x[i] = tmp
#
# NOTE: if x is an array variable, then
#
#  instance.x = tmp
#
# will initialize all of the variable values to the value of 'tmp'.
#
####
#### Var attributes
####
#
# Value - the value of a variable
# Note:  value(x) == x.value
#
instance.x.value = 1.0
tmp = instance.x.value
#
# Bounds
#
instance.x.setlb(0.0)  # Set a variable lower bound
instance.x.setub(1.0)  # Set a variable upper bound
#
# Fixed - variables that are fixed (and thus not optimized)
#
instance.x.fixed = True  # Fixes this variable value


## -------------------------------------------------------------------------
##
## Objective objects
##
## -------------------------------------------------------------------------
#
#  The Objective object defines an expression that is maximized or minimized
#  during optimization.  The default sense is minimization.  When more than
#  on objective is specified, or when an array of objectives is specified,
#  multi-objective optimizers are needed to simultaneously optimize these
#  objectives.
#
####
#### Objective Constructor Syntax
####
#
#  Scalar objective
#
model.obj = Objective()
#
#  Array of objectives
#
model.obj = Objective(model.A, model.B)
#
#  Option 'rule' can specify a function used to construct the objective
#       expression
#
model.Z = Param(model.A)
model.x = Var(model.A)


def f(model, i):
    return model.Z[i] * model.A[i]


model.obj = Objective(model.A, rule=f)
#
#  Option 'sense' specifies whether the objective is maximized or minimized
#  Note: this option applies to all objectives in an array objective
#
model.obj = Objective(sense=maximize)
#
####
#### Objective Methods
####
#
# len() - returns the number of objectives that are defined
#
len(instance.obj)
#
# dim() - returns the number of dimensions of the objective index
#
instance.obj.dim()
#
# keys() - returns the indices of the objective array
#
instance.obj.keys()
#
####
#### Objective attributes
####
#
# sense - the optimization sense
#
instance.x.sense = maximize
#
# value - returns the value of the objective
# NOTE: this attribute cannot be set
#
tmp = instance.obj.value
tmp = value(instance.obj)
#
# NOTE: if the objective is an array, then this returns a dictionary of
#       objective values, indexed by the array indices.  Thus, the following
#       example computes _all_ objectives, and then returns the objective value
#       for index '2'
#
tmp = instance.obj.value[2]
tmp = value(instance.obj)[2]


## -------------------------------------------------------------------------
##
## Constraint objects
##
## -------------------------------------------------------------------------
#
#  The Constraint object defines an expression that is either bounded or
#  set equal to another expression.  Constraints can be bounded both
#  above and below.
#
####
#### Constraint Constructor Syntax
####
#
#  Scalar constraint
#
model.con = Constraint()
#
#  Array of constraint
#
model.con = Constraint(model.A, model.B)
#
#  Option 'rule' can specify a function used to construct the constraint
#       expression
#
model.Z = Param(model.A)
model.x = Var(model.A)


def f(model, i):
    expr = model.Z[i] * model.A[i]
    return (0, expr, 1)


model.con = Constraint(model.A, rule=f)


#
# Note: the constructor rule must include the specification of bounds
# information for the constraint.  This can be done in one of two ways.  First,
# the rule can return a tuple that includes the values of the lower and
# upper bounds.  The value 'None' indicates that no bound is specified.
# An equality constraint can be specified as follows:
#
def f(model, i):
    expr = model.Z[i] * model.A[i]
    return (expr, 0)


#
# Second, the constructor rule can augment the expression to include
# bound information.  For example, the previous rule can be rewritten as
#
def f(model, i):
    expr = model.Z[i] * model.A[i]
    expr = expr >= 0
    expr = expr <= 1
    return expr


#
# The following illustrate the type of bounds information that can be
# specified:
#    expr = expr >= val     Lower bound
#    expr = expr > val      Strict lower bound
#    expr = expr < val      Strict upper bound
#    expr = expr <= val     Upper bound
#    expr = expr == val     Equality constraint
#


#
# If the constructor rule returns Constraint.Skip, then the constraint index
# is ignored.  Alternatively, a constructor rule can return a dictionary
# whose keys define the valid constraint indices.  Thus, the following two
# constraints are equivalent:
#
def f1(model, i):
    if i % 2 == 0:
        return Constraint.Skip
    expr = model.Z[i] * model.A[i]
    return (0, expr, 1)


model.con1 = Constraint(model.A, rule=f1)


def f2(model):
    res = {}
    for i in model.A:
        if i % 2 != 0:
            expr = model.Z[i] * model.A[i]
            res[i] = (0, expr, 1)
    return res


model.con2 = Constraint(model.A, rule=f2)

####
#### Constraint Methods
####
#
# len() - returns the number of constraints that are defined
#
len(instance.con)
#
# dim() - returns the number of dimensions of the constraint index
#
instance.con.dim()
#
# keys() - returns the indices of the constraint array
#
instance.con.keys()
#
####
#### Constraint attributes
####
#
# value - returns the value of the constraint body
# NOTE: this attribute cannot be set
#
tmp = instance.con.value
tmp = value(instance.con)
#
# NOTE: if the constraint is an array, then this returns a dictionary of
#       constraint values, indexed by the array indices.  Thus, the following
#       example computes _all_ constraint, and then returns the constraint value
#       for index '2'
#
tmp = instance.con.value[2]
tmp = value(instance.con)[2]
