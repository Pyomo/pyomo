"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Set.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()
# @Declare_set
model.A = Set()
# @Declare_set

# @Set_dim
model.B = Set(dimen=2)
# @Set_dim

def DoubleA_init(m):
    return ['red', 'green', 'blue']

# @Set_initialize
model.C = Set(initialize=DoubleA_init)
# @Set_initialize

# @Initialize_python_sets
model.D = Set(initialize=['red', 'green', 'blue'])
# @Initialize_python_sets

# @Sets_without_keywords
model.E = Set(model.A)
# @Sets_without_keywords

# @Combined_arguments
model.F = Set(model.A, dimen=3)
# @Combined_arguments

# @RangeSet_simple_sequence
model.G = RangeSet(1.5, 10, 3.5)
# @RangeSet_simple_sequence

# @Set_operators
model.I = model.A | model.D # union
model.J = model.A & model.D # intersection
model.K = model.A - model.D # difference
model.L = model.A ^ model.D # exclusive-or
# @Set_operators

# @Set_cross_product
model.K = model.B * model.C
# @Set_cross_product

# @Restrict_to_crossproduct
model.K = Set(within=model.B * model.C)
# @Restrict_to_crossproduct

# @Assign_to_crossproduct
model.C = Set(model.A * model.B)
# @Assign_to_crossproduct

# @Contain_crossproduct_subset
model.C = Set(within=model.A * model.B)
# @Contain_crossproduct_subset

# @Predefined_set_example
model.M = Set(within=NegativeIntegers)
# @Predefined_set_example

# @Declare_nodes
model.Nodes = Set()
# @Declare_nodes

# @Declare_arcs_crossproduct
model.arcs = model.Nodes*model.Nodes
# @Declare_arcs_crossproduct

# @Declare_arcs_within
model.Arcs = Set(within=model.Nodes*model.Nodes)
# @Declare_arcs_within

# @Declare_arcs_dimen
model.Arcs = Set(dimen=2)
# @Declare_arcs_dimen


# @Define_constraint_tuples
def kv_init(model):
    return ((k,v) for k in model.K for v in model.V[k])
model.KV=Set(dimen=2, initialize=kv_init)
# @Define_constraint_tuples


# @Declare_constraints_example
from pyomo.environ import *

model = AbstractModel()

model.I=Set()
model.K=Set()
model.V=Set(model.K)

def kv_init(model):
    return ((k,v) for k in model.K for v in model.V[k])
model.KV=Set(dimen=2, initialize=kv_init)

model.a = Param(model.I, model.K)

model.y = Var(model.I)
model.x = Var(model.I, model.KV)

#include a constraint
#x[i,k,v] <= a[i,k]*y[i], for i in model.I, k in model.K, v in model.V[k]

def c1Rule(model,i,k,v):
   return model.x[i,k,v] <= model.a[i,k]*model.y[i]
model.c1 = Constraint(model.I,model.KV,rule=c1Rule)
# @Declare_constraints_example

# @Define_another_constraint
model.MyConstraint = Constraint(model.I,model.KV,rule=c1Rule)
# @Define_another_constraint

