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

from pyomo.environ import *
from math import isclose
import math
import copy

# ---------------------------------------------
# @ex1
import pyomo.core.expr as EXPR

M = ConcreteModel()
M.x = Var()

e = sin(M.x) + 2 * M.x

# sin(x) + 2*x
print(EXPR.expression_to_string(e))

# sum(sin(x), prod(2, x))
print(EXPR.expression_to_string(e, verbose=True))
# @ex1

# ---------------------------------------------
# @ex2
import pyomo.core.expr as EXPR

M = ConcreteModel()
M.x = Var()
M.y = Var()

e = sin(M.x) + 2 * M.y

# sin(x1) + 2*x2
print(EXPR.expression_to_string(e, labeler=NumericLabeler('x')))
# @ex2

# ---------------------------------------------
# @ex5
M = ConcreteModel()
M.x = Var()
M.x.value = math.pi / 2.0
val = value(M.x)
assert isclose(val, math.pi / 2.0)
# @ex5
# @ex6
val = M.x()
assert isclose(val, math.pi / 2.0)
# @ex6

# ---------------------------------------------
# @ex7
M = ConcreteModel()
M.x = Var()
val = value(M.x, exception=False)
assert val is None
# @ex7

# ---------------------------------------------
# @ex8
import pyomo.core.expr as EXPR

M = ConcreteModel()
M.x = Var()
M.p = Param(mutable=True)

e = M.p + M.x
s = set([type(M.p)])
assert list(EXPR.identify_components(e, s)) == [M.p]
# @ex8

# ---------------------------------------------
# @ex9
import pyomo.core.expr as EXPR

M = ConcreteModel()
M.x = Var()
M.y = Var()

e = M.x + M.y
M.y.value = 1
M.y.fixed = True

assert set(id(v) for v in EXPR.identify_variables(e)) == set([id(M.x), id(M.y)])
assert set(id(v) for v in EXPR.identify_variables(e, include_fixed=False)) == set(
    [id(M.x)]
)
# @ex9

# ---------------------------------------------
# @visitor1
import pyomo.core.expr as EXPR


class SizeofVisitor(EXPR.SimpleExpressionVisitor):
    def __init__(self):
        self.counter = 0

    def visit(self, node):
        self.counter += 1

    def finalize(self):
        return self.counter

    # @visitor1


# ---------------------------------------------
# @visitor2
def sizeof_expression(expr):
    #
    # Create the visitor object
    #
    visitor = SizeofVisitor()
    #
    # Compute the value using the :func:`xbfs` search method.
    #
    return visitor.xbfs(expr)
    # @visitor2


# ---------------------------------------------
# @visitor3
import pyomo.core.expr as EXPR


class CloneVisitor(EXPR.ExpressionValueVisitor):
    def __init__(self):
        self.memo = {'__block_scope__': {id(None): False}}

    def visit(self, node, values):
        #
        # Clone the interior node
        #
        return node.create_node_with_local_data(values)

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types or not node.is_expression_type():
            return True, copy.deepcopy(node, self.memo)

        return False, None
        # @visitor3


# ---------------------------------------------
# @visitor4
def clone_expression(expr):
    #
    # Create the visitor object
    #
    visitor = CloneVisitor()
    #
    # Clone the expression using the :func:`dfs_postorder_stack`
    # search method.
    #
    return visitor.dfs_postorder_stack(expr)
    # @visitor4


# Test:
m = ConcreteModel()
m.x = Var(range(2))
m.p = Param(range(5), mutable=True)
e = m.x[0] + 5 * m.x[1]
ce = clone_expression(e)
print(e is not ce)
# True
print(str(e))
# x[0] + 5*x[1]
print(str(ce))
# x[0] + 5*x[1]
print(e.arg(0) is not ce.arg(0))
# True
print(e.arg(1) is not ce.arg(1))
# True

# ---------------------------------------------
# @visitor5
import pyomo.core.expr as EXPR


class ScalingVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self, scale):
        super(ScalingVisitor, self).__init__()
        self.scale = scale

    def beforeChild(self, node, child, child_idx):
        #
        # Native numeric types are terminal nodes; this also catches all
        # nodes that do not conform to the ExpressionBase API (i.e.,
        # define is_variable_type)
        #
        if child.__class__ in native_numeric_types:
            return False, child
        #
        # Replace leaf variables with scaled variables
        #
        if child.is_variable_type():
            return False, self.scale[id(child)] * child
        #
        # Everything else can be processed normally
        #
        return True, None
        # @visitor5


# ---------------------------------------------
# @visitor6
def scale_expression(expr, scale):
    #
    # Create the visitor object
    #
    visitor = ScalingVisitor(scale)
    #
    # Scale the expression using the :func:`dfs_postorder_stack`
    # search method.
    #
    return visitor.walk_expression(expr)
    # @visitor6


# ---------------------------------------------
# @visitor7
M = ConcreteModel()
M.x = Var(range(5))
M.p = Param(range(5), mutable=True)

scale = {}
for i in M.x:
    scale[id(M.x[i])] = M.p[i]

e = quicksum(M.x[i] for i in M.x)
f = scale_expression(e, scale)

# p[0]*x[0] + p[1]*x[1] + p[2]*x[2] + p[3]*x[3] + p[4]*x[4]
print(f)
# @visitor7
