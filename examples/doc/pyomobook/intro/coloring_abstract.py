# Python import statement
from pyomo.environ import *

# Create a Pyomo model object
model = AbstractModel()

# Define model sets and parameters
model.vertices = Set()
model.edges = Set(within=model.vertices*model.vertices)
model.ncolors = Param(within=PositiveIntegers)
model.colors = RangeSet(1,model.ncolors)

# Define model variables
model.x = Var(model.vertices, model.colors, within=Binary)
model.y = Var()

# Each node is colored with one color
def node_coloring_rule(model, v):
    return sum(model.x[v,c] for c in model.colors) == 1
model.node_coloring = Constraint(model.vertices,
                                    rule=node_coloring_rule)

# Nodes that share an edge cannot be colored the same
def edge_coloring_rule(model, v, w, c):
    return model.x[v,c] + model.x[w,c] <= 1
model.edge_coloring = Constraint(model.edges, model.colors,
                                    rule=edge_coloring_rule)

# Provide a lower bound on the minimum number of colors
# that are needed
def min_coloring_rule(model, v, c):
    return model.y >= c * model.x[v,c]
model.min_coloring = Constraint(model.vertices, 
                                    model.colors,
                                    rule=min_coloring_rule)

# Minimize the number of colors that are needed
model.obj = Objective(expr=model.y)
