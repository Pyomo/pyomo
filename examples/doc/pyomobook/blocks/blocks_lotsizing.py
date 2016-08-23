from pyomo.environ import *
from lotsizing_models import inventory_block

model = ConcreteModel()
model.T = RangeSet(5)    # time periods

i_pos0 = 5.0       # initial inventory
i_neg0 = 0.0       # initial backlog
c = 4.6            # setup cost
h_pos = 0.7        # inventory holding cost
h_neg = 1.2        # shortage cost
P = 5.0            # maximum production amount

# demand during period t
d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

### create the model
# create the blocks that correspond to each time period
def inventory_blocks_rule(b,t):
    return inventory_block(c, h_pos, h_neg, P, d[t])
model.ib = Block(model.T, rule=inventory_blocks_rule)

# link the inventory variables of one block to the previous block
def i_pos_linking_rule(m,t):
    if t == m.T.first():
        return m.ib[t].i_pos_prev == i_pos0
    return m.ib[t].i_pos_prev == m.ib[t-1].i_pos
model.i_pos_linking = Constraint(model.T, rule=i_pos_linking_rule)

def i_neg_linking_rule(m,t):
    if t == m.T.first():
        return m.ib[t].i_neg_prev == i_neg0
    return m.ib[t].i_neg_prev == m.ib[t-1].i_neg
model.i_neg_linking = Constraint(model.T, rule=i_neg_linking_rule)

# construct the objective function over all the blocks
def obj_rule(m):
    return sum(m.ib[t].obj_expr for t in m.T)
model.obj = Objective(rule=obj_rule)

### solve the problem
solver = SolverFactory('glpk')
solver.solve(model)

### print the results
for v in model.ib[:].y:
    print(v.cname(fully_qualified=True), value(v))

