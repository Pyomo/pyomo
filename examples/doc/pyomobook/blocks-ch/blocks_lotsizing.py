from pyomo.environ import *

model = ConcreteModel()
model.T = RangeSet(5)    # time periods

i0 = 5.0           # initial inventory
c = 4.6            # setup cost
h_pos = 0.7        # inventory holding cost
h_neg = 1.2        # shortage cost
P = 5.0            # maximum production amount

# demand during period t
d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

# @blockrule:
# create a block for a single time period
def lotsizing_block_rule(b, t):
    # define the variables
    b.y = Var(domain=Binary)
    b.x = Var(domain=NonNegativeReals)
    b.i = Var()
    b.i0 = Var()
    b.i_pos = Var(domain=NonNegativeReals)
    b.i_neg = Var(domain=NonNegativeReals)

    # define the constraints
    b.inventory = Constraint(expr=b.i == b.i0 + b.x - d[t])
    b.pos_neg = Constraint(expr=b.i == b.i_pos - b.i_neg)
    b.prod_indicator = Constraint(expr=b.x <= P * b.y)
model.lsb = Block(model.T, rule=lotsizing_block_rule)
# @:blockrule

# link the inventory variables between blocks
def i_linking_rule(m, t):
    if t == m.T.first():
        return m.lsb[t].i0 == i0
    return m.lsb[t].i0 == m.lsb[t-1].i
model.i_linking = Constraint(model.T, rule=i_linking_rule)

# construct the objective function over all the blocks
def obj_rule(m):
    return sum(c*m.lsb[t].y + h_pos*m.lsb[t].i_pos + h_neg*m.lsb[t].i_neg for t in m.T)
model.obj = Objective(rule=obj_rule)

### solve the problem
solver = SolverFactory('glpk')
solver.solve(model)

# print the results
for t in model.T:
    print('Period: {0}, Prod. Amount: {1}'.format(t, value(model.lsb[t].x))) 
