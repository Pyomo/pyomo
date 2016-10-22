from pyomo.environ import *

def inventory_block(c, h_pos, h_neg, P, d):
    b = Block()
    b.y = Var(domain=Binary)
    b.x = Var(domain=NonNegativeReals)
    b.i_pos_prev = Var(domain=NonNegativeReals)
    b.i_neg_prev = Var(domain=NonNegativeReals)
    b.i_pos = Var(domain=NonNegativeReals)
    b.i_neg = Var(domain=NonNegativeReals)

    def obj_expr_rule(b):
        return c*b.y + h_pos*b.i_pos + h_neg*b.i_neg
    b.obj_expr = Expression(rule=obj_expr_rule)

    def inventory_rule(b):
        return b.i_pos - b.i_neg == b.i_pos_prev - b.i_neg_prev + b.x - d
    b.inventory = Constraint(rule=inventory_rule)

    def production_indicator_rule(b):
        return b.x <= P*b.y
    b.production_indicator = Constraint(rule=production_indicator_rule)
    return b
