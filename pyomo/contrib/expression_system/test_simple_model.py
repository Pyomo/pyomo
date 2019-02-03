from expression_system import *
from pyomo.environ import *
from pyomo.gdp import *

m = ConcreteModel()

m.y_idx = RangeSet(3)
m.y = Var(m.y_idx, domain=Binary)

n1 = LeafNode(m.y[1])
n2 = LeafNode(m.y[2])
n3 = LeafNode(m.y[3])
N1 = NotNode(n1)
N2 = OrNode([N1,n2])
N3 = AndNode([N2,n3])


# Conjunctive normal form 

# Idea: Do multiple disjunctions using @m.Disjunction(idx)
def genDisjunctsFromCNF(m, N):
    assert(isAndNode(N3))
    m.b = Block()
    num_disjunctions = len(n in N.children if isOrNode(n))
    max_disjuncts = max([len(n.children) for n in N.children])
    for n in N.children:
        if isOrNode(n):
            m.b.or_idx = RangeSet(len(n.children))
            m.b.or_disjunct = Disjunct(m.b.or_idx)
            for (c,d_i) in zip(n.children,m.b.or_idx):
                if isNotNode(c):
                    assert(isLeafNode(c.child))
                    m.b.or_disjunct[d_i].c = Constraint(expr = c.child.child==0)
                elif isLeafNode(c):
                    m.b.or_disjunct[d_i].c = Constraint(expr = c.child==1)
                else:
                    raise Exception('Child is neither NotNode nor LeafNode')
            m.b.or_disjunction = Disjunction(expr=[m.b.or_disjunct[d_i] for d_i in m.b.or_idx])
        else:
            if isNotNode(n):
                assert(isLeafNode(n.child))
                m.b.and_constr = Constraint(expr = n.child.child==0)
            elif isLeafNode(n):
                m.b.and_constr = Constraint(expr = n.child==1)
            else:
                raise Exception('Child is neither NotNode nor LeafNode')
            
