from expression_system import *
from pyomo.environ import *
from pyomo.gdp import *

m = ConcreteModel()

m.y_idx = RangeSet(6)
m.y = Var(m.y_idx, domain=Binary)

n1 = LeafNode(m.y[1])
n2 = LeafNode(m.y[2])
n3 = LeafNode(m.y[3])
n4 = LeafNode(m.y[4])
n5 = LeafNode(m.y[5])
n6 = LeafNode(m.y[6])

N1 = NotNode(n1)
N2 = OrNode([N1,n2])
N3 = OrNode([n4,n5,n6])
N4 = AndNode([N2,n3,N3])

# (!y1 v y2) ^ y3 ^ (y4 v y5 v y6)
m.logical_constr_idx = RangeSet(len(N4.children))
m.logical_constr = Constraint(m.logical_constr_idx)
for (i,n_or) in zip(m.logical_constr_idx, N4.children):
    if isinstance(n_or, OrNode):
        m.logical_constr[i] = sum(n.var() if isinstance(n, LeafNode) else (1-n.child.var()) for n in n_or.children) >= 1
    elif isinstance(n_or, NotNode):
        m.logical_constr[i] = n_or.child.var() == 0
    elif isinstance(n_or, LeafNode):
        m.logical_constr[i] = n_or.var() == 1

m.obj = Objective(expr = 10*m.y[3] + 10*m.y[2] + 100*m.y[1] + 1*m.y[5] + 100*(m.y[4]+m.y[6]))

m_trafo = TransformationFactory('gdp.chull').create_using(m)
res = SolverFactory('baron').solve(m_trafo,tee=True)


# Conjunctive normal form 

# Idea: Do multiple disjunctions using @m.Disjunction(idx)
def genDisjunctsFromCNF(m, N):
    assert(isAndNode(N))
    m.b = Block()
    num_disjunctions = sum(1 for n in N.children)
    max_disjuncts = max([len(n.children) for n in N.children if isinstance(n, MultiNode)])
    m.possible_disjuncts = Disjunct(range(num_disjunctions), range(max_disjuncts))
    #m.possible_disjuncts = Disjunct(range(10),range(10))
    actual_disjuncts = dict()
    actual_disjuncts_set = []
    for (i,n) in enumerate(N.children):
        if isOrNode(n):
            m.b.or_idx = RangeSet(0,len(n.children)-1)
            m.b.or_disjunct = Disjunct(m.b.or_idx)
            for (c,d_i) in zip(n.children,m.b.or_idx):
                if isNotNode(c):
                    assert(isLeafNode(c.child))
                    m.possible_disjuncts[i,d_i].c = Constraint(expr = c.child.child==0)
                elif isLeafNode(c):
                    m.possible_disjuncts[i,d_i].c = Constraint(expr = c.child==1)
                else:
                    raise Exception('Child is neither NotNode nor LeafNode')
            #m.b.or_disjunction = Disjunction(expr=[m.b.or_disjunct[d_i] for d_i in m.b.or_idx])
            actual_disjuncts[i] = (idx for idx in m.b.or_idx)
            actual_disjuncts_set.append((i, (idx for idx in m.b.or_idx)))
        else:
            if isNotNode(n):
                assert(isLeafNode(n.child))
                m.b.and_constr = Constraint(expr = n.child.child==0)
            elif isLeafNode(n):
                m.b.and_constr = Constraint(expr = n.child==1)
            else:
                raise Exception('Child is neither NotNode nor LeafNode')
    # By doing this inline, the key get's obscured. Is that a problem?
    @m.Disjunction(actual_disjuncts_set)
    def disjunctions(m,t0,t1):
        return [m.possible_disjuncts[t0,j] for j in t1]
            
if __name__ == '__oldmain__':
    genDisjunctsFromCNF(m,N4)
