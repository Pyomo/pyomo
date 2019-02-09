from random import randint
from pyomo.environ import \
    (ConcreteModel, Constraint, Set, RangeSet, Param,
     Objective, Var, NonNegativeReals, Block,
     TransformationFactory, SolverFactory)
from pyomo.gdp import Disjunct
from pyomo.contrib.logical_expression_system.nodes import \
    (LeafNode, NotNode, EquivalenceNode, OrNode)
from pyomo.contrib.logical_expression_system.util import \
    (bring_to_conjunctive_normal_form, CNF_to_linear_constraints)

m = ConcreteModel()
m.T_max = randint(9, 11)
m.T = RangeSet(m.T_max)
m.s = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='stock')
m.x = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='purchased')
m.c = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='cost')
m.f = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='feed')

m.max_q_idx = RangeSet(1, m.T_max)

m.D = Param(m.T, doc='demand',
            initialize=dict((t, randint(50, 100)) for t in m.T))
m.alpha = Param(m.T, doc='storage cost',
                initialize=dict((t, randint(5, 20)) for t in m.T))
m.gamma = Param(m.T, doc='base buying cost',
                initialize=dict((t, randint(10, 30)) for t in m.T))
m.beta_B = Param(m.T, doc='bulk discount',
                 initialize=dict((t, randint(50, 500)/1000) for t in m.T))

m.F_B_lo = Param(m.T, doc='bulk minimum purchase amount',
                 initialize=dict((t, randint(50, 100)) for t in m.T))

# Have to review beta_l
# Currently he never picks any of it
m.beta_L = Param(m.T, m.max_q_idx,
                 initialize=dict(((t, q), randint(10, 999)/1000)
                                 for t in m.T for q in m.max_q_idx),
                 doc='long-term discount')
m.F_L_lo = Param(m.T, m.max_q_idx,
                 initialize=dict(((t, q), randint(50, 100))
                                 for t in m.T for q in m.max_q_idx),
                 doc='long-term minimum purchase amount')

m.disjunct_choices = Set(
    initialize=['S', 'B', *[str(t) for t in range(m.T_max)]])
m.disjuncts = Disjunct(m.T, m.disjunct_choices)

for t in m.T:
    m.disjuncts[t, 'S'].cost = Constraint(expr=m.c[t] == m.gamma[t]*m.x[t])

    m.disjuncts[t, 'B'].cost = Constraint(
        expr=m.c[t] == (1-m.beta_B[t])*m.gamma[t]*m.x[t])
    m.disjuncts[t, 'B'].amount = Constraint(
        expr=m.x[t] >= m.F_B_lo[t])

    m.disjuncts[t, '0'].c = Constraint(expr=0 <= m.c[t])

    for q in range(1, m.T_max-t+1):
        m.disjuncts[t, str(q)].t_idx = RangeSet(t, t+q)
        m.disjuncts[t, str(q)].cost = Constraint(m.disjuncts[t, str(q)].t_idx)
        m.disjuncts[t, str(q)].amount = Constraint(m.disjuncts[t, str(q)].t_idx)
        for t_ in m.disjuncts[t, str(q)].t_idx:
            m.disjuncts[t, str(q)].cost[t_] =\
                m.c[t_] == (1-m.beta_L[t, q])*m.gamma[t]*m.x[t_]
            m.disjuncts[t, str(q)].amount[t_] =\
                m.x[t_] >= m.F_L_lo[t, q]


@m.Disjunction(m.T, xor=True)
def disjunctions(m, t):
    return [m.disjuncts[t, 'S'], m.disjuncts[t, 'B'], m.disjuncts[t, '0'],
            *[m.disjuncts[t, str(q)] for q in range(1, m.T_max-t+1)]]


m.logical_blocks = Block(range(1, m.T_max+1))

not_y_1_0 = NotNode(LeafNode(m.disjuncts[1, '0'].indicator_var))
bring_to_conjunctive_normal_form(not_y_1_0)
CNF_to_linear_constraints(m.logical_blocks[1], not_y_1_0)

for t in range(2, m.T_max+1):
    l1 = LeafNode(m.disjuncts[t, '0'].indicator_var)
    or_node = OrNode([LeafNode(m.disjuncts[t_, str(q)].indicator_var)
                      for t_ in range(1, t) for q in range(t-t_, m.T_max-t_+1)])
    equivalence_node = EquivalenceNode(l1, or_node)
    bring_to_conjunctive_normal_form(equivalence_node)
    CNF_to_linear_constraints(m.logical_blocks[t], equivalence_node)


m.objective = Objective(expr=sum(m.alpha[t]*m.s[t]+m.c[t] for t in m.T))

m.demand_satisfaction = Constraint(m.T)
for t in m.T:
    m.demand_satisfaction[t] = m.f[t] >= m.D[t]

m.material_balance = Constraint(m.T)
for t in m.T:
    m.material_balance[t]=m.s[t] == (m.s[t-1] if t>1 else 0) + m.x[t] - m.f[t]


m.pprint('model.log')
m_trafo = TransformationFactory('gdp.chull').create_using(m)
res = SolverFactory('gams').solve(m_trafo, solver='baron', tee=True)
m_trafo.pprint()

print()
print('#################')
print('Solution choices:')
print('#################')
choices = []
for t in m.T:
    choice = filter(
        lambda y: m_trafo.disjuncts[t, y].indicator_var.value == 1.0,
        m.disjunct_choices)
    choices.append(next(iter(choice)))

from pandas import DataFrame
df = DataFrame(
    columns=['choice', m.s.doc, m.x.doc, 'price', m.c.doc, m.f.doc, m.D.doc])
df.choice = choices
df.stock = [m_trafo.s[t].value for t in m.T]
df.purchased = [m_trafo.x[t].value for t in m.T]
df.cost = [m_trafo.c[t].value for t in m.T]
df.feed = [m_trafo.f[t].value for t in m.T]
df.demand = [m_trafo.D[t] for t in m.T]
df.index = [t for t in m.T]

t = 1
while t <= m.T_max:
    if df.loc[t, 'choice'] == 'S':
        df.loc[t, 'price'] = m.gamma[t]
        t = t+1
    elif df.loc[t, 'choice'] == 'B':
        df.loc[t, 'price'] = (1-m.beta_B[t])*m.gamma[t]
        t = t+1
    elif int(df.loc[t, 'choice']) == 0:
        t = t+1
    else:
        q = int(df.loc[t, 'choice'])
        t_contract = t
        for t_ in range(t, t+q+1):
            df.loc[t_, 'price'] = (1-m.beta_L[t_contract, q])*m.gamma[t_contract]
        t = t_
print(df)
