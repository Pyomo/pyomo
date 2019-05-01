""" Example from 'Lagrangean Relaxation of the Hull-Reformulation of Linear \
    Generalized Disjunctive Programs and its use in Disjunctive Branch \
    and Bound' Page 25 f.
    Francisco Trespalacios and Ignacio E. Grossmann, 2015

    Pyomo model implementation by @RomeoV

    Optimal solution (for random seed 1, with T=10): 1208.998
"""

from random import randint, seed
from pyomo.environ import \
    (ConcreteModel, Constraint, Set, RangeSet, Param,
     Objective, Var, NonNegativeReals, Block,
     TransformationFactory, SolverFactory)
from pyomo.gdp import Disjunct
from pyomo.contrib.logical_expression_system.nodes import \
    (LeafNode, NotNode, EquivalenceNode, OrNode)
from pyomo.contrib.logical_expression_system.util import \
    (bring_to_conjunctive_normal_form, CNF_to_linear_constraints)


def build_model():
    m = ConcreteModel()

    seed(1)  # Fix seed to generate same parameters and solution every time
    m.T_max = randint(10, 10)
    m.T = RangeSet(m.T_max)

    # Variables
    m.s = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='stock')
    m.x = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='purchased')
    m.c = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='cost')
    m.f = Var(m.T, domain=NonNegativeReals, bounds=(0, 10000), doc='feed')

    m.max_q_idx = RangeSet(m.T_max)

    # Randomly generated parameters
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

    m.beta_L = Param(m.T, m.max_q_idx,
                     initialize=dict(((t, q), randint(10, 999)/1000)
                                     for t in m.T for q in m.max_q_idx),
                     doc='long-term discount')
    m.F_L_lo = Param(m.T, m.max_q_idx,
                     initialize=dict(((t, q), randint(50, 100))
                                     for t in m.T for q in m.max_q_idx),
                     doc='long-term minimum purchase amount')

    # Contract choices 'standard', 'bulk' and long term contracts '0','1',...
    m.disjunct_choices = Set(
        initialize=['S', 'B', *[str(t) for t in range(m.T_max)]])
    m.disjuncts = Disjunct(m.T, m.disjunct_choices)

    # Create disjuncts for contracts in each timeset
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

    # Create disjunctions
    @m.Disjunction(m.T, xor=True)
    def disjunctions(m, t):
        return [m.disjuncts[t, 'S'], m.disjuncts[t, 'B'], m.disjuncts[t, '0'],
                *[m.disjuncts[t, str(q)] for q in range(1, m.T_max-t+1)]]

    # Connect the disjuncts indicator variables using logical expressions
    m.logical_blocks = Block(range(1, m.T_max+1))

    not_y_1_0 = NotNode(LeafNode(m.disjuncts[1, '0'].indicator_var))
    bring_to_conjunctive_normal_form(not_y_1_0)
    CNF_to_linear_constraints(m.logical_blocks[1], not_y_1_0)

    # Long-term contract implies '0'-disjunct in following timesteps
    for t in range(2, m.T_max+1):
        l1 = LeafNode(m.disjuncts[t, '0'].indicator_var)
        or_node = OrNode([LeafNode(m.disjuncts[t_, str(q)].indicator_var)
                          for t_ in range(1, t) for q in range(t-t_, m.T_max-t_+1)])
        equivalence_node = EquivalenceNode(l1, or_node)
        bring_to_conjunctive_normal_form(equivalence_node)
        CNF_to_linear_constraints(m.logical_blocks[t], equivalence_node)

    # Objective function
    m.objective = Objective(expr=sum(m.alpha[t]*m.s[t]+m.c[t] for t in m.T))

    # Global constraints
    m.demand_satisfaction = Constraint(m.T)
    for t in m.T:
        m.demand_satisfaction[t] = m.f[t] >= m.D[t]

    m.material_balance = Constraint(m.T)
    for t in m.T:
        m.material_balance[t]=m.s[t] == (m.s[t-1] if t>1 else 0) + m.x[t] - m.f[t]

    return m


def pprint_result(model):
    """Use pandas to print solution variables

    Printed variables:
    contract choice, base cost, reduction (relative), reduced_cost, spending, stock, storage cost, minimal purchase amount, purchase amount, feed, demand
    """

    print()
    print('#################')
    print('Solution choices:')
    print('#################')
    choices = []
    for t in model.T:
        # Find activated disjunct/contract in each timestep
        choice = filter(
            lambda y: model.disjuncts[t, y].indicator_var.value == 1.0,
            model.disjunct_choices)
        choices.append(next(iter(choice)))

    try:
        from pandas import DataFrame
        df = DataFrame(
            columns=['choice', 'base_cost', 'reduction', 'reduced_cost', 'spending', 'stock', 'storage_cost', 'min_purchase', 'purchased', 'feed', 'demand'])
        df.choice = choices
        df.stock = [model.s[t].value for t in model.T]
        df.storage_cost = [model.alpha[t] for t in model.T]
        df.purchased = [model.x[t].value for t in model.T]
        df.base_cost = [model.gamma[t] for t in model.T]
        df.spending = [model.c[t].value for t in model.T]
        df.feed = [model.f[t].value for t in model.T]
        df.demand = [model.D[t] for t in model.T]
        df.index = [t for t in model.T]

        # Set properties based on contract type
        t = 1
        while t <= model.T_max:
            if df.loc[t, 'choice'] == 'S':
                df.loc[t, 'reduction'] = 0
                df.loc[t, 'min_purchase'] = 0
                df.loc[t, 'reduced_cost'] = model.gamma[t]
                df.loc[t, 'base_cost'] = model.gamma[t]
                t = t+1
            elif df.loc[t, 'choice'] == 'B':
                df.loc[t, 'reduction'] = model.beta_B[t]
                df.loc[t, 'min_purchase'] = model.F_B_lo[t]
                df.loc[t, 'reduced_cost'] = (1-model.beta_B[t])*model.gamma[t]
                df.loc[t, 'base_cost'] = model.gamma[t]
                t = t+1
            elif int(df.loc[t, 'choice']) == 0:
                t = t+1
            else:
                q = int(df.loc[t, 'choice'])
                t_contract = t
                for t_ in range(t, t+q+1):
                    df.loc[t_, 'reduction'] = model.beta_L[t_contract, q]
                    df.loc[t_, 'min_purchase'] = model.F_L_lo[t_contract, q]
                    df.loc[t_, 'reduced_cost'] = (1-model.beta_L[t_contract, q])*model.gamma[t_contract]
                    df.loc[t_, 'base_cost'] = model.gamma[t_contract]
                t = t + q + 1
        print(df)
        print(f'Solution: {model.objective()}')
    except ImportError:
        print("Failed to load module 'pandas' to display solution")


if __name__ == "__main__":
    m = build_model()

    # m.pprint('model.log')
    m_trafo = TransformationFactory('gdp.chull').create_using(m)
    # m_trafo = TransformationFactory('gdp.bigm').create_using(m, bigM=5000)
    # res = SolverFactory('gams').solve(m_trafo, solver='cbc', tee=True)
    res = SolverFactory('glpk').solve(m_trafo, tee=True)
    # m_trafo.pprint('model_trafo.log')
    pprint_result(m_trafo)
