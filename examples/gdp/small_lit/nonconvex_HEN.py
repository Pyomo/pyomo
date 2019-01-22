""" Example from 'Systematic Modeling of Discrete-Continuous Optimization
      Models through Generalized Disjunctive Programming
    Ignacio E. Grossmann and Francisco Trespalacios, 2013

    Pyomo model implementation by @RomeoV
"""


from pyomo.environ import (ConcreteModel, Constraint, NonNegativeReals,
                           Objective, Var, RangeSet, minimize, TransformationFactory)
from pyomo.gdp import Disjunction
from pyomo.opt import SolverFactory

def build_model():

    T1_lo, T1_up = 350., 400.
    T2_lo, T2_up = 450., 500.

    U = {'1':1.5, '2':0.5, '3':1}
    FCP = {'hot':10.0, 'cold':7.5}
    T_in = {'hot':500., 'cold':350., 'cooling':300., 'steam':600.}
    T_out = {'hot':340., 'cold':560., 'cooling':320., 'steam':600.}
    Cost = {'cooling':20., 'steam':80.}

    m = ConcreteModel()
    m.T1 = Var(domain=NonNegativeReals, bounds=(T1_lo, T1_up))
    m.T2 = Var(domain=NonNegativeReals, bounds=(T2_lo, T2_up))
    m.A1 = Var(domain=NonNegativeReals, bounds=(0,50), initialize=1)
    m.A2 = Var(domain=NonNegativeReals, bounds=(0,50), initialize=1)
    m.A3 = Var(domain=NonNegativeReals, bounds=(0,50), initialize=1)
    m.exchangers = RangeSet(1,3)
    m.CP = Var(m.exchangers, domain=NonNegativeReals, bounds=(0,600*pow(50,0.6)+2*46500))

    m.objective = Objective(
        expr=(sum(m.CP[i] for i in m.exchangers)
              + FCP['hot']*(m.T1-T_out['hot'])*Cost['cooling']
              + FCP['cold']*(T_out['cold']-m.T2)*Cost['steam'])
    )

    m.constr1 = Constraint(
        expr=FCP['hot']*(T_in['hot']-m.T1)==m.A1*U['1']*((T_in['hot']-m.T2)+(m.T1-T_in['cold']))/2.
    )
    m.constr2 = Constraint(
        expr=FCP['hot']*(m.T1-T_out['hot'])==m.A2*U['2']*((T_out['hot']-T_in['cooling'])+(m.T1-T_out['cooling']))/2.
    )
    m.constr3 = Constraint(
        expr=FCP['cold']*(T_out['cold']-m.T2)==m.A3*U['3']*((T_out['steam']-m.T2)+(T_in['steam']-T_out['cold']))/2.
    )
    m.constr4 = Constraint(
        expr=FCP['hot']*(T_in['hot']-m.T1)==FCP['cold']*(m.T2-T_in['cold'])
    )

    # TODO: Can we combine this with a for-loop?

    m.disjunc1 = Disjunction(expr=[
        [m.CP[1] == 2750*pow(m.A1,0.6)+3000,
         0. <= m.A1, m.A1 <= 10.],
        [m.CP[1] == 1500*pow(m.A1,0.6)+15000,
         10. <= m.A1, m.A1 <= 25.],
        [m.CP[1] == 600*pow(m.A1,0.6)+46500,
         25. <= m.A1, m.A1 <= 50.]
    ])

    m.disjunc2 = Disjunction(expr=[
        [m.CP[2] == 2750*pow(m.A2,0.6)+3000,
         0. <= m.A2, m.A2 <= 10.],
        [m.CP[2] == 1500*pow(m.A2,0.6)+15000,
         10. <= m.A2, m.A2 <= 25.],
        [m.CP[2] == 600*pow(m.A2,0.6)+46500,
         25. <= m.A2, m.A2 <= 50.]
    ])

    m.disjunc3 = Disjunction(expr=[
        [m.CP[3] == 2750*pow(m.A3,0.6)+3000,
         0. <= m.A3, m.A3 <= 10.],
        [m.CP[3] == 1500*pow(m.A3,0.6)+15000,
         10. <= m.A3, m.A3 <= 25.],
        [m.CP[3] == 600*pow(m.A3,0.6)+46500,
         25. <= m.A3, m.A3 <= 50.]
    ])

    TransformationFactory('gdp.bigm').apply_to(m,bigM=600*pow(50,0.6)+2*46500)
    #TransformationFactory('gdp.chull').apply_to(m)
    return m

if __name__ == "__main__":
    model = build_model()
    model.pprint()
    res = SolverFactory('gams').solve(model, tee=True, solver='baron', add_options=['option optcr = 0;'], keepfiles=True)
    #res = SolverFactory('gdpopt').solve(model, tee=True, strategy='LOA')
    #model.display()
    print(res)
