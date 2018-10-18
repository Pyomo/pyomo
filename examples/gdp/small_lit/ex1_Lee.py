"""Simple example of nonlinear problem modeled with GDP framework.
Taken from Example 1 of the paper "New Algorithms for Nonlinear Generalized Disjunctive Programming" by Lee and Grossmann

This GDP problem has one disjunction containing three terms, and exactly one of them must be true. The literature can be found here:
http://egon.cheme.cmu.edu/Papers/LeeNewAlgo.pdf

"""

from pyomo.environ import (ConcreteModel, Constraint, NonNegativeReals,
                           Objective, Var, minimize)
from pyomo.gdp import Disjunct, Disjunction


def build_model():
    m = ConcreteModel()
    m.x1 = Var(domain=NonNegativeReals, bounds=(0, 8))
    m.x2 = Var(domain=NonNegativeReals, bounds=(0, 8))
    m.c = Var(domain=NonNegativeReals, bounds=(1, 3))

    m.y1 = Disjunct()
    m.y2 = Disjunct()
    m.y3 = Disjunct()
    m.y1.constr1 = Constraint(expr=m.x1**2 + m.x2**2 - 1 <= 0)
    m.y1.constr2 = Constraint(expr=m.c == 2)
    m.y2.constr1 = Constraint(expr=(m.x1 - 4)**2 + (m.x2 - 1)**2 - 1 <= 0)
    m.y2.constr2 = Constraint(expr=m.c == 1)
    m.y3.constr1 = Constraint(expr=(m.x1 - 2)**2 + (m.x2 - 4)**2 - 1 <= 0)
    m.y3.constr2 = Constraint(expr=m.c == 3)
    m.GPD123 = Disjunction(expr=[m.y1, m.y2, m.y3])

    m.obj = Objective(expr=(m.x1 - 3)**2 + (m.x2 - 2)**2 + m.c, sense=minimize)

    return m

if __name__ == "__main__":
    model = build_model()
    results = SolverFactory('gdpopt').solve(m, tee=True)
    print(results)

