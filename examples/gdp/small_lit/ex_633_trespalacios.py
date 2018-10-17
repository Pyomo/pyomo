"""Analytical example from Section 6.3.3 of F. Trespalacions Ph.D. Thesis (2015)

Analytical example for a nonconvex GDP with 2 disjunctions, each with 2 disjuncts.

Ref:
    ANALYTICAL NONCONVEX GDP EXAMPLE.
    FRANCISCO TRESPALACIOS , PH.D. THESIS (EQ 6.6) , 2015.
    CARNEGIE-MELLON UNIVERSITY , PITTSBURGH , PA.

Solution is 4.46 with (Z, x1, x2) = (4.46, 1.467, 0.833),
with the second and first disjuncts active in
the first and second disjunctions, respectively.

Pyomo model implementation by @bernalde and @qtothec.

"""
from __future__ import division

from pyomo.environ import *
from pyomo.gdp import *


def build_simple_nonconvex_gdp():
    """Build the Analytical Problem."""
    m = ConcreteModel(name="Example 6.3.3")

    # Variables x1 and x2
    m.x1 = Var(bounds=(0, 5), doc="variable x1")
    m.x2 = Var(bounds=(0, 3), doc="variable x2")
    m.obj = Objective(expr=5 + 0.2 * m.x1 - m.x2, doc="Minimize objective")

    m.disjunction1 = Disjunction(expr=[
        [m.x2 <= 0.4*exp(m.x1/2.0),
         m.x2 <= 0.5*(m.x1 - 2.5)**2 + 0.3,
         m.x2 <= 6.5/(m.x1/0.3 + 2.0) + 1.0],
        [m.x2 <= 0.3*exp(m.x1/1.8),
         m.x2 <= 0.7*(m.x1/1.2 - 2.1)**2 + 0.3,
         m.x2 <= 6.5/(m.x1/0.8 + 1.1)]
    ])
    m.disjunction2 = Disjunction(expr=[
        [m.x2 <= 0.9*exp(m.x1/2.1),
         m.x2 <= 1.3*(m.x1/1.5 - 1.8)**2 + 0.3,
         m.x2 <= 6.5/(m.x1/0.8 + 1.1)],
        [m.x2 <= 0.4*exp(m.x1/1.5),
         m.x2 <= 1.2*(m.x1 - 2.5)**2 + 0.3,
         m.x2 <= 6.0/(m.x1/0.6 + 1.0) + 0.5]
    ])

    return m


if __name__ == "__main__":
    model = build_simple_nonconvex_gdp()
    model.pprint()
    res = SolverFactory('gdpopt').solve(model, tee=True, strategy='GLOA')

    model.display()
    print(res)
