#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Example from Section 3.2 in paper of Pseudo Basic Steps

Ref:
    Pseudo basic steps: bound improvement guarantees from Lagrangian
    decomposition in convex disjunctive programming
    Papageorgiou and Trespalacios, 2017

Solution is 2.99 with (x1,x2) = (-0.1, 3.01)

Pyomo model implementation by @RomeoV
"""

import pyomo.environ as pyo
from pyomo.gdp import Disjunct
from pyomo.gdp.basic_step import apply_basic_step


def build_gdp_model():
    model = pyo.ConcreteModel()

    model.x1 = pyo.Var(bounds=(-1, 6), initialize=0)
    model.x2 = pyo.Var(bounds=(0, 7), initialize=3.5)

    model.objective = pyo.Objective(expr=0.2 * model.x1 + model.x2)
    model.disjunction_set = pyo.RangeSet(1, 3)

    model.disjuncts = Disjunct([1, 2, 3], [1, 2])
    model.disjuncts[1, 1].c = pyo.Constraint(
        expr=model.x1**2 + (1 / 4) * (model.x2 - 5) ** 2 <= 1
    )
    model.disjuncts[2, 1].c = pyo.Constraint(
        expr=model.x1**2 + (1 / 4) * (model.x2 - 2) ** 2 <= 1
    )
    model.disjuncts[3, 1].c = pyo.Constraint(
        expr=model.x1**2 + (1 / 4) * (model.x2 - 3.5) ** 2 <= 1
    )

    model.disjuncts[1, 2].c = pyo.Constraint(
        expr=(model.x1 - 5) ** 2 + (1 / 4) * (model.x2 - 2) ** 2 <= 1
    )
    model.disjuncts[2, 2].c = pyo.Constraint(
        expr=(model.x1 - 5) ** 2 + (1 / 4) * (model.x2 - 5) ** 2 <= 1
    )
    model.disjuncts[3, 2].c = pyo.Constraint(
        expr=(model.x1 - 5) ** 2 + (1 / 4) * (model.x2 - 3.5) ** 2 <= 1
    )

    @model.Disjunction(model.disjunction_set, xor=True)
    def disjunctions(model, i):
        return [model.disjuncts[i, 1], model.disjuncts[i, 2]]

    return model


def solve_base_model():
    m_base = build_gdp_model()
    m_hull = pyo.TransformationFactory('gdp.hull').create_using(m_base)
    # m_bigm = pyo.TransformationFactory('gdp.bigm').create_using(m_base, bigM=100)
    solver = pyo.SolverFactory('gams')
    solver.solve(m_hull, solver='baron')
    # m_hull.pprint()
    m_hull.objective.display()
    m_hull.x1.display()
    m_hull.x2.display()


def solve_basic_step_model():
    m_base = build_gdp_model()
    m_base.BS = apply_basic_step([m_base.disjunctions[1], m_base.disjunctions[2]])

    # crux to pprint component
    # with open('pprint.log','w') as outputfile:
    #    m_base.disjunctions.pprint(outputfile)

    # m_bs_hull = pyo.TransformationFactory('gdp.hull').create_using(m_base)
    m_bigm = pyo.TransformationFactory('gdp.bigm').create_using(m_base, bigM=100)
    m_bigm.pprint()

    solver = pyo.SolverFactory('gams')
    solver.solve(m_bigm, tee=True, solver='baron')

    m_bigm.objective.display()
    m_bigm.x1.display()
    m_bigm.x2.display()


if __name__ == '__main__':
    print('################################')
    print('[1] Sanity check: solving base model')
    solve_base_model()

    print('################################')
    print('[2] Solving basic step model')
    solve_basic_step_model()
