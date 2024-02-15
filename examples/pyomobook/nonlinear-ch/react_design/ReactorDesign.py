#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ
import pyomo.environ as pyo


def create_model(k1, k2, k3, caf):
    # create the concrete model
    model = pyo.ConcreteModel()

    # create the variables
    model.sv = pyo.Var(initialize=1.0, within=pyo.PositiveReals)
    model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
    model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

    # create the objective
    model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

    # create the constraints
    model.ca_bal = pyo.Constraint(
        expr=(
            0
            == model.sv * caf
            - model.sv * model.ca
            - k1 * model.ca
            - 2.0 * k3 * model.ca**2.0
        )
    )

    model.cb_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cb + k1 * model.ca - k2 * model.cb)
    )

    model.cc_bal = pyo.Constraint(expr=(0 == -model.sv * model.cc + k2 * model.cb))

    model.cd_bal = pyo.Constraint(expr=(0 == -model.sv * model.cd + k3 * model.ca**2.0))

    return model


if __name__ == '__main__':
    # solve a single instance of the problem
    k1 = 5.0 / 6.0  # min^-1
    k2 = 5.0 / 3.0  # min^-1
    k3 = 1.0 / 6000.0  # m^3/(gmol min)
    caf = 10000.0  # gmol/m^3

    m = create_model(k1, k2, k3, caf)
    status = pyo.SolverFactory('ipopt').solve(m)
    pyo.assert_optimal_termination(status)
    m.pprint()
