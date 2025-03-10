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
"""
Continuously stirred tank reactor model, based on
pyomo/examples/doc/pyomobook/nonlinear-ch/react_design/ReactorDesign.py
"""

from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment


def reactor_design_model():

    # Create the concrete model
    model = pyo.ConcreteModel()

    # Rate constants
    model.k1 = pyo.Param(
        initialize=5.0 / 6.0, within=pyo.PositiveReals, mutable=True
    )  # min^-1
    model.k2 = pyo.Param(
        initialize=5.0 / 3.0, within=pyo.PositiveReals, mutable=True
    )  # min^-1
    model.k3 = pyo.Param(
        initialize=1.0 / 6000.0, within=pyo.PositiveReals, mutable=True
    )  # m^3/(gmol min)

    # Inlet concentration of A, gmol/m^3
    model.caf = pyo.Param(initialize=10000, within=pyo.PositiveReals, mutable=True)

    # Space velocity (flowrate/volume)
    model.sv = pyo.Param(initialize=1.0, within=pyo.PositiveReals, mutable=True)

    # Outlet concentration of each component
    model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
    model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

    # Objective
    model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

    # Constraints
    model.ca_bal = pyo.Constraint(
        expr=(
            0
            == model.sv * model.caf
            - model.sv * model.ca
            - model.k1 * model.ca
            - 2.0 * model.k3 * model.ca**2.0
        )
    )

    model.cb_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb)
    )

    model.cc_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cc + model.k2 * model.cb)
    )

    model.cd_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cd + model.k3 * model.ca**2.0)
    )

    return model


class ReactorDesignExperiment(Experiment):

    def __init__(self, data, experiment_number):
        self.data = data
        self.experiment_number = experiment_number
        self.data_i = data.loc[experiment_number, :]
        self.model = None

    def create_model(self):
        self.model = m = reactor_design_model()
        return m

    def finalize_model(self):
        m = self.model

        # Experiment inputs values
        m.sv = self.data_i['sv']
        m.caf = self.data_i['caf']

        # Experiment output values
        m.ca = self.data_i['ca']
        m.cb = self.data_i['cb']
        m.cc = self.data_i['cc']
        m.cd = self.data_i['cd']

        return m

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update(
            [
                (m.ca, self.data_i['ca']),
                (m.cb, self.data_i['cb']),
                (m.cc, self.data_i['cc']),
                (m.cd, self.data_i['cd']),
            ]
        )

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.k1, m.k2, m.k3]
        )

        return m

    def get_labeled_model(self):
        m = self.create_model()
        m = self.finalize_model()
        m = self.label_model()

        return m


def main():

    # For a range of sv values, return ca, cb, cc, and cd
    results = []
    sv_values = [1.0 + v * 0.05 for v in range(1, 20)]
    caf = 10000
    for sv in sv_values:

        # make model
        model = reactor_design_model()

        # add caf, sv
        model.caf = caf
        model.sv = sv

        # solve model
        solver = pyo.SolverFactory("ipopt")
        solver.solve(model)

        # save results
        results.append([sv, caf, model.ca(), model.cb(), model.cc(), model.cd()])

    results = pd.DataFrame(results, columns=["sv", "caf", "ca", "cb", "cc", "cd"])
    print(results)


if __name__ == "__main__":
    main()
