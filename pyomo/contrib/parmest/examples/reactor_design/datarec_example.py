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

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    reactor_design_model,
    ReactorDesignExperiment,
)

np.random.seed(1234)


class ReactorDesignExperimentDataRec(ReactorDesignExperiment):

    def __init__(self, data, data_std, experiment_number):

        super().__init__(data, experiment_number)
        self.data_std = data_std

    def create_model(self):

        self.model = m = reactor_design_model()
        m.caf.fixed = False

        return m

    def label_model(self):

        m = self.model

        # experiment outputs
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update(
            [
                (m.ca, self.data_i['ca']),
                (m.cb, self.data_i['cb']),
                (m.cc, self.data_i['cc']),
                (m.cd, self.data_i['cd']),
            ]
        )

        # experiment standard deviations
        m.experiment_outputs_std = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs_std.update(
            [
                (m.ca, self.data_std['ca']),
                (m.cb, self.data_std['cb']),
                (m.cc, self.data_std['cc']),
                (m.cd, self.data_std['cd']),
            ]
        )

        # no unknowns (theta names)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        return m


class ReactorDesignExperimentPostDataRec(ReactorDesignExperiment):

    def __init__(self, data, data_std, experiment_number):

        super().__init__(data, experiment_number)
        self.data_std = data_std

    def label_model(self):

        m = super().label_model()

        # add experiment standard deviations
        m.experiment_outputs_std = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs_std.update(
            [
                (m.ca, self.data_std['ca']),
                (m.cb, self.data_std['cb']),
                (m.cc, self.data_std['cc']),
                (m.cd, self.data_std['cd']),
            ]
        )

        return m


def generate_data():

    ### Generate data based on real sv, caf, ca, cb, cc, and cd
    sv_real = 1.05
    caf_real = 10000
    ca_real = 3458.4
    cb_real = 1060.8
    cc_real = 1683.9
    cd_real = 1898.5

    data = pd.DataFrame()
    ndata = 200
    # Normal distribution, mean = 3400, std = 500
    data["ca"] = 500 * np.random.randn(ndata) + 3400
    # Random distribution between 500 and 1500
    data["cb"] = np.random.rand(ndata) * 1000 + 500
    # Lognormal distribution
    data["cc"] = np.random.lognormal(np.log(1600), 0.25, ndata)
    # Triangular distribution between 1000 and 2000
    data["cd"] = np.random.triangular(1000, 1800, 3000, size=ndata)

    data["sv"] = sv_real
    data["caf"] = caf_real

    return data


def main():

    # Generate data
    data = generate_data()
    data_std = data.std()

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(ReactorDesignExperimentDataRec(data, data_std, i))

    # Define sum of squared error objective function for data rec
    def SSE_with_std(model):
        expr = sum(
            ((y - y_hat) / model.experiment_outputs_std[y]) ** 2
            for y, y_hat in model.experiment_outputs.items()
        )
        return expr

    ### Data reconciliation
    pest = parmest.Estimator(exp_list, obj_function=SSE_with_std)

    obj, theta, data_rec = pest.theta_est(return_values=["ca", "cb", "cc", "cd", "caf"])
    print(obj)
    print(theta)

    parmest.graphics.grouped_boxplot(
        data[["ca", "cb", "cc", "cd"]],
        data_rec[["ca", "cb", "cc", "cd"]],
        group_names=["Data", "Data Rec"],
    )

    ### Parameter estimation using reconciled data
    data_rec["sv"] = data["sv"]

    # make a new list of experiments using reconciled data
    exp_list = []
    for i in range(data_rec.shape[0]):
        exp_list.append(ReactorDesignExperimentPostDataRec(data_rec, data_std, i))

    pest = parmest.Estimator(exp_list, obj_function=SSE_with_std)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)

    theta_real = {"k1": 5.0 / 6.0, "k2": 5.0 / 3.0, "k3": 1.0 / 6000.0}
    print(theta_real)


if __name__ == "__main__":
    main()
