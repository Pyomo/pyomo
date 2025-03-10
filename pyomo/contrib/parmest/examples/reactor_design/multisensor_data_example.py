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

from pyomo.common.dependencies import pandas as pd
from os.path import join, abspath, dirname
import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)


class MultisensorReactorDesignExperiment(ReactorDesignExperiment):

    def finalize_model(self):

        m = self.model

        # Experiment inputs values
        m.sv = self.data_i['sv']
        m.caf = self.data_i['caf']

        # Experiment output values
        m.ca = (self.data_i['ca1'] + self.data_i['ca2'] + self.data_i['ca3']) * (1 / 3)
        m.cb = self.data_i['cb']
        m.cc = (self.data_i['cc1'] + self.data_i['cc2']) * (1 / 2)
        m.cd = self.data_i['cd']

        return m

    def label_model(self):

        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update(
            [
                (m.ca, [self.data_i['ca1'], self.data_i['ca2'], self.data_i['ca3']]),
                (m.cb, [self.data_i['cb']]),
                (m.cc, [self.data_i['cc1'], self.data_i['cc2']]),
                (m.cd, [self.data_i['cd']]),
            ]
        )

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.k1, m.k2, m.k3]
        )

        return m


def main():
    # Parameter estimation using multisensor data

    # Read in data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data_multisensor.csv"))
    data = pd.read_csv(file_name)

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(MultisensorReactorDesignExperiment(data, i))

    # Define sum of squared error
    def SSE_multisensor(model):
        expr = 0
        for y, y_hat in model.experiment_outputs.items():
            num_outputs = len(y_hat)
            for i in range(num_outputs):
                expr += ((y - y_hat[i]) ** 2) * (1 / num_outputs)
        return expr

    # View one model
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()
    # print(SSE_multisensor(exp0_model))

    pest = parmest.Estimator(exp_list, obj_function=SSE_multisensor)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)


if __name__ == "__main__":
    main()
