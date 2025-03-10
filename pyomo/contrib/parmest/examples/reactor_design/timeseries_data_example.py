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

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)


class TimeSeriesReactorDesignExperiment(ReactorDesignExperiment):

    def __init__(self, data, experiment_number):
        self.data = data
        self.experiment_number = experiment_number
        data_i = data.loc[data['experiment'] == experiment_number, :]
        self.data_i = data_i.reset_index()
        self.model = None

    def finalize_model(self):
        m = self.model

        # Experiment inputs values
        m.sv = self.data_i['sv'].mean()
        m.caf = self.data_i['caf'].mean()

        # Experiment output values
        m.ca = self.data_i['ca'][0]
        m.cb = self.data_i['cb'][0]
        m.cc = self.data_i['cc'][0]
        m.cd = self.data_i['cd'][0]

        return m


def main():
    # Parameter estimation using timeseries data, grouped by experiment number

    # Data, includes multiple sensors for ca and cc
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, 'reactor_data_timeseries.csv'))
    data = pd.read_csv(file_name)

    # Create an experiment list
    exp_list = []
    for i in data['experiment'].unique():
        exp_list.append(TimeSeriesReactorDesignExperiment(data, i))

    def SSE_timeseries(model):

        expr = 0
        for y, y_hat in model.experiment_outputs.items():
            num_time_points = len(y_hat)
            for i in range(num_time_points):
                expr += ((y - y_hat[i]) ** 2) * (1 / num_time_points)

        return expr

    # View one model & SSE
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()
    # print(SSE_timeseries(exp0_model))

    pest = parmest.Estimator(exp_list, obj_function=SSE_timeseries)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)


if __name__ == "__main__":
    main()
