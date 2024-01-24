#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pandas as pd
from os.path import join, abspath, dirname

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)


class TimeSeriesReactorDesignExperiment(ReactorDesignExperiment):

    def __init__(self, data, experiment_number):    
        self.data = data
        self.experiment_number = experiment_number
        self.data_i = data[experiment_number]
        self.model = None
    
    def finalize_model(self):
        m = self.model
        
        # Experiment inputs values
        m.sv = self.data_i['sv']
        m.caf = self.data_i['caf']
        
        # Experiment output values
        m.ca = self.data_i['ca'][0]
        m.cb = self.data_i['cb'][0]
        m.cc = self.data_i['cc'][0]
        m.cd = self.data_i['cd'][0]
        
        return m


def group_data(data, groupby_column_name, use_mean=None):
    """
    Group data by scenario

    Parameters
    ----------
    data: DataFrame
        Data
    groupby_column_name: strings
        Name of data column which contains scenario numbers
    use_mean: list of column names or None, optional
        Name of data columns which should be reduced to a single value per
        scenario by taking the mean

    Returns
    ----------
    grouped_data: list of dictionaries
        Grouped data
    """
    if use_mean is None:
        use_mean_list = []
    else:
        use_mean_list = use_mean

    grouped_data = []
    for exp_num, group in data.groupby(data[groupby_column_name]):
        d = {}
        for col in group.columns:
            if col in use_mean_list:
                d[col] = group[col].mean()
            else:
                d[col] = list(group[col])
        grouped_data.append(d)

    return grouped_data


def main():
    # Parameter estimation using timeseries data

    # Data, includes multiple sensors for ca and cc
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, 'reactor_data_timeseries.csv'))
    data = pd.read_csv(file_name)

    # Group time series data into experiments, return the mean value for sv and caf
    # Returns a list of dictionaries
    data_ts = group_data(data, 'experiment', ['sv', 'caf'])

    # Create an experiment list
    exp_list= []
    for i in range(len(data_ts)):
        exp_list.append(TimeSeriesReactorDesignExperiment(data_ts, i))

    def SSE_timeseries(model):

        expr = 0
        for y, yhat in model.experiment_outputs.items():
            num_time_points = len(yhat)
            for i in range(num_time_points):
                expr += ((y - yhat[i])**2) * (1 / num_time_points)

        return expr

    # View one model & SSE
    # exp0_model = exp_list[0].get_labeled_model()
    # print(exp0_model.pprint())
    # print(SSE_timeseries(exp0_model))

    pest = parmest.Estimator(exp_list, obj_function=SSE_timeseries)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)


if __name__ == "__main__":
    main()
