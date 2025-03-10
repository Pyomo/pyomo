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

import sys
import pyomo.environ as pyo
import numpy.random as rnd
from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as po


def perform_estimation_pyomo_only(data_fname, solver_trace=False):
    # read in our data file - careful with formats
    df = pd.read_csv(data_fname)
    npts = len(df)

    # create our parameter estimation formulation
    m = pyo.ConcreteModel()
    m.df = df
    m.PTS = pyo.Set(initialize=range(npts), ordered=True)

    # create a separate Pyomo block for each data point
    def _model_i(b, i):
        po.build_single_point_model_pyomo_only(b)

    m.model_i = pyo.Block(m.PTS, rule=_model_i)

    # we want the parameters to be the same across all the data pts
    m.UA = pyo.Var()

    def _eq_parameter(m, i):
        return m.UA == m.model_i[i].UA

    m.eq_parameter = pyo.Constraint(m.PTS, rule=_eq_parameter)

    # define the least squares objective function
    def _least_squares(m):
        obj = 0
        for i in m.PTS:
            row = m.df.iloc[i]

            # error in inputs measured
            obj += (m.model_i[i].Th_in - float(row['Th_in'])) ** 2
            obj += (m.model_i[i].Tc_in - float(row['Tc_in'])) ** 2

            # error in outputs
            obj += (m.model_i[i].Th_out - float(row['Th_out'])) ** 2
            obj += (m.model_i[i].Tc_out - float(row['Tc_out'])) ** 2
        return obj

    m.obj = pyo.Objective(rule=_least_squares)

    solver = pyo.SolverFactory('ipopt')
    status = solver.solve(m, tee=solver_trace)

    return m


def perform_estimation_external(data_fname, solver_trace=False):
    # read in our data file - careful with formats
    df = pd.read_csv(data_fname)
    npts = len(df)

    # create our parameter estimation formulation
    m = pyo.ConcreteModel()
    m.df = df
    m.PTS = pyo.Set(initialize=range(npts), ordered=True)

    # create a separate Pyomo block for each data point
    def _model_i(b, i):
        po.build_single_point_model_external(b)

    m.model_i = pyo.Block(m.PTS, rule=_model_i)

    # we want the parameters to be the same across all the data pts
    # create a global parameter and provide equality constraints to
    # the parameters in each model instance
    m.UA = pyo.Var()

    def _eq_parameter(m, i):
        return m.UA == m.model_i[i].egb.inputs['UA']

    m.eq_parameter = pyo.Constraint(m.PTS, rule=_eq_parameter)

    # define the least squares objective function
    def _least_squares(m):
        obj = 0
        for i in m.PTS:
            row = m.df.iloc[i]

            # error in inputs measured
            obj += (m.model_i[i].egb.inputs['Th_in'] - float(row['Th_in'])) ** 2
            obj += (m.model_i[i].egb.inputs['Tc_in'] - float(row['Tc_in'])) ** 2

            # error in outputs
            obj += (m.model_i[i].egb.inputs['Th_out'] - float(row['Th_out'])) ** 2
            obj += (m.model_i[i].egb.inputs['Tc_out'] - float(row['Tc_out'])) ** 2
        return obj

    m.obj = pyo.Objective(rule=_least_squares)

    solver = pyo.SolverFactory('cyipopt')
    status, nlp = solver.solve(m, tee=solver_trace, return_nlp=True)

    if solver_trace:
        # use the NLP object to access additional information if so desired
        # for example:
        names = nlp.primals_names()
        values = nlp.evaluate_grad_objective()
        print({names[i]: values[i] for i in range(len(names))})

    return m


if __name__ == '__main__':
    m = perform_estimation_pyomo_only(sys.argv[1])
    print(pyo.value(m.UA))
    m = perform_estimation_external(sys.argv[1])
    print(pyo.value(m.UA))
