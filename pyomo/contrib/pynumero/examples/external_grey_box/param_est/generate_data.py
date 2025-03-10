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
import numpy.random as rnd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as pm
from pyomo.common.dependencies import pandas as pd


def generate_data(N, UA_mean, UA_std, seed=42):
    rnd.seed(seed)
    m = pyo.ConcreteModel()
    pm.build_single_point_model_pyomo_only(m)

    # dummy objective since this is a square problem
    m.obj = pyo.Objective(expr=1)

    # create the ipopt solver
    solver = pyo.SolverFactory('ipopt')

    data = {'run': [], 'Th_in': [], 'Tc_in': [], 'Th_out': [], 'Tc_out': []}
    for i in range(N):
        # draw a random value for the parameters
        ua = float(rnd.normal(UA_mean, UA_std))
        # draw a noisy value for the test input conditions
        Th_in = 100 + float(rnd.normal(0, 2))
        Tc_in = 30 + float(rnd.normal(0, 2))
        m.UA.fix(ua)
        m.Th_in.fix(Th_in)
        m.Tc_in.fix(Tc_in)

        status = solver.solve(m, tee=False)
        data['run'].append(i)
        data['Th_in'].append(pyo.value(m.Th_in))
        data['Tc_in'].append(pyo.value(m.Tc_in))
        data['Th_out'].append(pyo.value(m.Th_out))
        data['Tc_out'].append(pyo.value(m.Tc_out))

    return pd.DataFrame(data)


def generate_data_external(N, UA_mean, UA_std, seed=42):
    rnd.seed(seed)
    m = pyo.ConcreteModel()
    pm.build_single_point_model_external(m)

    # Add mutable parameters for the rhs of the equalities
    m.UA_spec = pyo.Param(initialize=200, mutable=True)
    m.Th_in_spec = pyo.Param(initialize=100, mutable=True)
    m.Tc_in_spec = pyo.Param(initialize=30, mutable=True)
    m.UA_spec_con = pyo.Constraint(expr=m.egb.inputs['UA'] == m.UA_spec)
    m.Th_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Th_in'] == m.Th_in_spec)
    m.Tc_in_spec_con = pyo.Constraint(expr=m.egb.inputs['Tc_in'] == m.Tc_in_spec)

    # dummy objective since this is a square problem
    m.obj = pyo.Objective(expr=(m.egb.inputs['UA'] - m.UA_spec) ** 2)

    # create the ipopt solver
    solver = pyo.SolverFactory('cyipopt')

    data = {'run': [], 'Th_in': [], 'Tc_in': [], 'Th_out': [], 'Tc_out': []}
    for i in range(N):
        # draw a random value for the parameters
        UA = float(rnd.normal(UA_mean, UA_std))
        # draw a noisy value for the test input conditions
        Th_in = 100 + float(rnd.normal(0, 2))
        Tc_in = 30 + float(rnd.normal(0, 2))
        m.UA_spec.value = UA
        m.Th_in_spec.value = Th_in
        m.Tc_in_spec.value = Tc_in

        status = solver.solve(m, tee=False)
        data['run'].append(i)
        data['Th_in'].append(pyo.value(m.egb.inputs['Th_in']))
        data['Tc_in'].append(pyo.value(m.egb.inputs['Tc_in']))
        data['Th_out'].append(pyo.value(m.egb.inputs['Th_out']))
        data['Tc_out'].append(pyo.value(m.egb.inputs['Tc_out']))

    return pd.DataFrame(data)


if __name__ == '__main__':
    # df = generate_data(50, 200, 5)
    df = generate_data_external(50, 200, 5)
    df.to_csv('data.csv', index=False)
