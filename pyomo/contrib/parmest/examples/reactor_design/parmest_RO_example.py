#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pandas as pd
import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model, reactor_design_model_RO

if __name__ == "__main__":
    # Note, if this is not called from main, you get multiprocessing warnings (windows issue)
    
    # Data
    data = pd.read_excel('reactor_data.xlsx') 
    
    ### Reactor design model with data from a single experiment
    m = reactor_design_model(data.iloc[-1])
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m)
    
    m.pprint()
    
    print('k1', m.k1.value)
    print('k2', m.k2.value)
    print('k3', m.k3.value)
    print('ca', m.ca.value)
    print('cb', m.cb.value)
    print('cc', m.cc.value)
    print('cd', m.cd.value)
    
    ### Estimate k1, k2, k3 using data from all experiments
    # Vars to estimate
    theta_names = ['k1', 'k2', 'k3']

    # Sum of squared error function
    def SSE(model, data): 
        expr = (float(data['ca']) - model.ca)**2 + \
               (float(data['cb']) - model.cb)**2 + \
               (float(data['cc']) - model.cc)**2 + \
               (float(data['cd']) - model.cd)**2
        return expr
    
    pest = parmest.Estimator(reactor_design_model, data.loc[0:17,:], theta_names, SSE)
    obj, theta, cov = pest.theta_est(calc_cov=True)
    print(obj)
    print(theta)
    print(cov)
    
    parmest.graphics.pairwise_plot((theta, cov, 1000), theta_star=theta, alpha=0.8, 
                               distributions=['MVN'])
    
    
    ### Reactor design model with data from a single experiment using RO with theta and cov
    m = reactor_design_model_RO(data.iloc[-1], theta, cov)
    
    solver = pyo.SolverFactory('romodel.cuts')
    solver.solve(m)
    
    m.pprint()
    
    theta_ro = {'k1': m.k['k1'].value,
                'k2': m.k['k2'].value,
                'k3': m.k['k3'].value}
    print(theta_ro)
    print('ca', m.ca.value)
    print('cb', m.cb.value)
    print('cc', m.cc.value)
    print('cd', m.cd.value)
    print(data.iloc[-1])

    parmest.graphics.pairwise_plot((theta_ro, cov, 1000), theta_star=theta_ro, alpha=0.8, 
                               distributions=['MVN'])

        