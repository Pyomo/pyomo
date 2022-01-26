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
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model 


def main():
    # Parameter estimation using multisensor data
    
    # Vars to estimate
    theta_names = ['k1', 'k2', 'k3']
    
    # Data, includes multiple sensors for ca and cc
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, 'reactor_data_multisensor.csv'))
    data = pd.read_csv(file_name)  
    
    # Sum of squared error function
    def SSE_multisensor(model, data): 
        expr = ((float(data['ca1']) - model.ca)**2)*(1/3) + \
               ((float(data['ca2']) - model.ca)**2)*(1/3) + \
               ((float(data['ca3']) - model.ca)**2)*(1/3) + \
                (float(data['cb'])  - model.cb)**2 + \
               ((float(data['cc1']) - model.cc)**2)*(1/2) + \
               ((float(data['cc2']) - model.cc)**2)*(1/2) + \
                (float(data['cd'])  - model.cd)**2
        return expr
    
    pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE_multisensor)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)
    
if __name__ == "__main__":
    main()
