import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.core import *
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
from reactor_design import reactor_design_model

results = {}

thetalist = ['k1', 'k2', 'k3'] # parmest makes sure these are unfixed

### Example 1. Data defined by design values
data = pd.read_excel('reactor_data.xlsx', 'design-values') # I think there is a way to read data from excel and make sure that data types are native python (instead of np.float), thus removing the need to wrap all values in float()
def LSE(model, data):  # Least squares error
    expr = (float(data['ca']) - model.ca)**2 + \
           (float(data['cb']) - model.cb)**2 + \
           (float(data['cc']) - model.cc)**2 + \
           (float(data['cd']) - model.cd)**2
    return expr
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE)
objval, thetavals = pest.theta_est()
results['Design values'] = thetavals

###  Example 2. Data defined by design values with noise
data = pd.read_excel('reactor_data.xlsx', 'with-noise') 
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE)
objval, thetavals = pest.theta_est()
results['Design values with noise'] = thetavals
"""
# Use other options in parmest
bootstrap_theta = pest.bootstrap(100)
grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, 
                             axis_limits = {'k1': [0.77, 0.87], 'k2': [1.5, 1.7], 'k3': [0.00015, 0.00018]}, 
                             filename="bootstrap.png")

search_ranges = {}
search_ranges["k1"] = np.arange(0.77, 0.87, 0.02) # 0.83
search_ranges["k2"] = np.arange(1.52, 1.68, 0.02) # 1.67
search_ranges["k3"] = np.arange(0.00015, 0.00018, 0.000005) # 0.000167
#search_ranges["k1"] = np.arange(0.76, 0.88, 0.01) # 0.83
#search_ranges["k2"] = np.arange(1.50, 1.70, 0.01) # 1.67
#search_ranges["k3"] = np.arange(0.000145, 0.00018, 0.0000025) # 0.000167
SSE = pest.likelihood_ratio(search_ranges=search_ranges)
grph.pairwise_likelihood_ratio_plot(SSE, objval, 0.8, data.shape[0], 
                                    axis_limits = {'k1': [0.77, 0.87], 'k2': [1.5, 1.7], 'k3': [0.00015, 0.00018]}, 
                                    filename="likelihood_ratio.png")
"""
###  Example 3. Data includes multiple sensors for ca and cc
data = pd.read_excel('reactor_data.xlsx', 'multisensor') 
def LSE_multisensor(model, data): 
    expr = ((float(data['ca1']) - model.ca)**2)*(1/3) + \
           ((float(data['ca2']) - model.ca)**2)*(1/3) + \
           ((float(data['ca3']) - model.ca)**2)*(1/3) + \
            (float(data['cb'])  - model.cb)**2 + \
           ((float(data['cc1']) - model.cc)**2)*(1/2) + \
           ((float(data['cc2']) - model.cc)**2)*(1/2) + \
            (float(data['cd'])  - model.cd)**2
    return expr
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE_multisensor)
objval, thetavals = pest.theta_est()
results['Multisensor example'] = thetavals

# Use other options in parmest
bootstrap_theta = pest.bootstrap(20)
grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, filename="bootstrap_multisensor.png")

###  Example 4. Data is reported as a timeseries labeled with experiment number (determined by stable regions for sv)
data = pd.read_excel('reactor_data.xlsx', 'timeseries') 
# Use each timestep as a separate experiment
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE)
objval, thetavals = pest.theta_est()
results['Timeseries example: Use each timestep as a separate experiment'] = thetavals
# Group time series into experiments, return the mean value for sv and caf
data_ts = parmest.group_experiments(data, 'experiment', ['sv', 'caf']) # returns a list of dictionaries
def LSE_timeseries(model, data): 
    expr = 0
    for val in data['ca']:
        expr = expr + ((float(val) - model.ca)**2)*(1/len(data['ca']))
    for val in data['cb']:
        expr = expr + ((float(val) - model.cb)**2)*(1/len(data['cb']))
    for val in data['cc']:
        expr = expr + ((float(val) - model.cc)**2)*(1/len(data['cc']))
    for val in data['cd']:
        expr = expr + ((float(val) - model.cd)**2)*(1/len(data['cd']))
    return expr
pest = parmest.Estimator(reactor_design_model, data_ts, thetalist, LSE_timeseries)
objval, thetavals = pest.theta_est()
results['Timeseries example: Group data by stable regions for sv'] = thetavals

# Use other options in parmest, NOT WORKING
#bootstrap_theta = pest.bootstrap(10)
#grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, filename="bootstrap_timeseries.png")

###  Example 5. Data is reported as a timeseries and includes multiple sensors, labeled with experiment number (determined by stable regions for sv)
data = pd.read_excel('reactor_data.xlsx', 'multisensor-timeseries') 
# Use each timestep as a separate experiment
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE_multisensor)
objval, thetavals = pest.theta_est()
results['Multisensor timeseries example: Use each timestep as a separate experiment'] = thetavals
# Group time series into experiments, return the mean value for sv and caf
data_mts = parmest.group_experiments(data, 'experiment', ['sv', 'caf']) # returns a list of dictionaries
def LSE_mutlisensor_timeseries(model, data): 
    expr = 0
    for val in data['ca1']:
        expr = expr + ((float(val) - model.ca)**2)*(1/len(data['ca1']))*(1/3)
    for val in data['ca2']:
        expr = expr + ((float(val) - model.ca)**2)*(1/len(data['ca2']))*(1/3)
    for val in data['ca3']:
        expr = expr + ((float(val) - model.ca)**2)*(1/len(data['ca3']))*(1/3)
    for val in data['cb']:
        expr = expr + ((float(val) - model.cb)**2)*(1/len(data['cb']))
    for val in data['cc1']:
        expr = expr + ((float(val) - model.cc)**2)*(1/len(data['cc1']))*(1/2)
    for val in data['cc2']:
        expr = expr + ((float(val) - model.cc)**2)*(1/len(data['cc2']))*(1/2)
    for val in data['cd']:
        expr = expr + ((float(val) - model.cd)**2)*(1/len(data['cd']))
    return expr
pest = parmest.Estimator(reactor_design_model, data_mts, thetalist, LSE_mutlisensor_timeseries)
objval, thetavals = pest.theta_est()
results['Multisensors timeseries example: Group data by stable regions for sv'] = thetavals

results = pd.DataFrame(results)
for theta in thetalist:
    print(results.loc[theta])
    print()
