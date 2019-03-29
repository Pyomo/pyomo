import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from reactor_design import reactor_design_model

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'k3']

# Data, includes multiple sensors for ca and cc
data = pd.read_excel('reactor_data_timeseries.xlsx')  

# Group time series data into experiments, return the mean value for sv and caf
# Returns a list of dictionaries
data_ts = parmest.group_data(data, 'experiment', ['sv', 'caf'])

def SSE_timeseries(model, data): 
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

pest = parmest.Estimator(reactor_design_model, data_ts, theta_names, SSE_timeseries)
obj, theta = pest.theta_est()
print(obj)
print(theta)
