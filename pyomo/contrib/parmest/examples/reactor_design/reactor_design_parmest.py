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
    expr = (float(data.ca) - model.ca)**2 + \
           (float(data.cb) - model.cb)**2 + \
           (float(data.cc) - model.cc)**2 + \
           (float(data.cd) - model.cd)**2
    return expr
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE)
objval, thetavals = pest.theta_est()
results['Design values'] = thetavals

###  Example 2. Data defined by design values with noise
data = pd.read_excel('reactor_data.xlsx', 'with-noise') 
pest = parmest.Estimator(reactor_design_model, data, thetalist, LSE)
objval, thetavals = pest.theta_est()
results['Design values with noise'] = thetavals

# Use other options in parmest
bootstrap_theta = pest.bootstrap(20)
grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, filename="bootstrap.png")

###  Example 3. Data includes multiple sensors for ca and cc
data = pd.read_excel('reactor_data.xlsx', 'multisensor') 
def LSE_multisensor(model, data): 
    expr = sum((float(val) - model.ca) ** 2 for val in [data.ca1,data.ca2,data.ca3])/3
    expr = expr + (float(data.cb) - model.cb) ** 2
    expr = expr + sum((float(val) - model.ca) ** 2 for val in [data.cc1,data.cc2])/2
    expr = expr + (float(data.cd) - model.ca) ** 2
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
    expr = sum((float(val) - model.ca) ** 2 for val in data['ca'])/len(data['ca'])
    expr = expr + sum((float(val) - model.cb)**2 for val in data['cb'])/len(data['cb'])
    expr = expr + sum((float(val) - model.cc)**2 for val in data['cc'])/len(data['cc'])
    expr = expr + sum((float(val) - model.cd)**2 for val in data['cd'])/len(data['cd'])
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
def LSE_mutlisensor_timeseries(model, data): # there is probably a better way to write this
    data_ca = data['ca1'] + data['ca2'] + data['ca3'] # append lists
    expr = sum((val - model.ca) ** 2 for val in data_ca)/len(data_ca)
    expr = expr + sum((val - model.cb)**2 for val in data['cb'])/len(data['cb'])
    data_cc = data['cc1'] + data['cc2'] # append lists
    expr = expr + sum((val - model.cc)**2 for val in data_cc)/len(data_cc)
    expr = expr + sum((val - model.cd)**2 for val in data['cd'])/len(data['cd'])
    return expr
pest = parmest.Estimator(reactor_design_model, data_mts, thetalist, LSE_mutlisensor_timeseries)
objval, thetavals = pest.theta_est()
results['Multisensors timeseries example: Group data by stable regions for sv'] = thetavals

results = pd.DataFrame(results)
for theta in thetalist:
    print(results.loc[theta])
    print()