import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pylab as plt
import pyomo.contrib.parmest.parmest as parmest
from reactor_design_datarec import reactor_design_model
import scipy.stats as stats

plt.close('all')

# Generate data based on real theta values, sv, and caf
theta_real = {'k1': 5.0/6.0,
              'k2': 5.0/3.0,
              'k3': 1.0/6000.0}
sv_real = 1.05
caf_real = 10000
ca_real = 3458.4
cb_real = 1060.8
cc_real = 1683.9
cd_real = 1898.5

data = pd.DataFrame() 
ndata = 200
# Normal distribution, mean = 3400, std = 500
data['ca'] = 500 * np.random.randn(ndata) + 3400
# Random distribution between 500 and 1500
data['cb'] = np.random.rand(ndata)*1000+500
# Lognormal distribution
data['cc'] = np.random.lognormal(np.log(1600),0.25,ndata)
# Triangular distribution between 1000 and 2000
data['cd'] = np.random.triangular(1000,1800,3000,size=ndata)

data['sv'] = 0.2 * np.random.randn(ndata) + 1 #sv_real
data['caf'] = 500 * np.random.randn(ndata) + 10000 #caf_real

plt.figure()
data[['ca', 'cb', 'cc', 'cd']].boxplot()
plt.ylim([0,5000])

### Data reconciliation

# Vars to estimate
theta_names = ['caf']

# Sum of squared error function
def SSE(model, data): 
    expr = (float(data['ca']) - model.ca)**2 + \
           (float(data['cb']) - model.cb)**2 + \
           (float(data['cc']) - model.cc)**2 + \
           (float(data['cd']) - model.cd)**2
    return expr

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta, model_vals = pest.theta_est(return_model_values=['ca', 'cb', 'cc', 'cd'])
print(obj)
print(theta)

data_rec = pd.DataFrame(model_vals)

plt.figure()
data_rec.boxplot()
plt.ylim([0,5000])

data_diff = data[['ca', 'cb', 'cc', 'cd']] - data_rec
plt.figure()
data_diff.boxplot()


### Parameter estimation

theta_names = ['k1', 'k2', 'k3']
data_rec['sv'] = data['sv']
data_rec['caf'] = theta['caf']

pest = parmest.Estimator(reactor_design_model, data_rec, theta_names, SSE)
obj, theta, model_vals = pest.theta_est(return_model_values=['ca', 'cb', 'cc', 'cd'])
print(obj)
print(theta)

data_rec2 = pd.DataFrame(model_vals)

plt.figure()
data_rec2.boxplot()
plt.ylim([0,5000])

data_diff = data_rec - data_rec2
plt.figure()
data_diff.boxplot()


### Data reconciliation with prameter estimation

theta_names = ['k1', 'k2', 'k3', 'caf']

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta, model_vals = pest.theta_est(return_model_values=['ca', 'cb', 'cc', 'cd'])
print(obj)
print(theta)

data_rec3 = pd.DataFrame(model_vals)

plt.figure()
data_rec3.boxplot()
plt.ylim([0,5000])
