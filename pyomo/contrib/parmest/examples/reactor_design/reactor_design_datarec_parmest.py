import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pyomo.contrib.parmest.parmest as parmest
from reactor_design_datarec import reactor_design_model

plt.close('all')

np.random.seed(1234)

### Generate data based on real sv, caf, ca, cb, cc, and cd
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

data['sv'] = sv_real
data['caf'] = caf_real

plt.figure()
data[['ca', 'cb', 'cc', 'cd']].boxplot()

data_std = data.std()

# Define sum of squared error objective function
def SSE(model, data): 
    expr = ((float(data['ca']) - model.ca)/float(data_std['ca']))**2 + \
           ((float(data['cb']) - model.cb)/float(data_std['cb']))**2 + \
           ((float(data['cc']) - model.cc)/float(data_std['cc']))**2 + \
           ((float(data['cd']) - model.cd)/float(data_std['cd']))**2
    return expr

### Data reconciliation

theta_names = [] # no variables to estimate, use initialized values

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta, data_rec = pest.theta_est(return_values=['ca', 'cb', 'cc', 'cd'])
print(obj)
print(theta)

parmest.grouped_boxplot(data[['ca', 'cb', 'cc', 'cd']], data_rec, 
                        group_names=['Data', 'Data Rec'])


### Parameter estimation using reconciled data

theta_names = ['k1', 'k2', 'k3']
data_rec['sv'] = data['sv']
data_rec['caf'] = data['caf']

pest = parmest.Estimator(reactor_design_model, data_rec, theta_names, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)
print(theta_real)


### Data reconciliation with parameter estimation

theta_names = ['k1', 'k2', 'k3']

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta, data_rec2 = pest.theta_est(return_values=['ca', 'cb', 'cc', 'cd'])
print(obj)
print(theta)

parmest.grouped_boxplot(data[['ca', 'cb', 'cc', 'cd']], data_rec2, 
                        group_names=['Data', 'Data Rec'])
