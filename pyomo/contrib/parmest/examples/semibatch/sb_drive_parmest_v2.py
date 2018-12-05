import numpy as np
import json
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
from semibatch import generate_model

data = {} # 0 indexed dictionary
for exp_num in range(10):
    fname = 'exp'+str(exp_num+1)+'.out'
    with open(fname,'r') as infile:
        data[exp_num] = json.load(infile)
        data[exp_num]['experiment'] = exp_num+1

thetalist = ['k1', 'k2', 'E1', 'E2']

# The model is already setup to use a single experiment and includes SecondStageCost
model = generate_model(data[1]) 
print(model.SecondStageCost)

pest = parmest.Estimator(generate_model, data, thetalist)
objval, thetavals = pest.theta_est()
print ("objective value=",str(objval))
print ("theta-star=",str(thetavals))

### Parameter estimation with bootstrap
num_bootstraps = 10
bootstrap_theta = pest.bootstrap(num_bootstraps)
print ("Bootstrap:")
print(bootstrap_theta)
grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, 
                                 filename="SB_boot.png")

### Make the LR plot
### special hard-wired search ranges for debugging
search_ranges = {}
search_ranges["E1"] = np.arange(29000, 32000, 1000)
search_ranges["E2"] = np.arange(38000, 42000, 1000) 
search_ranges["k1"] = np.arange(4, 24, 6) 
search_ranges["k2"] = np.arange(40, 160, 80)
SSE = pest.likelihood_ratio(search_ranges=search_ranges)
print ("Likelihood Ratio:")
print(SSE)
grph.pairwise_likelihood_ratio_plot(SSE, objval, 0.8, len(data), 
                                        filename="SB_LR.png")
