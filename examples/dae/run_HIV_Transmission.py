#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *
from pyomo.dae import *
#from HIV_Transmission import m
from HIV_Transmission import m

##########################################
# The control values to use for simulation 
##########################################
vp_control_value = 0.75
vt_control_value = 0.75

###########################################################
# Option for running the simulation or optimization problem
###########################################################
optimization_problem = True

####################
# Simulate the model
####################
from pyomo.dae.simulator import Simulator

vp_profile = {0:vp_control_value}
vt_profile = {0:vt_control_value}

# To simulate a step-profile change profile dictionaries as shown 
# Note: This does not fix a step-profile for IPOPT solve and will 
# not appear in the plot of the controls
# vp_profile = {0:vp_control_value, 10:0.75}

m.u_input = Suffix(direction=Suffix.LOCAL)
m.u_input[m.vp] = vp_profile
m.u_input[m.vt] = vt_profile

sim = Simulator(m, package='scipy')
tsim, profiles = sim.simulate(numpoints=100, varying_inputs=m.u_input)

#################################
# Discretize and initialize model
#################################
# Discretize model using Finite Difference Method
# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=200,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=10,ncp=3,scheme='LAGRANGE-RADAU')

sim.initialize_model()

#######################################
# Use element-by-element initialization
#######################################
# from pyomo.dae.initialization import initialize_by_element
# initialize_by_element(m, solver='ipopt', groupby=2, contset=m.t)

#####################
# Plot Initialization
#####################
import matplotlib.pyplot as plt

cmap = plt.get_cmap('Set1')
cmap = [cmap(i) for i in range(10)]

time = list(m.t)
label=True
for idx, color in zip(m.ij, cmap):
    yval = [value(m.yy[t,idx]) for t in m.t]
    plt.subplot(121)
    if label:
        plt.plot(time, yval, '.', color=color, label='Initialization', markersize=8)
        label=False
    else:
        plt.plot(time, yval, '.', color=color, markersize=8)

        

#########################################################
# Differences between simulation and optimization problem
#########################################################

if optimization_problem:
    discretizer.reduce_collocation_points(m,var=m.vp,ncp=1,contset=m.t)
    discretizer.reduce_collocation_points(m,var=m.vt,ncp=1,contset=m.t)
else:
    m.vp.fix(vp_control_value)
    m.vt.fix(vt_control_value)
    m.del_component(m.cf)
    m.cf = Objective(expr=5)

############################
# Solve the model with IPOPT
############################
solver=SolverFactory('ipopt')
results = solver.solve(m, tee=True)

print('Cost: ',value(m.L[m.t.last()]))

#####################
# Plot IPOPT Solution
#####################
time = list(m.t)
label = True
for idx, color in zip(m.ij, cmap):
    yval = [value(m.yy[t,idx]) for t in m.t]
    plt.subplot(121)
    if label:
        label=False
        plt.plot(time, yval, color=color, label='IPOPT Solution', linewidth=2)
        plt.legend(loc='best')
    else:
        plt.plot(time, yval, color=color, linewidth=2)

plt.subplot(122)
plt.plot(time[1:],[value(m.vp[t]) for t in m.t if t!=0],'-', label='vp')
plt.plot(time[1:],[value(m.vt[t]) for t in m.t if t!=0],'-', label='vt')
plt.legend(loc='best', prop={'size':16})
plt.show()
