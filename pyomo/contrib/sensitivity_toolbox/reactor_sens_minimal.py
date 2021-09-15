#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
#from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from idaes.apps.uncertainty_propagation.sens import get_dsdp


'''
This is a minimal example of a Pyomo.DAE problem.  
 
Consider two chemical reactions that converts molecule $A$ to desired product $B$ and a less valuable side-product $C$.
 
$A \overset{k_1}{\rightarrow} B \overset{k_2}{\rightarrow} C$
 
 
The concenrations in a batch reactor evolve with time per the following differential equations:
 
$$ \frac{d C_A}{dt} = r_A = -k_1 C_A $$
 
$$ \frac{d C_B}{dt} = r_B = k_1 C_A - k_2 C_B $$

$$ \frac{d C_C}{dt} = r_C = k_2 C_B $$

This is a linear system of differential equations. Assuming the feed is only species $A$, i.e., 

$$C_A(t=0) = C_{A0} \quad C_B(t=0) = 0 \quad C_C(t=0) = 0$$

Measurements (s) in this problem: CA, CB, CC

Parameters (p): A1, A2, E1, E2

Dynamic variable: T (temperature). When T is constant, we have analytical solution to ds/dp. 

This problem is integrated by Pyomo.DAE. We use the get_dsdp() function to achieve the partial derivative ds/dp
'''

def create_model(CA_init=5, T_init=300):
    '''Create the model with Pyomo.DAE
    '''
    # parameters initialization
    theta_p = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}

    ### add variable 
    m= ConcreteModel()
    
    ### Define sets and expressions
    # timepoint
    m.t = ContinuousSet(bounds=(0.0,1))
    # control time points
    m.t_con = Set(initialize=[0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1])
    
    m.CA0 = Var(initialize = CA_init, bounds=(1.0,5.0), within=NonNegativeReals) # mol/L
    m.CA0.fix()
    
    m.T = Var(m.t, initialize = T_init, bounds=(300, 700), within=NonNegativeReals)

    # ideal gas constant
    m.R = 8.31446261815324 # J / K / mol

    # parameters 
    m.A1 = Var(initialize = theta_p['A1'])
    m.A2 = Var(initialize = theta_p['A2'])
    m.E1 = Var(initialize = theta_p['E1'])
    m.E2 = Var(initialize = theta_p['E2'])
    
    # Concentration of A, B, and C [mol/L]
    m.CA = Var( m.t, initialize=0.0, within=NonNegativeReals)
    m.CB = Var( m.t, initialize=0.0, within=NonNegativeReals)
    m.CC = Var( m.t, initialize=0.0, within=NonNegativeReals)
    
    # time derivative of C
    m.dCAdt = DerivativeVar(m.CA, wrt=m.t)  
    m.dCBdt = DerivativeVar(m.CB, wrt=m.t)  
    m.dCCdt = DerivativeVar(m.CC, wrt=m.t)  

    # state variables (rate constants)
    def cal_kp1(m,t):
        return m.A1*exp(-m.E1*1000/(m.R*m.T[t])) 
            
    def cal_kp2(m,t):
        return m.A2*exp(-m.E2*1000/(m.R*m.T[t])) 

    m.kp1 = Expression(m.t, rule = cal_kp1 ) # 1/hr
    m.kp2 = Expression(m.t, rule = cal_kp2 )
    
    
    # Calculate model response variables
    def CA_conc(m,t):
        return m.dCAdt[t] == -m.kp1[t]*m.CA[t]
    
    def CB_conc(m,t):
        return m.dCBdt[t] == m.kp1[t]*m.CA[t] - m.kp2[t]*m.CB[t]
    
    def CC_conc(m,t):
        return m.CC[t] == m.CA0 - m.CA[t] - m.CB[t]
    
    # add constraints
    m.dCAdt_rule = Constraint( m.t, rule=CA_conc)
    m.dCBdt_rule = Constraint(m.t, rule=CB_conc)
    m.dCCdt_rule = Constraint(m.t, rule=CC_conc)

    m.Obj = Objective(rule=0, sense=maximize)
    
    # initial state
    m.CB[0.0].fix(0.0)
    m.CC[0.0].fix(0.0)

    # fix parameters
    m.A1.setlb(theta_p['A1'])
    m.A2.setlb(theta_p['A2'])
    m.E1.setlb(theta_p['E1'])
    m.E2.setlb(theta_p['E2'])
    m.A1.setub(theta_p['A1'])
    m.A2.setub(theta_p['A2'])
    m.E1.setub(theta_p['E1'])
    m.E2.setub(theta_p['E2'])
        
    return m

# ### DAE model 


def discretizer(m, NFE=32):
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t) 
    return m 

if __name__ == '__main__':
    sigma_p = np.array([[1, 0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0, 1]])

    variable_name = ['A1', 'A2', 'E1', 'E2']
    theta_p = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}

    # create model
    model_un = create_model(CA_init=5, T_init=300)
    # use pyomo.DAE
    model_un = discretizer(model_un)
    model_un.T.fix()

    # call get_dsdp()
    dsdp_re, col = get_dsdp(model_un, variable_name, theta_p, tee=False)
    
    # organize results
    dsdp_array = dsdp_re.toarray().T

    timepoint = [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]
    index_list = []

    ### extract dCA/dp, dCB/dp and dCC/dp from all columns
    measure_name = []
    for cname in ['CA', 'CB', 'CC']:
        for tim in timepoint:
            gene_name = cname+'['+str(tim)+']'
            measure_name.append(gene_name)

    # from the given column names by get_dsdp(), find columns we are interested
    for itera in range(len(measure_name)):
        no = col.index(measure_name[itera])
        index_list.append(no)

    # extract dCA/dp, dCB/dp and dCC/dp
    dsdp_extract = []
    for kk in index_list:
        dsdp_extract.append(dsdp_array[kk])

    # organize results in a dictionary 
    jac = {}
    jac['dC/dA1'] = []
    jac['dC/dA2'] = []
    jac['dC/dE1'] = []
    jac['dC/dE2'] = []

    for d in range(len(dsdp_extract)):
        jac['dC/dA1'].append(dsdp_extract[d][0])
        jac['dC/dA2'].append(dsdp_extract[d][1])
        jac['dC/dE1'].append(dsdp_extract[d][2])
        jac['dC/dE2'].append(dsdp_extract[d][3])

    print('======Pyomo.DAE and get_dsdp() solution======')
    print(jac)

def analytical_calc(CA_init=5, T_init=300):
    '''This functions calculates the dsdp analytically
    '''
    time = [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]
    
    # time 
    T = np.ones((8))*T_init 
    k1 = np.zeros((8))
    k2 = np.zeros((8))
    CA = np.zeros((8))
    CB = np.zeros((8))
    CC = np.zeros((8))
    
    # dsdp 
    dCAdA1 = np.zeros((8))
    dCAdE1 = np.zeros((8))
    dCAdA2 = np.zeros((8))
    dCAdE2 = np.zeros((8))
    dCBdA1 = np.zeros((8))
    dCBdE1 = np.zeros((8))
    dCBdA2 = np.zeros((8))
    dCBdE2 = np.zeros((8))
    dCCdA1 = np.zeros((8))
    dCCdE1 = np.zeros((8))
    dCCdA2 = np.zeros((8))
    dCCdE2 = np.zeros((8))
    
    R = 8.31446261815324
    
    # parameters 
    A1 = 84.79085853498033
    A2 = 371.71773413976416
    E1 = 7.777032028026428
    E2 = 15.047135137500822
    
    for i in range(8):
        # state variables
        k1[i] = A1*np.exp(-E1*1000/R/T[i])
        k2[i] = A2*np.exp(-E2*1000/R/T[i])
        CA[i] = CA_init*np.exp(-k1[i]*time[i])
        CB[i] = k1[i]/(k2[i]-k1[i]) * CA_init * (np.exp(-k1[i]*time[i]) - np.exp(-k2[i]*time[i]))
        CC[i] = CA_init - CA[i] - CB[i]
        
        # dCA/dp
        dCAdA1[i] = -CA_init * time[i] * np.exp(-k1[i]*time[i]-1000*E1/(R*T[i]))
        dCAdE1[i] = CA_init * A1* time[i]*1000/R/T[i] * np.exp(-k1[i]*time[i] - E1*1000/R/T[i])
        dCAdA2[i] = 0
        dCAdE2[i] = 0 
        
        # dCB/dp
        item1 = k2[i]/(k2[i] - k1[i])/(k2[i]-k1[i])*np.exp(-E1*1000/R/T[i])
        item2 = -time[i]*np.exp(-k1[i]*time[i] - E1*1000/R/T[i])
        dCBdA1[i] = item1*CA_init*np.exp(-k1[i]*time[i]) + k1[i]/(k2[i]-k1[i])*CA_init*item2 - CA_init*np.exp(-k2[i]*time[i])*item1

        item3 = -k1[i]/(k2[i] -k1[i])/(k2[i]-k1[i])*np.exp(-E2*1000/R/T[i])
        item4 = -time[i]*np.exp(-k2[i]*time[i]-E2*1000/R/T[i])
        dCBdA2[i] = item3*CA_init*np.exp(-k1[i]*time[i]) - item3*CA_init*np.exp(-k2[i]*time[i])-k1[i]/(k2[i]-k1[i])*CA_init*item4

        item5 = -k2[i]/(k2[i]-k1[i])/(k2[i]-k1[i])*A1*1000/R/T[i]*np.exp(-E1*1000/R/T[i])
        item6 = A1*1000*time[i]/R/T[i] * np.exp(-k1[i]*time[i] - E1*1000/R/T[i])
        dCBdE1[i] = item5*CA_init*np.exp(-k1[i]*time[i]) + k1[i]/(k2[i]-k1[i])*CA_init*item6 -item5*CA_init*np.exp(-k2[i]*time[i])

        item7 = k1[i]*A2*1000/(k2[i]-k1[i])/(k2[i]-k1[i])/R/T[i]*np.exp(-E2*1000/R/T[i])
        item8 = A2*1000*time[i]/R/T[i]*np.exp(-k2[i]*time[i] - E2*1000/R/T[i])
        dCBdE2[i] = item7*CA_init*np.exp(-k1[i]*time[i]) - k1[i]/(k2[i]-k1[i])*CA_init*item8 - item7*CA_init*np.exp(-k2[i]*time[i])

        # dCC/dp
        dCCdA1[i] = -dCAdA1[i] - dCBdA1[i]
        dCCdE1[i] = -dCAdE1[i] - dCBdE1[i]
        dCCdA2[i] = -dCAdA2[i] - dCBdA2[i]
        dCCdE2[i] = -dCAdE2[i] - dCBdE2[i]
        
    # organize dsdp
    jac={}
    jac['dC/dA1']= np.concatenate((dCAdA1, dCBdA1, dCCdA1))
    jac['dC/dA2']= np.concatenate((dCAdA2, dCBdA2, dCCdA2))
    jac['dC/dE1']= np.concatenate((dCAdE1, dCBdE1, dCCdE1))
    jac['dC/dE2']= np.concatenate((dCAdE2, dCBdE2, dCCdE2))
    
    return jac
   
print('=====Analytical solution for dsdp======')
jac_analytic = analytical_calc()
print(jac_analytic)