#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________


import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import numpy as np
from pyomo.contrib.doe.measurements import Measurements

def disc_for_measure(m, NFE=32):
    """Pyomo.DAE discretization
    """
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    for z in m.scena:
        for t in m.t:
            m.dCdt_rule[z,'CC',t].deactivate()
    return m


def create_model(scena, const=False, control_time=None, control_val=None, 
                     t_range=[0.0,1], CA_init=1, C_init=0.1, model_form='dae-index-1', args=[True]):
    """
    This is an example user model provided to DoE library. 
    It is a dynamic problem solved by Pyomo.DAE.
    
    Arguments
    ---------
    scena: a dictionary of scenarios, achieved from scenario_generator()
    control_time: time-dependent design (control) variables, a list of control timepoints
    control_val: control design variable values T at corresponding timepoints
    t_range: time range, h 
    CA_init: time-independent design (control) variable, an initial value for CA
    C_init: An initial value for C
    model_form: choose from 'ode-index-0' and 'dae-index-1'
    args: a list, deciding if the model is for k_aug or not. If [False], it is for k_aug, the parameters are defined as Var instead of Param.
        
    Return
    ------
    m: a Pyomo.DAE model 
    """
    # parameters initialization, results from parameter estimation
    theta_pe = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}
    # concentration initialization
    y_init = {'CA': CA_init, 'CB':0.0, 'CC':0.0}
    
    para_list = ['A1', 'A2', 'E1', 'E2']

    if not control_time:
        control_time = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]
        
    if not control_val:
        control_val = [300]*9

    controls = {}
    for i, t in enumerate(control_time):
        controls[t]=control_val[i]
    
    ### Add variables 
    m = pyo.ConcreteModel()
    
    m.CA_init = CA_init
    m.para_list = para_list
    t_control = control_time
    
    m.scena_all = scena 
    m.scena = pyo.Set(initialize=scena['scena-name'])
    
    if model_form == 'ode-index-0':
        m.index1 = False
    elif model_form == 'dae-index-1':
        m.index1 = True 
    else:
        raise ValueError('Please choose from "ode-index-0" and "dae-index-1"')
    
    # timepoints
    m.t = ContinuousSet(bounds=(t_range[0], t_range[1]))
    
    # Control time points
    m.t_con = pyo.Set(initialize=t_control)
    
    m.t0 = pyo.Set(initialize=[0])
    
    # time-independent design variable
    m.CA0 = pyo.Var(m.t0, initialize = CA_init, bounds=(1.0,5.0), within=pyo.NonNegativeReals) # mol/L
    
    # time-dependent design variable, initialized with the first control value
    def T_initial(m,t):
        if t in m.t_con:
            return controls[t]
        else:
            # count how many control points are before the current t;
            # locate the nearest neighbouring control point before this t
            j = -1 
            for t_con in m.t_con:
                if t>t_con:
                    j+=1
            neighbour_t = t_control[j]
            return controls[neighbour_t]
    
    m.T = pyo.Var(m.t, initialize =T_initial, bounds=(300, 700), within=pyo.NonNegativeReals)
     
    m.R = 8.31446261815324 # J / K / mole
       
    # Define parameters as Param
    if args[0]:
        m.A1 = pyo.Param(m.scena, initialize=scena['A1'],mutable=True)
        m.A2 = pyo.Param(m.scena, initialize=scena['A2'],mutable=True)
        m.E1 = pyo.Param(m.scena, initialize=scena['E1'],mutable=True)
        m.E2 = pyo.Param(m.scena, initialize=scena['E2'],mutable=True)
    
    # if False, define parameters as Var (for k_aug)
    else:
        m.A1 = pyo.Var(m.scena, initialize = m.scena_all['A1'])
        m.A2 = pyo.Var(m.scena, initialize = m.scena_all['A2'])
        m.E1 = pyo.Var(m.scena, initialize = m.scena_all['E1'])
        m.E2 = pyo.Var(m.scena, initialize = m.scena_all['E2'])
    
    # Concentration variables under perturbation
    m.C_set = pyo.Set(initialize=['CA','CB','CC'])
    m.C = pyo.Var(m.scena, m.C_set, m.t, initialize=C_init, within=pyo.NonNegativeReals)

    # time derivative of C
    m.dCdt = DerivativeVar(m.C, wrt=m.t)  

    # kinetic parameters
    def kp1_init(m,s,t):
        return m.A1[s] * pyo.exp(-m.E1[s]*1000/(m.R*m.T[t]))
    def kp2_init(m,s,t):
        return m.A2[s] * pyo.exp(-m.E2[s]*1000/(m.R*m.T[t]))
    
    m.kp1 = pyo.Var(m.scena, m.t, initialize=kp1_init)
    m.kp2 = pyo.Var(m.scena, m.t, initialize=kp2_init)


    def T_control(m,t):
        """
        T at interval timepoint equal to the T of the control time point at the beginning of this interval
        Count how many control points are before the current t;
        locate the nearest neighbouring control point before this t
        """
        if t in m.t_con:
            return pyo.Constraint.Skip
        else:
            j = -1 
            for t_con in m.t_con:
                if t>t_con:
                    j+=1
            neighbour_t = t_control[j]
            return m.T[t] == m.T[neighbour_t]
        
    
    def cal_kp1(m,z,t):
        """
        Create the perturbation parameter sets 
        m: model
        z: scenario number
        t: time
        """
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp1[z,t] == m.A1[z]*pyo.exp(-m.E1[z]*1000/(m.R*m.T[t])) 
            
    def cal_kp2(m,z,t):
        """
        Create the perturbation parameter sets 
        m: model
        z: m.pert, upper or normal or lower perturbation
        t: time
        """
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp2[z,t] == m.A2[z]*pyo.exp(-m.E2[z]*1000/(m.R*m.T[t])) 
        
    def dCdt_control(m,z,y,t):
        """
        Calculate CA in Jacobian matrix analytically 
        z: scenario No.
        y: CA, CB, CC
        t: timepoints
        """
        if y=='CA':
            return m.dCdt[z,y,t] == -m.kp1[z,t]*m.C[z,'CA',t]    
        elif y=='CB':
            return m.dCdt[z,y,t] == m.kp1[z,t]*m.C[z,'CA',t] - m.kp2[z,t]*m.C[z,'CB',t]
        elif y=='CC':
            return m.dCdt[z,y,t] == m.kp2[z,t]*m.C[z,'CB',t]
        
    def alge(m,z,t):
        """
        The algebraic equation for mole balance
        z: m.pert
        t: time
        """
        return m.C[z,'CA',t] + m.C[z,'CB',t] + m.C[z,'CC', t] == m.CA0[0]
    
        
    # Control time
    m.T_rule = pyo.Constraint(m.t, rule=T_control)
    
    # calculating C, Jacobian, FIM
    m.k1_pert_rule = pyo.Constraint(m.scena, m.t, rule=cal_kp1)
    m.k2_pert_rule = pyo.Constraint(m.scena, m.t, rule=cal_kp2)
    m.dCdt_rule = pyo.Constraint(m.scena,m.C_set, m.t, rule=dCdt_control)

    m.alge_rule = pyo.Constraint(m.scena, m.t, rule=alge)

    # B.C. 
    for z in m.scena:
        m.C[z,'CB',0.0].fix(0.0)
        m.C[z,'CC',0.0].fix(0.0)
    
    return m 
