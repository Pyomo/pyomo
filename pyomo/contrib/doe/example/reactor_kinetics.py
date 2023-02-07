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
from measurements import Measurements

def disc_for_measure(m, NFE=32):
    """Pyomo.DAE discretization
    """
    discretizer = pyo.TransformationFactory('dae.collocation')
    #discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    for s in range(8):
        discretizer.apply_to(m.lsb[s], nfe=NFE, ncp=3, wrt=m.lsb[s].t)
    return m


def create_model(m=None, control_time=None, control_val=None, 
                     t_range=[0.0,1], CA_init=1, C_init=0.1):
    """
    This is an example user model provided to DoE library. 
    It is a dynamic problem solved by Pyomo.DAE.
    
    Arguments
    ---------
    m: Pyomo model. If None, a Pyomo concrete model is created
    control_time: time-dependent design (control) variables, a list of control timepoints
    control_val: control design variable values T at corresponding timepoints
    t_range: time range, h 
    CA_init: time-independent design (control) variable, an initial value for CA
    C_init: An initial value for C

    Return
    ------
    m: a Pyomo.DAE model 
    """

    theta = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

    if not m: 
        m = pyo.ConcreteModel()
        return_m = True
    else:
        return_m = False
    
    para_list = ['A1', 'A2', 'E1', 'E2']

    if not control_time:
        control_time = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]
        
    if not control_val:
        control_val = [300]*9

    controls = {}
    for i, t in enumerate(control_time):
        controls[t]=control_val[i]
    
    ### Add variables 
    m.CA_init = CA_init
    m.para_list = para_list
    t_control = control_time
    
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
    m.A1 = pyo.Var(initialize = theta['A1'])
    m.A2 = pyo.Var(initialize = theta['A2'])
    m.E1 = pyo.Var(initialize = theta['E1'])
    m.E2 = pyo.Var(initialize = theta['E2'])
    
    # Concentration variables under perturbation
    m.C_set = pyo.Set(initialize=['CA','CB','CC'])
    m.C = pyo.Var(m.C_set, m.t, initialize=C_init, within=pyo.NonNegativeReals)

    # time derivative of C
    m.dCdt = DerivativeVar(m.C, wrt=m.t)  

    # kinetic parameters
    def kp1_init(m,t):
        return m.A1 * pyo.exp(-m.E1*1000/(m.R*m.T[t]))
    def kp2_init(m,t):
        return m.A2 * pyo.exp(-m.E2*1000/(m.R*m.T[t]))
    
    m.kp1 = pyo.Var(m.t, initialize=kp1_init)
    m.kp2 = pyo.Var(m.t, initialize=kp2_init)


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
        
    
    def cal_kp1(m,t):
        """
        Create the perturbation parameter sets 
        m: model
        t: time
        """
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp1[t] == m.A1*pyo.exp(-m.E1*1000/(m.R*m.T[t])) 
            
    def cal_kp2(m,t):
        """
        Create the perturbation parameter sets 
        m: model
        t: time
        """
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp2[t] == m.A2*pyo.exp(-m.E2*1000/(m.R*m.T[t])) 
        
    def dCdt_control(m,y,t):
        """
        Calculate CA in Jacobian matrix analytically 
        y: CA, CB, CC
        t: timepoints
        """
        if y=='CA':
            return m.dCdt[y,t] == -m.kp1[t]*m.C['CA',t]    
        elif y=='CB':
            return m.dCdt[y,t] == m.kp1[t]*m.C['CA',t] - m.kp2[t]*m.C['CB',t]
        elif y=='CC':
            return pyo.Constraint.Skip
        
    def alge(m,t):
        """
        The algebraic equation for mole balance
        z: m.pert
        t: time
        """
        return m.C['CA',t] + m.C['CB',t] + m.C['CC', t] == m.CA0[0]
    
        
    # Control time
    m.T_rule = pyo.Constraint(m.t, rule=T_control)
    
    # calculating C, Jacobian, FIM
    m.k1_pert_rule = pyo.Constraint(m.t, rule=cal_kp1)
    m.k2_pert_rule = pyo.Constraint(m.t, rule=cal_kp2)
    m.dCdt_rule = pyo.Constraint(m.C_set, m.t, rule=dCdt_control)

    m.alge_rule = pyo.Constraint(m.t, rule=alge)

    # B.C. 
    m.C['CB',0.0].fix(0.0)
    m.C['CC',0.0].fix(0.0)
    
    if return_m:
        return m 
