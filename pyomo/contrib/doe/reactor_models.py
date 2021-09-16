from pyomo.environ import *
from pyomo.dae import *
import numpy as np



def discretizer(m, NFE=32):
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    #m = discretizer.reduce_collocation_points(m, var=m.T, ncp=1, contset=m.t)
    return m 

time_an = list(np.linspace(0,1,97))
def create_model_overall(scena, options,controls={0: 300, 0.125: 300, 0.25: 300, 0.375: 300, 0.5: 300, 0.625: 300, 0.75: 300, 0.875: 300, 1: 300}, t_range=[0.0,1], CA_init=3, C_init=0.3, model_form='dae-index-1'):
    '''
    This is an example user model provided to DoE library. 
    It is a dynamic problem solved by Pyomo.DAE.
    
    Arguments:
        scena: a dictionary of scenarios, achieved from scenario_generator()
        
        Controlled time-dependent design variable:
            - controls: a Dict, keys are control timepoints, values are the controlled T at that timepoint
        
        t_range: time range 

        Time-independent design variable: 
            - CA_init: CA0 value
        
        C_init: An initial value for C
        model_form: choose from 'ode-index-0' and 'dae-index-1'
        
    Return:
        m: a Pyomo.DAE model 
    '''
    # parameters initialization, results from parameter estimation
    theta_pe = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}
    # concentration initialization
    y_init = {'CA': CA_init, 'CB':0.0, 'CC':0.0}
    
    para_list = ['A1', 'A2', 'E1', 'E2']
    
    if_dae = options[0]
    if_constant = options[1]
    if_kaug = options[2]
    
    ### Add variables 
    m = ConcreteModel()
    
    m.CA_init = CA_init
    m.para_list = para_list
    t_control = list(controls.keys())
    
    m.scena_all = scena 
    m.scena = Set(initialize=scena['scena-name'])
    
    if model_form == 'ode-index-0':
        m.index1 = False
    elif model_form == 'dae-index-1':
        m.index1 = True 
    else:
        raise ValueError('Please choose from "ode-index-0" and "dae-index-1"')
    
    # timepoints
    if if_dae:
        m.t = ContinuousSet(bounds=(t_range[0], t_range[1]))
    else:
        m.t = Set(initialize=time_an)
    
    # Control time points
    if if_dae:
        m.t_con = Set(initialize=t_control)
    else:
        m.t_con = Set(initialize=[0])
        
    m.t0 = Set(initialize=[0])
    
    # time-independent design variable
    m.CA0 = Var(m.t0, initialize = CA_init, bounds=(1.0,5.0), within=NonNegativeReals) # mol/L
    
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
    
    if if_dae:
        m.T = Var(m.t, initialize =T_initial, bounds=(300, 700), within=NonNegativeReals)
    else:
        m.T = Var(m.t, initialize = T_initial, bounds=(300, 700), within=NonNegativeReals)
    
    m.R = 8.31446261815324 # J / K / mole
       
    # Define parameters
    if if_kaug: 
        m.A1 = Var(m.scena, initialize = m.scena_all['A1'])
        m.A2 = Var(m.scena, initialize = m.scena_all['A2'])
        m.E1 = Var(m.scena, initialize = m.scena_all['E1'])
        m.E2 = Var(m.scena, initialize = m.scena_all['E2'])
    else:
        m.A1 = Param(m.scena, initialize=scena['A1'],mutable=True)
        m.A2 = Param(m.scena, initialize=scena['A2'],mutable=True)
        m.E1 = Param(m.scena, initialize=scena['E1'],mutable=True)
        m.E2 = Param(m.scena, initialize=scena['E2'],mutable=True)
    
    # Concentration variables under perturbation
    m.CA = Var(m.scena, m.t, initialize=C_init, within=NonNegativeReals)
    m.CB = Var(m.scena, m.t, initialize=C_init, within=NonNegativeReals)
    m.CC = Var(m.scena, m.t, initialize=C_init, within=NonNegativeReals)

    if if_dae:
        # time derivative of C
        m.dCAdt = DerivativeVar(m.CA, wrt=m.t)  
        m.dCBdt = DerivativeVar(m.CB, wrt=m.t)  
        m.dCCdt = DerivativeVar(m.CC, wrt=m.t)  
    
    # kinetic parameters
    def kp1_init(m,s,t):
        return m.A1[s] * exp(-m.E1[s]*1000/(m.R*m.T[t]))
    def kp2_init(m,s,t):
        return m.A2[s] * exp(-m.E2[s]*1000/(m.R*m.T[t]))
    
    m.kp1 = Var(m.scena, m.t, initialize=kp1_init)
    m.kp2 = Var(m.scena, m.t, initialize=kp2_init)


    def T_control(m,t):
        '''
        T at interval timepoint equal to the T of the control time point at the beginning of this interval
        Count how many control points are before the current t;
        locate the nearest neighbouring control point before this t
        '''
        if t in m.t_con:
            return Constraint.Skip
        else:
            j = -1 
            for t_con in m.t_con:
                if t>t_con:
                    j+=1
            neighbour_t = t_control[j]
            return m.T[t] == m.T[neighbour_t]
        
    
    def cal_kp1(m,z,t):
        '''
        Create the perturbation parameter sets 
        m: model
        z: scenario number
        t: time
        '''
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp1[z,t] == m.A1[z]*exp(-m.E1[z]*1000/(m.R*m.T[t])) 
            
    def cal_kp2(m,z,t):
        '''
        Create the perturbation parameter sets 
        m: model
        z: m.pert, upper or normal or lower perturbation
        t: time
        '''
        # LHS: 1/h
        # RHS: 1/h*(kJ/mol *1000J/kJ / (J/mol/K) / K)
        return m.kp2[z,t] == m.A2[z]*exp(-m.E2[z]*1000/(m.R*m.T[t])) 
        
    def dCAdt_control(m,z,t):
        '''
        Calculate CA in Jacobian matrix analytically 
        z: scenario No.
        t: timepoints
        '''
        return m.dCAdt[z,t] == -m.kp1[z,t]*m.CA[z,t]    
    
    def dCBdt_control(m,z,t):
        '''
        Calculate CB in Jacobian matrix analytically 
        z: scenario No.
        t: timepoints
        '''
        return m.dCBdt[z,t] == m.kp1[z,t]*m.CA[z,t] - m.kp2[z,t]*m.CB[z,t]
    
    def dCCdt_control(m,z,t):
        '''
        Calculate CC in Jacobian matrix analytically 
        z: scenario No.
        t: timepoints
        '''
        return m.dCCdt[z,t] == m.kp2[z,t]*m.CB[z,t]
        
    def alge(m,z,t):
        '''
        The algebraic equation for mole balance
        z: m.pert
        t: time
        '''
        return m.CA[z,t] + m.CB[z,t] + m.CC[z,t] == m.CA0[0]
    
    # Calculate model response variables
    def CA_conc(m,z,t):
        '''
        Calculate the model predictions
        Argument: 
            z: scenario
            t: timepoints
        '''
        return m.CA[z,t] == m.CA0[0]*exp(-m.kp1[z,t]*t)
    
    def CB_conc(m,z,t):
        '''
        Calculate the model predictions
        Argument: 
            z: scenario
            t: timepoints
        '''
        return m.CB[z,t] == m.kp1[z,t]*m.CA0[0]/(m.kp2[z,t]-m.kp1[z,t]) * (exp(-m.kp1[z,t]*t) - exp(-m.kp2[z,t]*t))
    
    def obj_rule(m):
        return 0
    
    m.Obj = Objective(rule=obj_rule, sense=maximize)
        
    
    # calculating C, Jacobian, FIM
    m.k1_pert_rule = Constraint(m.scena, m.t, rule=cal_kp1)
    m.k2_pert_rule = Constraint(m.scena, m.t, rule=cal_kp2)
    if if_dae:
        m.dCAdt_rule = Constraint(m.scena, m.t, rule=dCAdt_control)
        m.dCBdt_rule = Constraint(m.scena, m.t, rule=dCBdt_control)
        # switch between DAE and ODE model
        if m.index1:
            m.alge_rule = Constraint(m.scena, m.t, rule=alge)
        else:
            m.dCCdt_rule = Constraint(m.scena, m.t, rule=dCCdt_control)
            
        if if_constant:
            m.T_rule = Constraint(m.t, rule=T_control)
            
    else:
        m.dCAdt_rule = Constraint(m.scena, m.t, rule=CA_conc)
        m.dCBdt_rule = Constraint(m.scena, m.t, rule=CB_conc)
        m.dCCdt_rule = Constraint(m.scena, m.t, rule=alge)

    if if_dae:    
        # B.C. 
        for z in m.scena:
            # only ODE model needs this BC
            if not m.index1:
                m.CA[z,0.0].fix(value(m.CA0[0]))
            m.CB[z,0.0].fix(0.0)
            m.CC[z,0.0].fix(0.0)
    
    return m 


