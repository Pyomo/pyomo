def create_model_dae(scena, CA_init=3, controls={0: 300, 0.125: 300, 0.25: 300, 0.375: 300, 0.5: 300, 0.625: 300, 0.75: 300, 0.875: 300, 1: 300}, t_range=[0.0,1]):
    
    # parameters initialization, results from parameter estimation
    theta_pe = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}
    
    ### Add variables 
    m = ConcreteModel()
    
    m.para_list = ['A1', 'A2', 'E1', 'E2']
    t_control = list(controls.keys())
    
    m.scena_all = scena 
    m.scena = Set(initialize=scena['scena-name'])
    
    # timepoints
    m.t = ContinuousSet(bounds=(t_range[0], t_range[1]))
    
    # Control time points for time dependent control variables
    m.t_con = Set(initialize=t_control)
    # Control time point for time independent control variables
    m.t0 = Set(initialize=[0])
    
    # time-independent design variable
    m.CA0 = Var(m.t0, initialize = CA_init, bounds=(1.0,5.0), within=NonNegativeReals) # mol/L
    # time-dependent design variable
    m.T = Var(m.t, initialize =350, bounds=(300, 700), within=NonNegativeReals)
     
    m.R = 8.31446261815324 # J / K / mole
       
    # Define parameters
    m.A1 = Param(m.scena, initialize=scena['A1'],mutable=True)
    m.A2 = Param(m.scena, initialize=scena['A2'],mutable=True)
    m.E1 = Param(m.scena, initialize=scena['E1'],mutable=True)
    m.E2 = Param(m.scena, initialize=scena['E2'],mutable=True)
    
    # Concentration variables under perturbation
    m.CA = Var(m.scena, m.t, initialize=0.3, within=NonNegativeReals)
    m.CB = Var(m.scena, m.t, initialize=0.3, within=NonNegativeReals)
    m.CC = Var(m.scena, m.t, initialize=0.3, within=NonNegativeReals)

    # time derivative of C
    m.dCAdt = DerivativeVar(m.CA, wrt=m.t)  
    m.dCBdt = DerivativeVar(m.CB, wrt=m.t)  
    m.dCCdt = DerivativeVar(m.CC, wrt=m.t)  
    
    # kinetic parameters
    m.kp1 = Var(m.scena, m.t, initialize=m.A1[s] * exp(-m.E1[s]*1000/(m.R*m.T[t])))
    m.kp2 = Var(m.scena, m.t, initialize=m.A2[s] * exp(-m.E2[s]*1000/(m.R*m.T[t]))


    def T_control(m,t):
        '''
        Enable piecewise constant control profile
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
        return m.kp1[z,t] == m.A1[z]*exp(-m.E1[z]*1000/(m.R*m.T[t])) 
            
    def cal_kp2(m,z,t):
        return m.kp2[z,t] == m.A2[z]*exp(-m.E2[z]*1000/(m.R*m.T[t])) 
        
    def dCAdt_control(m,z,t):
        return m.dCAdt[z,t] == -m.kp1[z,t]*m.CA[z,t]    
    
    def dCBdt_control(m,z,t):
        return m.dCBdt[z,t] == m.kp1[z,t]*m.CA[z,t] - m.kp2[z,t]*m.CB[z,t]
        
    def alge(m,z,t):
        return m.CA[z,t] + m.CB[z,t] + m.CC[z,t] == m.CA0[0]
        
        
    # Control time
    m.T_rule = Constraint(m.t, rule=T_control)
    
    # calculating C
    m.k1_pert_rule = Constraint(m.scena, m.t, rule=cal_kp1)
    m.k2_pert_rule = Constraint(m.scena, m.t, rule=cal_kp2)
    m.dCAdt_rule = Constraint(m.scena, m.t, rule=dCAdt_control)
    m.dCBdt_rule = Constraint(m.scena, m.t, rule=dCBdt_control)
    m.alge_rule = Constraint(m.scena, m.t, rule=alge)

    # B.C.
    for z in m.scena:
        m.CB[z,0.0].fix(0.0)
        m.CC[z,0.0].fix(0.0)
    
    return m 