import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *

import numpy as np
from scipy.interpolate import interp2d
import pandas as pd

### Process options

# Options related to the sigmoid function
# two sigmoid functions, or 0/1. 
alpha_option = 2
# If alpha is treating as a variable or a parameter
alpha_variable = False
# eps_alpha: small number used in smoothed absolute value for alpha
eps_alpha = 1.0E-6 
# scale alpha. [??-bar]^2. When alpha_scale = 100, then [??-bar] is [hecto-bar].
alpha_scale = 500

print("alpha option:")
if alpha_option == 1:
    print("1 / sqrt(eps + x*x)")
elif alpha_option == 2:
    print("1 / (1 + exp(-x))")
else:
    print("WARNING! Not Supported")
    

### Data and model constants

# Molecular weight [g/mol]    
MW = {'N2':28.013, 'CO2':44.010}

# Number of axial grid elements
Ngrid = 20

# R as gas constant [kJ/mol/K]
RPV = 8.31446261815324E-3

# CO2 partial pressure when the pressure is too low , 0.003[MPa]
# Value from ACM 
plin = 0.03

# Radius of the bed [m]
# Value from MosaicDataAssessment.docx
rbed = 2.3E-3 

# Inlet gas volume rate [m^3/min]
# Value from Breakthrough-dmpn-Mg2dobpdc.xlsx
# Value is given as 10 sccm (standard cubic centimeters per minute), 1 m^3 = 1E6 cm^3
volume_in_standard = 1.0E-5
    
# Inlet assumption 1
# volume converted
# inlet temp 313.15
# inlet pressure unchanged
totp_f = 1
    


# CO2 viscosity, [Pa*s] = [kg/m/s]
# Value from the viscosity data(found online and uploaded to git repo)
# Get from interpolation for CO2 at 313.15K
mu_co2 = 1.56E-5

# CO2 mole mass, [kg/mol]
M_co2 = 0.044010

# Total feed concentration [kmol/m^3]
# Calculation result is consistent with that given by Breakthrough-dmpn-Mg2dobpdc.xlsx
# Helpful converstions: J = kg*m2/s2, Pa = kg/m/s2, 1 MPa = 1000 * 1000 Pa
# LHS: mol/m3                         
# RHS: kPa/ (kJ/mol) = mol/m3 = mol/m3
###totden_f = totp_f*100/(den_feed_temp*RPV)
    
# Inlet velocity [cm/s]
# RHS: m3/min/(m2)/(60s/min) *100cm/m = cm/s
#vel_f = volume_in* 100 /(np.pi*rbed*rbed) / 60
vel_f = 1.16638

# Check inlet molar flowrate (which should be 0.4467mmol/min)
# LHS: mmol/min
# RHS: mol/m3 * 1000 mmol/mol * m3/min
#mol_in = 0.4467

# Porosity of the bed [\]
# The interparticle voidage, value from MosaicDataAssessment.docx
ads_epsb = 0.68 

# Porosity of the particle [\]
# Intra-particle voidage, value from MosaicDataAssessment.docx
ads_epsp = 0.33 


# Density of the solid (no external voidage, only internal voidage) [kg/m3]
# Value from MosaicDataAssessment.docx
#den_bed = 1000 
den_s = 1000

# Solid density [kg/m3]
# Value of \rho_b in ACM 
den_b = den_s*(1-ads_epsb)

# sorbent specific heat [J/kg/K]
# value from Aug.report
cps = 1457

# Length of bed [m]
# Value from MosaicDataAssessment.docx
Lbed = 0.1334

# Length of each axial element [m]
dz = Lbed/Ngrid

# Viscosity [Pa*s]
# Calculated from viscosity data(found online and have uploaded to git repo)
# fp_mu = (n2_mu(313k,1atm) + co2_mu(313k,1atm))/2, kg/m/s = Pa * s
fp_mu = 1.7035E-5  


# Initialization for the axial dispersion coefficients
# Values set by no reference
Dax = {'N2':0.0001, 'CO2':0.00005}

## Bounds
# Used to use as a tighter bound[bar]
#P_low = 0.9
#P_high = 1.1

P_low = 0.1
P_high = 21

# Total density, mol/m3, MPa/RPV/temp
# LHS: mol/m3 # RHS: kPa/(kJ/mol) = mol/m3
tden_low = P_low*100 / RPV / 313.15
tden_high = P_high*100 / RPV / 313.15

# Adsorption kinetics constants
# Value from ParameterValues.xlsx
n_max = 3.82 # mol/kg
nmax_p1 = 3.52 # mol/kg 
Ka = -0.92 # dimensionless 
Kb = 324.86 # K
Kc = -71.14 # dimensionless
Kd = 28386.66 # K
b_a0 = 28.56 # 1/bar
b_b0 = 0.62 # 1/bar
Q_sta = 72.56 # kJ/mol
Q_stb = 43.83 # kJ/mol
T0 = 318 # K
n_a1 = 0.21 # dimensionless
E_na = 11.29 # kJ/mol

# Use kinetic parameters in March excel, not in Aug report
K_c0 = 0.011882 #1/s
K_p0 = 0.07895 #1/s
E_c = 23.20784 #kJ/mol
E_p = 7.17967 #kJ/mol 

# Radius of particles [m]
# Value from MosaicDataAssessment.docx
radp = 2625.0E-7 

# Film mass transfer coefficient [m/s]
# N2 and CO2 values from (Cavenati et al., 2006)
k_f ={'N2':2.56E-4, 'CO2':1.92E-4}

# Value from emails with Ryan, Aug 19, 2019 [m^2/s]
# D_m = 0.53cm2/s = 0.53 E-4 m2/s = 5.3E-5 m^2/s
D_m = 5.3E-5

# Small number used in isotherm
spp_small_number = 1E-8
#spp_small_number=1E-6

# Replace 0 for every lower bound to be this number
small_bound = 1E-8
#small_bound=1E-6

#small_initial = 1E-6
small_initial=0.01

# Initialize alpha
alpha0 = 1

# pore diameter
# fitted value from WVU report, [m]
rpore = 1.13E-11

# tortuosity
# fitted value from WVU report, [dimensionless]
tort = 50

# Reference temperature for calculation of h
temp_base = 298.15


def create_model(scena, temp=313.15, temp_bath=313.15, y=0.15, para=212, ua=5.0E6, opt = True, conti=False, optimize_trace=True, diff=0, eps=0.01, energy = True, doe_model=True, est_tr=False, est_ua=False, k_aug=False, temp_option=1, isotherm = True, chemsorb = True, physsorb = True, dispersion = False, fix_pres = False, v_fix=False):
    ''' 
    Creates a concrete Pyomo model and adds sets/parameters.
    Toggles are saved into the model object.
    
    Arguments: 
        temp: the gas feed temperature, [K]
        temp_bath: the water bathing temperature, [K]. If the energy balance is not included, temp == temp_bath
        y: CO2 feed composition, [0,1]
        para: the value of fitted_transport_parameter
        opt: if true, toggle on the design of experiments objective
        conti: if true, make the experiment design decisions (T, yinlet) degrees of freedom
        diff: 0: no derivative estimate, 1: forward, -1: backward, 2: central
        eps: step size for the finite difference perturbation
        energy: decide if energy balance is added to the model 
        doe_model: if this model is for the DesignOfExperiment.py. if true, k and ua will be defined as variable, not expressions.
        est_tr: if transport_coefficient will be estimated 
        est_ua: if ua will be estimated 
        k_aug: if this model is created for k_aug solver to get sensitivities
        temp_option: 1: Twall=313.15K; 2: Twall=Tinlet; 3(to be added):Tinlet=313.15K, Twall=specified temp. 
        isotherm: decide if isotherm part is opened. Must open if one of the chemsorb/physsorb is opened.
        chemsorb: decide if chemical adsorption part is opened to calculate adsorption kinetics
        physsorb: decide if physical adsorption part is opened to calculate adsorption kinetics
        dispersion: decide if dispersion part is included in gas molar balance calculation
        fix_pres: decide if fixing the pressure and removing the ergun equation. For debugging
        v_fix: decide if fixing the velocity and removing the ergun equation. For debugging 
            
    Return: the model
    '''
    
    # create concrete Pyomo model
    m = ConcreteModel()

    # Store model toggles
    m.para = para
    m.ua_init = ua
    m.opt     = opt
    m.conti = conti
    m.optimize_trace = optimize_trace
    m.diff = diff
    m.eps = eps
    m.energy = energy
    m.doe_model = doe_model
    m.est_tr = est_tr
    m.est_ua = est_ua
    m.k_aug = k_aug
    m.temp_option = temp_option
    m.isotherm = isotherm
    m.chemsorb = chemsorb
    m.physsorb = physsorb
    m.dispersion = dispersion
    m.fix_pres = fix_pres
    m.v_fix = v_fix
    
    m.temp_bath = temp_bath
    m.scena_all = scena
    m.scena = Set(initialize=scena['scena-name'])
    
    # If DOE is open, estimation cannot be opened, vice versa 
    #assert (m.est_tr or m.est_ua) != m.opt, 'Parameter estimation and design of experiment cannot be run simultaneously'
    assert m.opt >= m.conti, 'DoE must be opened when continuous DoE is opened'
    
    # declare components set
    m.COMPS = Set(initialize=['N2','CO2'])

    # declare components that adsorb
    m.SCOMPS = Set(initialize=['CO2'], within=m.COMPS)
    
    # components that didn't adsorb
    m.USCOMP = Set(initialize=['N2'], within=m.COMPS)
    
    # composition of feed
    yfeed = {'N2':1-y, 'CO2':y}
    
    # Original bed temperature, [K]
    if m.doe_model:
        m.temp_orig = Var(initialize=temp, bounds=(100, 600))
    elif m.conti:
        m.temp_orig = Var(initialize=temp, bounds=(293, 373))
    else:
        m.temp_orig = temp

    print('The inlet gas temperature is ', value(m.temp_orig))
    
    if m.conti:
        m.yfeed = Var(m.COMPS, initialize=yfeed, bounds=(0.1,0.45), within=NonNegativeReals)
    elif m.doe_model:
        m.yfeed = Var(initialize=y, bounds=(0,1), within=NonNegativeReals)
    else:
        m.yfeed = Param(m.COMPS, initialize=yfeed)
        
    
    
    # Film mass transfer coefficient, m/s
    m.kf = Param(m.COMPS, initialize=k_f)

    # Molecular weight [kg/kmol]    
    m.MW = Param(m.COMPS,initialize=MW)
    
    # Dax, the dispersion efficient 
    m.Dax = Param(m.COMPS,initialize=Dax)
    
    # Manually discretize axial dimension
    m.zgrid = Set(initialize=range(0,Ngrid))
   
    # Initial bed N2 concentration at time 0.0 [mol/m3]
    # LHS: mol/m3
    # RHS: [kPa] / ([K] * [kJ/mol/K]) = Pa/(J/mol) = mol/m^3
    m.den_inert = Param(initialize=totp_f*100/(m.temp_bath*RPV))

    # Total feed density, [mol/m3]
    m.totden_f = Param(initialize=totp_f*100/(m.temp_orig*RPV))
    
    print('The inlet feed density is', value(m.totden_f), '[mol/m3]')
    
    
        
    def den_f_rule(m,i):
        return m.yfeed[i]*value(m.totden_f)
    
    def den_f_rule_doe(m,i):
        if i=='CO2':
            return m.yfeed*value(m.totden_f)
        elif i=='N2':
            return (1-m.yfeed)*value(m.totden_f)
    
    if m.doe_model:
        m.den_f = Expression(m.COMPS, rule=den_f_rule_doe)
    else:
        m.den_f = Expression(m.COMPS, rule=den_f_rule)
        
    # Estimate coefficient for parameter estimation 
    if not m.est_tr:
        if m.doe_model: 
            m.fitted_transport_coefficient = Param(m.scena, initialize=m.scena_all['fitted_transport_coefficient'], mutable=True)
        elif m.k_aug:
            m.fitted_transport_coefficient = Var(m.scena, initialize=m.para)
            #m.fitted_transport_coefficient.setlb(m.para)
            #m.fitted_transport_coefficient.setub(m.para)
        else:
            m.fitted_transport_coefficient = Expression(m.perturb, rule=perturbations_k)
    else:
        m.fitted_transport_coefficient = Var(initialize=para,bounds=(100,300))
        
    # energy balance is added 
    if not m.energy:
        # For continuous, temp and yfeed will be variable
        if m.conti:
            m.temp = Var(initialize = m.temp_orig, bounds=(293.15, 373.15), within=NonNegativeReals) 
        else:
            m.temp = Param(initialize=m.temp_orig)
            
        # define inv_k_oc, inv_k_op according to perturbation
        # inv_k_oc is in [1/s], therefore the k_trans is in [1/s]
        def inv_k_oc_init(m, i):
            return m.fitted_transport_coefficient[i]+1/(K_c0*exp(-E_c/(RPV*m.temp) + E_c/(RPV*T0)))

        def inv_k_op_init(m, i):
            return m.fitted_transport_coefficient[i]+1/(K_p0*exp(-E_p/(RPV*m.temp) + E_p/(RPV*T0)))

        # For square problem/optimization problem, parameter, k_oc/p are parameters
        if not m.est_tr:
            m.inv_K_oc = Param(m.perturb,initialize=inv_k_oc_init)
            m.inv_K_op = Param(m.perturb,initialize=inv_k_op_init)
        
        # Eq 14 in Aug. 2018 WVU report
        # LHS: [1/bar]
        # RHS: [1/bar] * exp( [kJ/mol]/[kJ/mol/K * K] ) = [1/b] * exp( [ dimensionless ] )
        m.b_a = Param(initialize=b_a0*exp(Q_sta/(RPV*T0)*(T0/m.temp-1))) 

        # Eq 15 in Aug. 2018 WVU report
        # LHS: dimensionless
        # RHS: dimensionless * exp( [kJ/mol]/[kJ/mol/K * K] ) = [dimensionless] * exp([dimensionless ])
        m.n_a = Param(initialize=n_a1*exp(E_na/(RPV*T0)*(T0/m.temp-1)))

        #m.inv_n_a = Param(initialize=1/m.n_a)
        inv_n_a = 1/m.n_a

        # Eq 13 in Aug. 2018 WVU report
        # LHS: dimensionless
        # RHS: dimensionless + K/K
        m.K_eq = Param(initialize=exp(Ka + Kb/m.temp))

        ### Physical adsorption isotherm

        # Eq 14 in Aug. 2018 WVU report
        # LHS: bar-1
        # RHS: bar-1 * (kJ/mol)/(kJ/mol/K * K)
        m.b_b = Param(initialize = b_b0*exp(Q_stb/(RPV*T0)*(T0/m.temp-1)))

        # Eq 16 in Aug. 2018 WVU report
        # LHS: mol/kg
        # RHS: mol/kg * 1
        m.nmax_p = Param(initialize=nmax_p1*(exp(Kc+Kd/m.temp)/(1+exp(Kc+Kd/m.temp))))

        # Eq 12 in Aug. 2018 WVU report
        # LHS: mol/kg
        # RHS: mol/kg * 1 / 1 = mol/kg
        m.nmax_c = Param(initialize=n_max*m.K_eq/(1+m.K_eq))

        # Linear pressure adsorption when P < Plinear 
        # According to ACM
        # LHS: mol/kg 
        # RHS: mol/kg * ((1/bar * bar )/ (bar^-1 * 10bar))
        nplin_num = (m.b_a*plin)**(inv_n_a)
        nplin_de = 1+(m.b_a*plin)**(inv_n_a)
        m.nplin = Param(initialize=m.nmax_c*(nplin_num/nplin_de))
                 
    return m


    
def bounds_velocity(m, z, t):
    ''' Add bounds to velocity [cm/s]
    '''
    if t == 0.0:
        #return (1E-4, 0.3)  # bounds for 313K, 0.15 
        return (1E-2, 30)  # bounds for DoE problem
    else:
        #return (0.8,1.4)  # bounds for 313K, 0.15 
        return (0.01, 5.0)  # bounds for DoE problem
    
def bounds_velocity_pert(m, j, z, t):
    ''' Add bounds to velocity [cm/s]
    '''
    # Change to cm/s
    if t == 0.0:
        #return (1E-4, 0.3)  # bounds for 313K, 0.15 
        return (1E-2, 30) # bounds for DoE problem
    else:
        #return (0.8,1.4)  # bounds for 313K, 0.15 
        return (0.01, 5.0) # bounds for DoE problem

def breakthrough_bounds(z,t):
    ''' Check if position and time coordinates in region with
    no breakthrough.
    
        We draw a line between (t = 50% breakthrough, end of bed) and
        (t=0, z = 25% bed length). Anything "above" this line we designate as in the 
        region of no breakthrough and return True. We impose tighter bounds if true.
    '''

    t50 = 1500 # 50% breakthrough time [s]
    t0_intercept = 0.4

    # return (t0_intercept - 1)/t50*value(t) + t0_intercept > value(z)/Ngrid
    return False

    
def den_bounds(m,c,t,z):
    '''[mol/m3]
    '''
    if breakthrough_bounds(z,t):
        if c == 'N2':
            return (0.9*tden_low, tden_high)
        elif c == 'CO2':
            return (small_bound, 0.1*tden_high)
        else:
            return (small_bound, tden_high)
    else:
        return (small_bound, tden_high)
    
def den_bounds_pert(m,j,c,t,z):
    if breakthrough_bounds(z,t):
        if c == 'N2':
            return (0.9*tden_low, tden_high)
        elif c == 'CO2':
            return (small_bound, 0.1*tden_high)
        else:
            return (small_bound, tden_high)
    else:
        return (small_bound, tden_high)

def surface_partial_pressure_bounds(m,c,z,t):
    ''' [bar]
    '''
    
    if breakthrough_bounds(z,t):
        return (1E-5, 0.5)
    else:
        return (1E-8, P_high)
    
def surface_partial_pressure_bounds_pert(m,j,c,z,t):
    ''' [bar]
    '''
    if breakthrough_bounds(z,t):
        return (1E-5, 0.5)
    else:
        return (small_bound, P_high)

def phys_star_bounds(m,c,z,t):
    '''[mmol/g]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.1)
    else:
        #return (0.0, 0.8) # bounds for 313K, 0.15
        return (small_bound, 10.0)
    
def phys_star_bounds_pert(m,j,c,z,t):
    '''[mmol/g]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.1)
    else:
        #return (0.0, 0.8)  # bounds for 313K, 0.15
        return (small_bound, 10.0)

def chem_star_bounds(m,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.3)
    else:
        #return (0.0, 2.6)  # bounds for 313K, 0.15
        return (small_bound, 10.0)
    
def chem_star_bounds_pert(m,j,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.3)
    else:
        #return (0.0, 2.6)   # bounds for 313K, 0.15
        return (small_bound, 10.0)

def phys_loading_bounds(m,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.1)
    else:
        #return (0.0, 0.8)   # bounds for 313K, 0.15
        return (small_bound, 10.0)

def phys_loading_bounds_pert(m,j,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 0.1)
    else:
        #return (0.0, 0.8)   # bounds for 313K, 0.15
        return (small_bound, 10.0)

    
def chem_loading_bounds(m,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 2.0)
    else:
        #return (0.0, 2.6)   # bounds for 313K, 0.15
        return (small_bound, 10.0)
    
def chem_loading_bounds_pert(m,j,c,z,t):
    '''[mol/kg]'''
    
    if breakthrough_bounds(z,t):
        return (small_bound, 2.0)
    else:
        #return (0.0, 2.6)  # bounds for 313K, 0.15
        return (small_bound, 10.0)
        
def jac_bounds(m, t):
    '''very wide bounds based on units 
    '''
    return (-1, 1)

def add_variables(m,tf=3600, timesteps=None, start=0):
    '''
    Add variables to the Pyomo model using the toggles previously specified.
    
    Arguments: 
        m: the model 
        tf: the time scale [s]
        timesteps: time of the process, [s].
             single number: final time
             array: time points (non-uniform discretization)
        
    Return: None 
    '''
    
    # Store option
    m.dynamic = dynamic
    
    # Time [s]
    if timesteps is None:
        m.t = ContinuousSet(bounds=(0,tf))
        m.tf = tf
    else:
        m.t = ContinuousSet(bounds=(start,max(timesteps)), initialize=timesteps)
        m.tf = max(timesteps)
        m.t0 = min(timesteps)

          
    # Gas phase density (concentration) [mol/m^3]
    m.C = Var(m.scena, m.COMPS, m.zgrid, m.t, bounds=den_bounds_pert)

    # Gas phase density derivative, [mol/m^3] / [s]
    m.dCdt = DerivativeVar(m.C, wrt=m.t)

    # Total gas phase density [mol/m3]
    m.total_den = Var(m.scena, m.zgrid, m.t, bounds=(tden_low, tden_high))
        
    # Gas phase velocity [cm/s]
    m.v = Var(m.scena, m.zgrid, m.t, initialize=0.1, bounds=bounds_velocity_pert)
    if m.v_fix:
        m.v.fix()
        
    # Gas pressure [bar]
    #m.kfilm = Var(m.perturb, m.zgrid, m.t, initialize=1.92E-4, bounds=(0,1))
    
    # Gas pressure [bar]
    m.P = Var(m.scena, m.zgrid, m.t,initialize=1, bounds=(P_low, P_high))
    if m.fix_pres: 
        m.P.fix()
        
    # Temperature [K]
    if m.energy:
        # temperature is initialized to be the T_inlet. As this is the gas temperature, it should be started to be the inlet gas temperature. 
        #m.temp = Var(m.perturb, m.zgrid, m.t, initialize=value(m.temp_orig), bounds=(273.15,500.15), within=NonNegativeReals)
        m.temp = Var(m.scena, m.zgrid, m.t, initialize=m.temp_bath, bounds=(273.15,500.15), within=NonNegativeReals)
        #m.temp = Var(m.perturb, m.zgrid, m.t, initialize=m.temp_water,  bounds=(273.15,1000), within=NonNegativeReals)
        m.dTdt = DerivativeVar(m.temp, wrt=m.t)
        
        
        if not m.est_ua:
            # W/m3/K, value 0.2839 from [Dowling, 2012]
            # Estimated value 1.4E7 W/m3/K (DOE run2)
            # 1.4E4 kW/m3/K --> 1.4E1 W/cm3/K
            if m.doe_model:
                m.ua =  Param(m.scena, initialize=m.scena_all['ua'], mutable=True)
            elif m.k_aug:
                m.ua = Var(m.scena, initialize=m.ua_init)
                #m.ua.setlb(m.ua_init)
                #m.ua.setub(m.ua_init)
            else:
                m.ua = Expression(m.scena, rule=perturbations_ua)
        else:
            m.ua = Var(initialize=m.ua_init,bounds=(5, 12))
            #m.ua = Var(initialize=m.ua_init, bounds=(10,20))
            
        # define inv_k_oc, inv_k_op according to perturbation
        def inv_k_oc_init_en(m, j, z, t):
            '''
            Calculating 1/k_oc in [1/s]
            '''
            return m.fitted_transport_coefficient[j]+1/(K_c0*exp(-E_c/(RPV*m.temp[j,z,t]) + E_c/(RPV*T0)))

        def inv_k_op_init_en(m, j, z, t):
            '''
            Calculating 1/k_op in [1/s]
            '''
            return m.fitted_transport_coefficient[j]+1/(K_p0*exp(-E_p/(RPV*m.temp[j,z,t]) + E_p/(RPV*T0)))

        # For square problem/optimization problem, parameter, k_oc/p are parameters
        if (not m.est_tr) and (not m.k_aug):
            m.inv_K_oc = Expression(m.scena, m.zgrid, m.t, rule=inv_k_oc_init_en)
            m.inv_K_op = Expression(m.scena, m.zgrid, m.t, rule=inv_k_op_init_en)
            
        # If not square problem, k_oc/p are defined as parameters variable with temperature, defined in add_model()
          
        # J/mol/K
        def cpg_rule(m,j,i,z,t):
            '''
            Give Cpg in [J/mol/K]
            '''
            trans_t = m.temp[j,z,t]*0.001
            if i=='N2':
                return 29 + 1.85*trans_t - 9.65*trans_t**2 + 16.64*trans_t**3 + 0.000117/trans_t/trans_t
            elif i=='CO2':
                return 25 + 55.19*trans_t - 33.69*trans_t**2 + 7.95*trans_t**3 -0.1366/trans_t/trans_t
        m.cpg = Expression(m.scena, m.COMPS, m.zgrid, m.t, rule=cpg_rule)
        
        
        def h_rule(m,j,i,z,t):
            '''
            Calculate the flow heat based on the bed temperature. The reference temperature is 298.15K
            h in [J/mol]
            '''
            if i=='N2':
                return 29*(m.temp[j,z,t] - temp_base) + 1.85E-3/2*(m.temp[j,z,t]**2-temp_base**2) - 9.65E-6/3*(m.temp[j,z,t]**3-temp_base**3) + 16.64E-9/4*(m.temp[j,z,t]**4 - temp_base**4) -117/m.temp[j,z,t] + 117/temp_base
            elif i=='CO2':
                return 25*(m.temp[j,z,t] - temp_base) + 55.19E-3/2*(m.temp[j,z,t]**2-temp_base**2) - 33.69E-6/3*(m.temp[j,z,t]**3 - temp_base**3) + 7.95E-9/4*(m.temp[j,z,t]**4 - temp_base**4) +136638/m.temp[j,z,t] - 136638/temp_base
                
        m.h = Expression(m.scena, m.COMPS, m.zgrid, m.t, rule=h_rule)
        
        # Feed heat at the feed temperature [W/m2/K]
        def h_feed_rule(m,i):
            '''
            Calculating feed heat based on the inlet temperature. The reference temperature is 298.15K
            h in [J/mol]
            '''
            if i=='N2':
                return 29*(m.temp_orig-temp_base) + 1.85E-3/2*(m.temp_orig**2-temp_base**2) - 9.65E-6/3*(m.temp_orig**3-temp_base**3) + 16.64E-9/4*(m.temp_orig**4-temp_base**4) -117/m.temp_orig + 117/temp_base
            elif i=='CO2':
                return 25*(m.temp_orig-temp_base) + 55.19E-3/2*(m.temp_orig**2 - temp_base**2) - 33.69E-6/3*(m.temp_orig**3 - temp_base**3) + 7.95E-9/4*(m.temp_orig**4 - temp_base**4) +136638/m.temp_orig - 136638/temp_base

        m.h_feed = Expression(m.COMPS, rule=h_feed_rule)
        
        # Adsorption heat, -65 kJ/mol
        m.H_ads = Param(m.SCOMPS, initialize={'CO2': -65.0})
        
    
    if m.isotherm:
        
        # Surface partial pressure, [bar]
        m.spp = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize=small_initial, bounds=surface_partial_pressure_bounds_pert)
        
        # Chemical adsorption equilibrium, [mol / kg]
        m.nchemstar = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=chem_star_bounds_pert)
        
        # Chemical adsorption equilibrium, modified [mol / kg]
        m.nchemstar_mod = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=chem_star_bounds_pert)
        
        # Physical adsorption equilibrium, [mol / kg]
        m.nphysstar = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=phys_star_bounds_pert)
        
        # Physical adsorption equilibrium with linear region [mol/kg]
        m.nphysstar_mod = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize =small_initial, bounds=phys_star_bounds_pert)
        
        # alpha used to form linearized pressure
        if alpha_variable:
            m.alpha = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=(small_bound,1.0))

    if m.chemsorb:
         
        # Chemical adsorption loading, [mol of gas/ kg of sorbent]
        m.nchem = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=chem_star_bounds_pert)

        # Time derivative, [mol/kg/s]
        m.dnchemdt = DerivativeVar(m.nchem, wrt=m.t)
        
    if m.physsorb:
        # Physical adsorption equilibrium loading, [mol of gas/kg of sorbent]
        m.nphys = Var(m.scena, m.SCOMPS, m.zgrid, m.t, initialize = small_initial, bounds=phys_star_bounds_pert)
        
        # Time derivative, [mol/kg/s]
        m.dnphysdt = DerivativeVar(m.nphys, wrt=m.t)
            
            
    m.dv = Set(initialize=['k', 'ua'])
        
    if m.energy:
        def b_a_en(m,j,z,t):
            return b_a0*exp(Q_sta/(RPV*T0)*(T0/m.temp[j,z,t]-1))
        # Eq 14 in Aug. 2018 WVU report
        # LHS: [1/bar]
        # RHS: [1/bar] * exp( [kJ/mol]/[kJ/mol/K * K] ) = [1/b] * exp( [ dimensionless ] )
        m.b_a = Expression(m.scena, m.zgrid, m.t, rule=b_a_en) 

        # Eq 15 in Aug. 2018 WVU report
        # LHS: dimensionless
        # RHS: dimensionless * exp( [kJ/mol]/[kJ/mol/K * K] ) = [dimensionless] * exp( [ dimensionless ])
        def n_a_en(m,j,z,t):
            return n_a1*exp(E_na/(RPV*T0)*(T0/m.temp[j,z,t]-1) + small_bound)
        m.n_a = Expression(m.scena, m.zgrid, m.t, rule=n_a_en)

        def inv_n_a_en(m,j,z,t):
            return 1/m.n_a[j,z,t]
        m.inv_n_a = Expression(m.scena, m.zgrid, m.t, rule=inv_n_a_en)

        # Eq 13 in Aug. 2018 WVU report
        # LHS: dimensionless
        # RHS: dimensionless + K/K
        def k_eq_en(m,j,z,t):
            return exp(Ka + Kb/m.temp[j,z,t] + small_bound)
        m.K_eq = Expression(m.scena, m.zgrid, m.t, rule=k_eq_en)

        ### Physical adsorption isotherm
        # Eq 14 in Aug. 2018 WVU report
        # LHS: bar-1
        # RHS: bar-1 * (kJ/mol)/(kJ/mol/K * K)
        def b_b_en(m,j,z,t):
            return b_b0*exp(Q_stb/(RPV*T0)*(T0/m.temp[j,z,t]-1) + small_bound)
        m.b_b = Expression(m.scena, m.zgrid, m.t, rule=b_b_en)


        # Eq 16 in Aug. 2018 WVU report
        # LHS: mol/kg
        # RHS: mol/kg * 1
        def nmax_p_en(m,j,z,t):
            return nmax_p1*(exp(Kc+Kd/m.temp[j,z,t] + small_bound)/(1+exp(Kc+Kd/m.temp[j,z,t] + small_bound)))
        m.nmax_p = Expression(m.scena, m.zgrid, m.t, rule=nmax_p_en)

        # Eq 12 in Aug. 2018 WVU report
        # LHS: mol/kg
        # RHS: mol/kg * 1 / 1 = mol/kg
        def nmax_c_en(m,j,z,t):
            return n_max*m.K_eq[j,z,t]/(1+m.K_eq[j,z,t])
        m.nmax_c = Expression(m.scena, m.zgrid, m.t, rule=nmax_c_en)

        # Linear pressure adsorption when P < Plinear 
        # According to ACM
        # LHS: mol/kg 
        # RHS: mol/kg * ((1/bar * bar )/ (bar^-1 * 10bar))
        def nplin_en(m,j,z,t):
            nplin_num = (m.b_a[j,z,t]*plin)**(m.inv_n_a[j,z,t])
            nplin_de = 1+(m.b_a[j,z,t]*plin)**(m.inv_n_a[j,z,t])
            return m.nmax_c[j,z,t]*(nplin_num/nplin_de)
        
        m.nplin = Expression(m.scena, m.zgrid, m.t, rule=nplin_en)
        
### DEFINE EQUATION FUNCTION

def gas_comp_mb(m, j, i, z, t):
    '''
    Calculate bulk gas phase species balances with Eq 1 in Aug. report, (Dowling, 2012) and ACM code.
    Used when diffusion is not considered. 
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: ODE for calculating component species density
    '''
    
    # Eq 1 in Aug.report
    # LHS: mol/m3/s
    # RHS: (cm/s * mol/m3 / (100cm/m) - cm/s * mol/m3 / (100cm/m))/m = mol/m3/s
    if z == 0:
        dvCdz = (m.v[j,z,t]*0.01*m.C[j,i,z,t] - vel_f*0.01*m.totden_f*m.yfeed[i])/dz
    else:
        dvCdz = (m.v[j,z,t]*0.01*m.C[j,i,z,t] - m.v[j,z-1,t]*0.01*m.C[j,i,z-1,t])/dz

    diff_term = 0
    # LDF from ACM code
    if i in m.SCOMPS:
        if m.chemsorb:
            diff_term += (1-ads_epsb)*(den_s)*m.dnchemdt[j, i, z, t]
        
        if m.physsorb:
            diff_term += (1-ads_epsb)*(den_s)*m.dnphysdt[j, i, z, t]
            
    disp_term = 0
    if m.dispersion:
        # LHS: mol/m3/s
        # RHS: m2/s * mol/m3/m2
        # combine the two gas mas balances, so it can be toggled on and off 
        if z == 0:
            disp_term = Dax[i] * (m.C[j,i,z+2,t]-2*m.C[j,i,z+1,t]+m.C[j,i,z,t])/(dz*dz)
        elif z == 19:
            disp_term = Dax[i] * (m.C[j,i,z,t]-2*m.C[j,i,z-1,t]+m.C[j,i,z-2,t])/(dz*dz)
        else: 
            disp_term = Dax[i] * (m.C[j,i,z+1,t]-2*m.C[j,i,z,t]+m.C[j,i,z-1,t])/(dz*dz)
    # LHS: mol/m3/s; # RHS: mol/m3/s - mol/m3/s
    return m.dCdt[j,i,z,t] == -dvCdz / (ads_epsb) - diff_term/ (ads_epsb) - disp_term 


def gas_comp_mb_doe(m, j, i, z, t):
    '''
    Calculate bulk gas phase species balances with Eq 1 in Aug. report, (Dowling, 2012) and ACM code.
    Used when diffusion is not considered. 
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: ODE for calculating component species density
    '''
    
    # Eq 1 in Aug.report
    # LHS: mol/m3/s
    # RHS: (cm/s * mol/m3 / (100cm/m) - cm/s * mol/m3 / (100cm/m))/m = mol/m3/s
    if z == 0:
        if i=='CO2':
            dvCdz = (m.v[j,z,t]*0.01*m.C[j,i,z,t] - vel_f*0.01*m.totden_f*m.yfeed)/dz
        elif i=='N2':
            dvCdz = (m.v[j,z,t]*0.01*m.C[j,i,z,t] - vel_f*0.01*m.totden_f*(1-m.yfeed))/dz
            
    else:
        dvCdz = (m.v[j,z,t]*0.01*m.C[j,i,z,t] - m.v[j,z-1,t]*0.01*m.C[j,i,z-1,t])/dz

    diff_term = 0
    # LDF from ACM code
    if i in m.SCOMPS:
        if m.chemsorb:
            diff_term += (1-ads_epsb)*(den_s)*m.dnchemdt[j, i, z, t]
        
        if m.physsorb:
            diff_term += (1-ads_epsb)*(den_s)*m.dnphysdt[j, i, z, t]
            
    disp_term = 0
    if m.dispersion:
        # LHS: mol/m3/s
        # RHS: m2/s * mol/m3/m2
        # combine the two gas mas balances, so it can be toggled on and off 
        if z == 0:
            disp_term = Dax[i] * (m.C[j,i,z+2,t]-2*m.C[j,i,z+1,t]+m.C[j,i,z,t])/(dz*dz)
        else:
            disp_term = Dax[i] * (m.C[j,i,z+1,t]-2*m.C[j,i,z,t]+m.C[j,i,z-1,t])/(dz*dz)
    # LHS: mol/m3/s; # RHS: mol/m3/s - mol/m3/s
    return m.dCdt[j,i,z,t] == -dvCdz / (ads_epsb) - diff_term/ (ads_epsb) - disp_term 


    
def energy_balance(m, j, z, t):
    '''
    Energy balance assuming Tambient = Twall, Tg=Ts
    Equation from [Dowling, 2012]
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t

    Return: ODE for calculating component species density
    
    '''
    # calculate sum(ci*cpg)
    # mol/m3 * J/mol/K = J/m3/K
    sum_c = sum(m.C[j,i,z,t]*m.cpg[j,i,z,t] for i in m.COMPS)
    
    # calculate the coefficient of dT/dt
    # J/m3/K + kg/m3 * J/kg/K = J/m3/K
    dividant = ads_epsb*sum_c + den_b*cps 
    
    # calculate the sum of Hi*dni/dt
    # kJ/mol * mol/kg/s = kJ/kg/s
    sum_hdn = sum(-m.H_ads[a]*(m.dnchemdt[j,a,z,t]+m.dnphysdt[j,a,z,t]) for a in m.SCOMPS)
    
    # sum of heat flow, [J/m3]
    # RHS: mol/m3 * J/mol = J/m3 
    h_sum = sum(m.C[j,b,z,t]*m.h[j,b,z,t] for b in m.COMPS)
    
    # heat flow feed, [J/m3]
    h_sum_feed = sum(m.den_f[p]*m.h_feed[p] for p in m.COMPS)
    
    # LHS: W/m3 = J/s/m3 
    # RHS: cm/s * 1m/100cm * J/m3 * 1/m = J/s/m3 
    if z==0: 
        duhdz = (m.v[j,z,t]*0.01*h_sum - vel_f*0.01*h_sum_feed)/dz
    else:
        # heat flow of the previous grid, [J/m3]
        h_sum_back = sum(m.C[j,b,z-1,t]*m.h[j,b,z-1,t] for b in m.COMPS)
        duhdz = (m.v[j,z,t]*0.01*h_sum - m.v[j,z-1,t]*0.01*h_sum_back)/dz
    
    # LHS: K/s
    # RHS: (kg/m3 * kJ/kg/s * 1000J/1kJ - J/m3/s - J/s/m3/K *K)/ (J/m3/K) = K/s 
    #if m.temp_option==2:
    #    return m.dTdt[j,z,t] == (den_bed*sum_hdn*1000 - duhdz - exp(m.ua[j])*(m.temp[j,z,t]-m.temp_bath))/dividant
    #elif m.temp_option==1:
    if m.est_ua or m.k_aug:
        return m.dTdt[j,z,t] == (den_b*sum_hdn*1000 - duhdz - exp(m.ua)*(m.temp[j,z,t]-m.temp_bath))/dividant
    else:
        return m.dTdt[j,z,t] == (den_b*sum_hdn*1000 - duhdz - exp(m.ua[j])*(m.temp[j,z,t]-m.temp_bath))/dividant
    
def dalton(m, j, z, t):
    '''
    Calculate total gas density from component gas densities. 
    No corresponding equation in Aug.report or ACM.
    
    Arguments: 
        m: model
        j: model.perturb 
        z: model.zgrid
        t: model.t
        
    Return: Total gas density
    '''
    # LHS & RHS: [mol/m3]
    return m.total_den[j,z,t] == sum(m.C[j,i,z,t] for i in m.COMPS)


def ideal(m,j, z, t):
    '''
    Calculate total pressure from ideal gas law.
    
    Arguments: 
        m: model
        j: model.perturb 
        z: model.zgrid
        t: model.t
        
    Return: total gas pressure in the bulk phase
    '''
    # LHS: mol/m3 * kJ/mol/K * K * 0.01 bar/kPa = kJ/m3 *100 kPa/bar = [bar]
    # RHS: [bar]
    if m.energy:
        return m.total_den[j,z,t]*0.01 * RPV * m.temp[j,z,t]  == m.P[j,z,t]
    else:
        return m.total_den[j,z,t]*0.01 * RPV * m.temp  == m.P[j,z,t]

def ergun(m, j, z, t):
    '''
    Calculate velocity with Eq 6 in Aug. report.
    Modified according to Ergun equation in (Dowling, 2012). 
    An assumption was made: velocity is always positive
    
    Arguments: 
        m: model
        j: model.perturb 
        z: model.zgrid
        t: model.t
          
    Return: an ODE for velocity 
    '''
    # LHS: kg/m4
    # RHS: mol/m3*g/mol / m /(1000g/kg) = g/m4 / (1000g/kg) = kg/m4 
    
    mass_den=sum(m.C[j,i,z,t]*MW[i] for i in m.COMPS)
    
    aeff = 1.75*mass_den*(1-ads_epsb)/(2*radp*ads_epsb*ads_epsb*ads_epsb)/1000

    # LHS: kg/m3/s
    # RHS: [Pa*s]/[m^2] = [kg/m/s]/[m^2] = kg / m^3 / s
    beff = 150*fp_mu*(1-ads_epsb)*(1-ads_epsb)/(4*radp*radp*ads_epsb*ads_epsb*ads_epsb)

    if z == 0:
        dPdz = (m.P[j,z,t] - totp_f) / dz
    else:
        dPdz = (m.P[j,z,t] - m.P[j,z-1,t]) / dz
    
    # Assumption: velocity is always positive
    # LHS: [bar/m] 
    # RHS: [kg/m^4] * ([cm/s]*0.01m/cm)^2 + [kg / m^3 / s] * [cm/s*0.01m/cm] = [kg / m^2 / s^2] = [Pa/m] / (1E5Pa/bar)
    return -dPdz == (aeff*(m.v[j,z,t]*0.01)**2 + beff*m.v[j,z,t]*0.01)/1.0E5

'''
def kf_calc(m,j,z,t):
    
    Calculate the k_f with the dimensionless number equations.
    Not used for now.
    
    # RHS: m*m/s*kg/mol*mol/m3*m*s/kg = 1
    Re = 2*radp*m.v[j,z,t]*0.01*M_co2*m.C[j,'CO2',z,t]/mu_co2
    # RHS: kg/m/s * s/m2 * m3/mol * mol/kg = 1
    Sc = mu_co2/D_m/m.C[j,'CO2',z,t]/M_co2
    return  2*radp*m.kfilm[j,z,t]/D_m == 2+1.1*Re**(0.6)*Sc**(1/3)    
'''

def calc_surface_pressure(m, j, i, z, t):
    '''
    Calculate surface partial pressure in solid phase (particle void) according to ACM code. 
    For now, surface partial pressure is not bulk gas pressure.
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: surface partial pressure
    '''
    if m.energy:
        return m.spp[j,i,z,t] == RPV*m.temp[j,z,t]*0.01*m.C[j,i,z,t]
    else:
        return m.spp[j,i,z,t] == RPV*m.temp*0.01*m.C[j,i,z,t]

def chem_isotherm(m, j, i, z, t):
    '''
    Calculate chemical equilibrium adsorption with Eq 11, WVU Aug.report. 
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: chemical equilibrium adsorption
    '''
    # Eq.11 in Aug.report
    if m.energy:
        nchem_num = (m.b_a[j,z,t]*m.spp[j,i,z,t])**(m.inv_n_a[j,z,t])
        nchem_de = 1+(m.b_a[j,z,t]*m.spp[j,i,z,t])**(m.inv_n_a[j,z,t])
        # LHS: unit of m.nchemstar = unit of nmax_c = mmol/g = mol/kg
        return m.nchemstar[j,i,z,t]*nchem_de == m.nmax_c[j,z,t]*nchem_num
    else:
        inv_n_a = 1/m.n_a
        nchem_num = (m.b_a*m.spp[j,i,z,t])**(inv_n_a)
        nchem_de = 1+(m.b_a*m.spp[j,i,z,t])**(inv_n_a)
        # LHS: unit of m.nchemstar = unit of nmax_c = mmol/g = mol/kg
        return m.nchemstar[j,i,z,t]*nchem_de == m.nmax_c*nchem_num

def chem_isotherm_mod(m, j, i, z, t):
    '''
    Calculate MODIFIED chemical equilibrium adsorption
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: chemical equilibrium adsorption
    '''
    # RHS: [mol/kg] * [bar]/[bar] + [mol/kg]
    # LHS: [mol/kg]
    if m.energy:
        return m.nchemstar_mod[j,i,z,t] == m.alpha[j,i,z,t] * m.nplin[j,z,t] *m.spp[j,i,z,t]/plin + (1-m.alpha[j,i,z,t])*m.nchemstar[j,i,z,t]
    else: 
        return m.nchemstar_mod[j,i,z,t] == m.alpha[j,i,z,t] * m.nplin*m.spp[j,i,z,t]/plin + (1-m.alpha[j,i,z,t])*m.nchemstar[j,i,z,t]


def phys_isotherm(m, j, i, z, t):
    '''
    Calculate physical equilibrium adsorption with Eq 11, WVU Aug.report. 
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: physical equilibrium adsorption
    '''
    n_b = 1.46
    inv_n_b = 1/n_b
    if m.energy:
        nphys_num = (m.b_b[j,z,t]*m.spp[j,i,z,t])**(inv_n_b)
        nphys_de = 1+(m.b_b[j,z,t]*m.spp[j,i,z,t])**(inv_n_b)
        return m.nphysstar[j, i, z, t]*nphys_de == m.nmax_p[j,z,t]*nphys_num
    else:
        nphys_num = (m.b_b*m.spp[j,i,z,t])**(inv_n_b)
        nphys_de = 1+(m.b_b*m.spp[j,i,z,t])**(inv_n_b)
        # LHS: unit of m.nchemstar = unit of nmax_c = mmol/g = mol/kg
        return m.nphysstar[j, i, z, t]*nphys_de == m.nmax_p*nphys_num
    

def phys_isotherm_mod(m, i, z, t):
    '''
    Calculate MODIFIED chemical equilibrium adsorption
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: chemical equilibrium adsorption
    '''
    # RHS: [mol/kg] * [bar]/[bar] + [mol/kg]
    # LHS: [mol/kg]
    if m.energy:
        return m.nphysstar_mod[j,i,z,t] == m.alpha[j,i,z,t] * m.nplin[j,z,t]*m.spp[j,i,z,t]/plin + (1-m.alpha[j,i,z,t])*m.nphysstar[j,i,z,t]
    else:
        return m.nphysstar_mod[j,i,z,t] == m.alpha[j,i,z,t] * m.nplin*m.spp[j,i,z,t]/plin + (1-m.alpha[j,i,z,t])*m.nphysstar[j,i,z,t]


def alpha_calc(m, j, i, z, t):
    '''
    calculate alpha 
    ''' 
    # Pressure difference
    
    # [??-bar] = alpha_scale [??bar / bar] * ([bar] - [MPa]*10bar/MPa)
    
    # LHS: [bar]
    # RHS: [bar - bar]
    pres_dif = alpha_scale*(m.spp[j, i, z, t] - plin)

    # alpha is a variable and will be calculated with a constraint
    if alpha_variable:
    
        if alpha_option == 1:
            return (2*m.alpha[j,i,z,t] - 1)*sqrt(eps_alpha + pres_dif*pres_dif) == -pres_dif
        elif alpha_option == 2:
            return (1 - m.alpha[j,i,z,t])*(1 + exp(-pres_dif + small_bound)) == 1
        elif alpha_option == 3: 
            return m.alpha[j,i,z,t] == 0.0
        elif alpha_option == 4: 
            return m.alpha[j,i,z,t] == 1.0
        else:
            return Constraint.Skip

    # alpha is an expression
    else:
        if alpha_option == 1:
            return 0.5*(1-pres_dif/sqrt(eps_alpha+pres_dif*pres_dif))
        elif alpha_option == 2:
            return 1 - 1 /(1 + exp(-pres_dif + small_bound))
        elif alpha_option == 3: 
            return 0.0
        elif alpha_option == 4: 
            return 1.0
        else:
            return Constraint.Skip


def chem_adsorb(m, j, i, z, t):
    '''
    Calculate chemical adsorption kinetics according to ACM code. 
    Include a linear pressure part when P<0.3bar.
    
    Arguments: 
        m: model
        j: model.perturb
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t
        
    Return: an ODE for chemical adsorption amount [kmol/kg sorbent]
    '''
    # For parameter estimation, no perturbation index for inv_K_oc
    if m.est_tr or m.k_aug:
        if m.energy:
            return m.dnchemdt[j,i,z,t] == (1/m.inv_K_oc[j,z,t])*(m.nchemstar_mod[j,i,z,t] - m.nchem[j,i,z,t])
        else:
            return m.dnchemdt[j,i,z,t] == (1/m.inv_K_oc[j,z,t])*(m.nchemstar_mod[j,i,z,t] - m.nchem[j,i,z,t])
    # For DoE, inv_K_oc is indexed with perturbation 
    else:
        if m.energy:
            return m.dnchemdt[j,i,z,t] == (1/m.inv_K_oc[j,z,t])*(m.nchemstar_mod[j,i,z,t] - m.nchem[j,i,z,t])
        else:
            return m.dnchemdt[j,i,z,t] == (1/m.inv_K_oc[j])*(m.nchemstar_mod[j,i,z,t] - m.nchem[j,i,z,t])

def phys_adsorb_nonlinear(m, j, i, z, t):
    '''
    Calculate chemical adsorption kinetics without linear pressure part 
    '''
    if m.est_tr or m.k_aug:
        #if m.energy:
        #    return m.dnphysdt[j,i,z,t] == (1/m.inv_K_op[j,z,t])*(m.nphysstar[j,i,z,t] - m.nphys[j,i,z,t])
        #else:
        return m.dnphysdt[j,i,z,t] == (1/m.inv_K_op[j,z,t])*(m.nphysstar[j,i,z,t] - m.nphys[j,i,z,t])
    else:
        if m.energy:
            return m.dnphysdt[j,i,z,t] == (1/m.inv_K_op[j,z,t])*(m.nphysstar[j,i,z,t] - m.nphys[j,i,z,t])
        else:
            return m.dnphysdt[j,i,z,t] == (1/m.inv_K_op[j])*(m.nphysstar[j,i,z,t] - m.nphys[j,i,z,t])

def kinetics_para_express(m,j,z,t):
    '''
    Calculate inv_K_oc with Eq 8, WVU Aug.report.
    Used when transport_coefficient is estimated
    
    Arguments: 
        m: model
        i: model.SCOMPS
        
    Return: expression for inv_K_oc[i]
    '''
    # LHS: s; # RHS: (m*m/(1*m*m/s)) + 1/(1/s * 1) 
    if m.energy:
        return m.fitted_transport_coefficient + 1/(K_c0*exp(-E_c/(RPV*m.temp[j,z,t]) + E_c/(RPV*T0) + small_bound))
    else:
        return m.fitted_transport_coefficient + 1/(K_c0*exp(-E_c/(RPV*m.temp) + E_c/(RPV*T0) + small_bound))


def kinetics_para2_express(m,j,z,t):
    '''
    Calculate inv_K_op with Eq 8, WVU Aug.report.
    Used when estimating transport coefficient 
    
    Arguments: 
        m: model
        i: model.SCOMPS
        
    Return: expression for inv_K_op[i]
    '''
    # LHS: s; # RHS: (m*m/(1*m*m/s)) + 1/(1/s * 1)
    if m.energy:
        return m.fitted_transport_coefficient + 1/(K_p0*exp(-E_p/(RPV*m.temp[j,z,t]) + E_p/(RPV*T0)+ small_bound))
    else:
        return m.fitted_transport_coefficient + 1/(K_p0*exp(-E_p/(RPV*m.temp) + E_p/(RPV*T0)+ small_bound))

# add an expression to calculate the Jacobian elements 
def FCO2_calc(m,j,z,t):
    '''Calculate FCO2 in mmol/min
    '''
    return m.C[j,'CO2',z,t]*m.v[j,z,t]*3.1415926*rbed*rbed*600

def jac_ele_exp(m,t):
    '''[Out-of-dated]Calculate the Jacobian elements when only one parameter of k 
    '''
    p = []
    for posi in m.zgrid:
        p.append(posi)
        
    if m.diff ==1:
        return (m.FCO2['forward_k',p[-1],t] - m.FCO2['base',p[-1],t])/m.eps/m.para
    elif m.diff == -1:
        return (m.FCO2['base',p[-1],t] - m.FCO2['backward_k',p[-1],t])/m.eps/m.para
    elif m.diff == 2:
        return (m.FCO2['forward_k',p[-1],t] - m.FCO2['backward_k',p[-1], t])/2/m.eps/m.para
        
def jac_ele(m,dv,t):
    '''
    Calculate the Jacobian elements 
    Arguments:
        dv: the design variables, k for k_trans, ua for UA
        t: timepoints 
    Return: 
        jac['k',t] is the Jacobian of k_trans, jac['ua', t] is the Jacobian of UA
    '''
    p = []
    for posi in m.zgrid:
        p.append(posi)
    
    if m.diff ==1:
        if dv=='k':
            return (m.FCO2['forward_k', p[-1], t] - m.FCO2['base',p[-1],t])/m.eps/m.para
        elif dv=='ua':
            return (m.FCO2['forward_ua',p[-1],t] - m.FCO2['base',p[-1],t])/m.eps/m.ua_init
    elif m.diff==-1:
        if dv=='k':
            return (m.FCO2['base',p[-1], t] - m.FCO2['backward_k',p[-1],t])/m.eps/m.para
        elif dv=='ua':
            return (m.FCO2['base',p[-1],t] - m.FCO2['backward_ua',p[-1],t])/m.eps/m.ua_init
    elif m.diff==2:
        if dv=='k':
            return (m.FCO2['forward_k', p[-1],t] - m.FCO2['backward_k',p[-1],t])/m.eps/m.para/2
        elif dv=='ua':
            return (m.FCO2['forward_ua',p[-1],t] - m.FCO2['backward_ua',p[-1],t])/m.eps/m.ua_init/2

def jac_Tend_ele(m,dv,t):
    '''
    Calculate the Jacobian elements 
    Arguments:
        dv: the design variables, k for k_trans, ua for UA
        t: timepoints 
    Return: 
        jac['k',t] is the Jacobian of k_trans, jac['ua', t] is the Jacobian of UA
    '''
    p = []
    for posi in m.zgrid:
        p.append(posi)
    
    if m.diff ==1:
        if dv=='k':
            return (m.temp['forward_k', p[-1], t] - m.temp['base',p[-1],t])/m.eps/m.para
        elif dv=='ua':
            return (m.temp['forward_ua',p[-1],t] - m.temp['base',p[-1],t])/m.eps/m.ua_init
    elif m.diff==-1:
        if dv=='k':
            return (m.temp['base',p[-1], t] - m.temp['backward_k',p[-1],t])/m.eps/m.para
        elif dv=='ua':
            return (m.temp['base',p[-1],t] - m.temp['backward_ua',p[-1],t])/m.eps/m.ua_init
    elif m.diff==2:
        if dv=='k':
            return (m.temp['forward_k', p[-1],t] - m.temp['backward_k',p[-1],t])/m.eps/m.para/2
        elif dv=='ua':
            return (m.temp['forward_ua',p[-1],t] - m.temp['backward_ua',p[-1],t])/m.eps/m.ua_init/2

        
def feed_rule(m):
    '''The inlet mole fraction should add up to 1. Applied for continuous optimization problem 
    '''
    return sum(m.yfeed[i] for i in m.COMPS) == 1

    
def ObjRule_opt_k(m):
    '''objective function for the continuous DoE problem
    '''
    objfunc = -sum(m.jac['k', t]**2 for t in m.t)
    return objfunc    

def ObjRule_A(m):

    fim00 = sum(m.jac['k',t]**2 for t in m.t) + sum(m.jac_tend['k',t]**2 for t in m.t)
    #fim01 = sum(m.jac['k',p[-1],t]*m.jac['ua',t] for t in m.t)
    fim11 = sum(m.jac['ua',t]**2 for t in m.t) +sum(m.jac_tend['ua',t]**2 for t in m.t)
    # use -trace for minimize, to maximize trace
    trace = -fim00 - fim11
    return trace
     
def ObjRule_D(m):
    fim00 = sum(m.jac['k',t]**2 for t in m.t)+ sum(m.jac_tend['k',t]**2 for t in m.t)
    fim01 = sum(m.jac['k',t]*m.jac['ua',t] for t in m.t) + sum(m.jac_tend['k',t]*m.jac_tend['ua',t] for t in m.t)
    fim11 = sum(m.jac['ua',t]**2 for t in m.t)+sum(m.jac_tend['ua',t]**2 for t in m.t)
    # use -det for minimize, to maximize det
    det = fim01**2 - fim00*fim11
    return det
    
def ObjRule_con(m):
    '''objective function for the brute force DoE problem/square problem
    '''
    return 0 
    
'''
Adsorption isotherm calculation
Note: With constant temperature, many of the isotherm intermediates are CONSTANT
'''

def add_equations(mod):
    ''' 
    Adds equations to the Pyomo model using the already specified toggles.
    
    Arguments:
        mod: the model defined
    
    Return: the model with equations added
    
    '''
    
        
    # If not, the objective function will be added in the notebook 
        
    mod.dalton_law = Constraint(mod.scena, mod.zgrid, mod.t, rule=dalton)

    mod.ideal_gas_law = Constraint(mod.scena, mod.zgrid, mod.t, rule=ideal)
    
    # measurements
    if not mod.k_aug:
        mod.FCO2 = Expression(mod.scena, mod.zgrid, mod.t, rule=FCO2_calc)
    
    
    if mod.opt and mod.conti:
        
        mod.jac = Expression(mod.dv, mod.t, rule=jac_ele)
       
        
        # add an expression to calculate the Jacobian elements
        if mod.energy:
            mod.jac_tend = Expression(mod.dv, mod.t, rule=jac_Tend_ele)
            if mod.optimize_trace:
                mod.obj = Objective(rule=ObjRule_A, sense=minimize)
            else:
                mod.obj = Objective(rule=ObjRule_D, sense=minimize)
        else:
            mod.obj = Objective(rule=ObjRule_opt_k, sense=minimize)
        
        
        mod.feed_law = Constraint(rule=feed_rule)

        
    if not mod.fix_pres:
        if not mod.v_fix:
            mod.ergun_equation = Constraint(mod.perturb, mod.zgrid, mod.t, rule=ergun)

    if mod.doe_model:
        mod.gas_mass_balance = Constraint(mod.perturb, mod.COMPS, mod.zgrid, mod.t, rule = gas_comp_mb_doe)
    else:
        mod.gas_mass_balance = Constraint(mod.perturb, mod.COMPS, mod.zgrid, mod.t, rule = gas_comp_mb)

    if mod.energy:
        mod.energy_balance_law = Constraint(mod.perturb, mod.zgrid, mod.t, rule = energy_balance)
        
    mod.partial_pressure = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=calc_surface_pressure)
    mod.chemical_isotherm = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=chem_isotherm)
    mod.physical_isotherm = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=phys_isotherm)

    if alpha_variable:
        mod.alpha_constraint = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=alpha_calc)
    else:
        mod.alpha = Expression(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=alpha_calc)

    if mod.est_tr or mod.k_aug:
        mod.inv_K_oc = Expression(mod.perturb, mod.zgrid, mod.t, rule=kinetics_para_express)
        mod.inv_K_op = Expression(mod.perturb, mod.zgrid, mod.t, rule=kinetics_para2_express)

    mod.physical_adsorption = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=phys_adsorb_nonlinear) 

    mod.chemical_isotherm_mod = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=chem_isotherm_mod)
    mod.chemical_adsorption = Constraint(mod.perturb, mod.SCOMPS, mod.zgrid, mod.t, rule=chem_adsorb)
            

            
def extract_fixed(m, measure_temp=False):
    measurement=[]
    for t in m.t:
        measurement.append(value(m.FCO2['base', 19,t]))
    if measure_temp:
        for t in m.t:
            measurement.append(value(m.temp['base', 10, t]))
        for t in m.t:
            measurement.append(value(m.temp['base', 19, t]))    
    return measurement
            
def fix_initial_bed(m, v_init=2.0):
    '''
    Initialize the bed. 
    Make component density and phys/chem adsorption all over the bed at time0 to be 0.0. 
    '''
    for j in m.perturb:
        for z in m.zgrid:
            m.C[j,'CO2',z, m.t0].fix(small_initial)
            m.v[j,z,m.t0].fix(v_init)
            
            if m.energy:
                m.temp[j,z,m.t0].fix(value(m.temp_bath))

            if m.chemsorb:
                m.nchem[j,'CO2',z,m.t0].fix(small_initial)

            if m.physsorb:
                m.nphys[j,'CO2',z,m.t0].fix(small_initial)
        

def extract2d(m, var):
    ''' 
    Extract values for 2D variable
    
    Args:
        m - Pyomo model
        var - Pyomo Variable
        
    Returns:
        D - 2D numpy array with values
        
    Assumptions:
        * First variable index is grid position
        * Second variable index is time position
    '''
    # Extract length of two dimensions
    Ngrid = len(m.zgrid)
    NFEt = len(m.t) - 1

    # Create numpy array of zeros 
    D1 = np.zeros((Ngrid,NFEt+1))
    D2 = np.zeros((Ngrid,NFEt+1))
    D3 = np.zeros((Ngrid,NFEt+1))
    D4 = np.zeros((Ngrid,NFEt+1))
    
    # Loop over time and space, extract values from Pyomo variable
    for i,t in enumerate(m.t):
        for j,z in enumerate(m.zgrid):
            if m.diff == 0:
                D1[j,i] = value(var['base',z,t])
            elif m.diff == 1:
                D1[j,i] = value(var['base',z,t])
                D2[j,i] = value(var['forward_k',z,t])
                D3[j,i] = value(var['forward_ua',z,t])
            elif m.diff == -1:
                D1[j,i] = value(var['base',z,t])
                D2[j,i] = value(var['backward_k',z,t])
                D3[j,i] = value(var['backward_ua',z,t])
            elif m.diff == 2: 
                D1[j,i] = value(var['forward_k',z,t])
                D2[j,i] = value(var['forward_ua',z,t])
                D3[j,i] = value(var['backward_k',z,t])
                D4[j,i] = value(var['backward_ua',z,t])
    
    return D1,D2,D3,D4

def extract3d(m, var, ind):
    ''' 
    Extract values for 3D variable
    
    Args:
        m - Pyomo model
        var - Pyomo Variable
        ind - Index for first dimension
        
    Returns:
        D - 2D numpy array with values
        
    Assumptions:
        * First variable index is grid position
        * Second variable index is time position
    '''
    # Extract length of two dimensions
    Ngrid = len(m.zgrid)
    NFEt = len(m.t) - 1
    
    # Create numpy array of zeros 
    D1 = np.zeros((Ngrid,NFEt+1))
    D2 = np.zeros((Ngrid,NFEt+1))
    D3 = np.zeros((Ngrid,NFEt+1))
    D4 = np.zeros((Ngrid,NFEt+1))
    
    # Loop over time and space, extract values from Pyomo variable
    for i,t in enumerate(m.t):
        for j,z in enumerate(m.zgrid):
            if m.diff ==0:
                D1[j,i] = value(var['base',ind,z,t])
            elif m.diff ==1:
                D1[j,i] = value(var['base',ind,z,t])
                D2[j,i] = value(var['forward_k',ind,z,t])
                D3[j,i] = value(var['forward_ua',ind,z,t])
            elif m.diff ==-1:
                D1[j,i] = value(var['base',ind,z,t])
                D2[j,i] = value(var['backward_k',ind,z,t])
                D3[j,i] = value(var['backward_ua',ind,z,t])
            elif m.diff ==2:
                D1[j,i] = value(var['forward_k',ind,z,t])
                D2[j,i] = value(var['forward_ua',ind,z,t])
                D3[j,i] = value(var['backward_k',ind,z,t])
                D4[j,i] = value(var['backward_ua',ind,z,t])
    return D1,D2, D3, D4

def make_plots(m):
    '''
    Make plots. 
    
    Arguments:
        m: the model
        
    return: None 
    
    other: plots
    '''

    # Extract grid position values
    Z = []
    for j in m.zgrid:
        Z.append(value(j))

    # Extract time values   
    T = []
    for j in m.t:
        T.append(value(j))

    # Visualize pressure
    P1, P2 = extract2d(m, m.P)
    if(len(T) > 1):
        h = plt.contourf(T,Z,P1)
        plt.xlabel('Time [sec]')
        plt.ylabel('Bed Position [scaled]')
        plt.colorbar(h)
        plt.title('Pressure [bar]')
    else:
        plt.plot(Z,P1)
        plt.xlabel('Bed Position [scaled]')
        plt.ylabel('Pressure [bar]')
    plt.show()
    
    if m.energy:
        temp1, temp2 = extract2d(m,m.temp)
        if(len(T) > 1):
            h = plt.contourf(T,Z,temp1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Temperature [K]')
        else:
            plt.plot(Z,temp1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Temperature [K]')
        plt.show()
    
    if m.isotherm:
        SPP1, SPP2 = extract3d(m, m.spp, 'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,SPP1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Surface pressure [bar]')
        else:
            plt.plot(Z,SPP1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Surface pressure [bar]')
        plt.show()

    # Visualize velocity
    V1, V2 = extract2d(m, m.v)
    if(len(T) > 1):
        h = plt.contourf(T,Z,V1)
        plt.xlabel('Time [sec]')
        plt.ylabel('Bed Position [scaled]')
        plt.colorbar(h)
        plt.title('Velocity [cm/s]')
    else:
        plt.plot(Z,V1)
        plt.xlabel('Bed Position [scaled]')
        plt.ylabel('Velocity [cm/s]')
    
    plt.show()
   
        
    # Visualize density
    # for c in ['N2','CO2','He']:
    for c in ['N2','CO2']:
        den1, den2 = extract3d(m, m.C, c)
        if(len(T) > 1):
            h = plt.contourf(T,Z,den1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title(c +' Density [mol/m$^3$]')
        else:
            plt.plot(Z,den1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel(c +' Density [mol/m$^3$]')
        plt.show()

    if m.chemsorb:
        
        # Visualize isotherm
        nchemstar1, nchemstar2 = extract3d(m, m.nchemstar,'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,nchemstar1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Chemical Isotherm [mol/kg]')
        else:
            plt.plot(Z,nchemstar1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Chemical Isotherm [mol/kg]')
        plt.show()
        
        # Visualize loading
        nchem1, nchem2 = extract3d(m, m.nchem,'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,nchem1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Chemical Loading [mol/kg]')
        else:
            plt.plot(Z,nchem2)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Chemical Loading [mol/kg]')
        plt.show()
        
    if m.physsorb:
        nphysstar1, nphysstar2 = extract3d(m, m.nphysstar,'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,nphysstar1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Physical Isotherm [mol/kg]')
        else:
            plt.plot(Z,nphysstar1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Physical Isotherm [mol/kg]')
        plt.show()
           
        nphys1, nphys2 = extract3d(m, m.nphys,'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,nphys1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('Physical Loading [mol/kg]')
        else:
            plt.plot(Z,nphys1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('Physical Loading [mol/kg]')
        plt.show()
        
        alpha1, alpha2 = extract3d(m, m.alpha,'CO2')
        if(len(T) > 1):
            h = plt.contourf(T,Z,alpha1)
            plt.xlabel('Time [sec]')
            plt.ylabel('Bed Position [scaled]')
            plt.colorbar(h)
            plt.title('alpha is modified isotherm [dimensionless]')
        else:
            plt.plot(Z,alpha1)
            plt.xlabel('Bed Position [scaled]')
            plt.ylabel('alpha is modified isotherm [dimensionless]')
        plt.show()


### Functions that are no longer used
def init1D(dyn_var, t0_var,set1,T):
    ''' 
    Initialize dynamic Pyomo model with steady-state solution
    
    Args:
        dyn_var: variable in dynamic model
        t0_var: variable in single timestep model (often initial condition)
        set1: set to enumerate over
        T: time set for dyn_var
        
    Returns:
        Nothing
    
    Assumptions:
        dyn_var and t0_var have the same indicies
        time is the last index
        
    '''
    
    for i in set1:
        for t in T:
            dyn_var[i,t] = value(t0_var[i,0])


def custom_ipopt():
    ''' Return compiled version of Ipopt
    
    Arguments:
        None
        
    Returns:
        solver
    
    '''

    import socket
    name = socket.gethostname()
    print(name)

    if name == "dowling.comsel.nd.edu":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/Users/adowling/src/CoinIpopt/build/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    elif name[0:7] == "Laptop5" or name[0:7] == "laptop5":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/Users/adowling/src/CoinIpopt/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    elif name == "luuyoyodeMacBook-Pro.local":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/usr/local/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    elif name == "luuyoyodembp.dhcp.nd.edu":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/usr/local/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    elif name== "wangjialudembp.dhcp.nd.edu":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/usr/local/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    elif name== "wangjialudeMacBook-Pro.local":
        print("Loading custom Ipopt...")
        solver = SolverFactory('ipopt',executable="/usr/local/bin/ipopt")
        solver.options['linear_solver'] = "ma57"
    else:
        print("Loading default Ipopt.")
        solver = SolverFactory('ipopt')
        
    return solver

from pyomo.core.kernel.component_set import ComponentSet

def large_residuals_set(block, tol=1e-5, LOUD=True):
    """
    Method to return a ComponentSet of all Constraint components with a
    residual greater than a given threshold which appear in a model.
    Args:
        block : model to be studied
        tol : residual threshold for inclusion in ComponentSet
    Returns:
        A ComponentSet including all Constraint components with a residual
        greater than tol which appear in block
    """
    large_residuals_set = ComponentSet()
    for c in block.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        if c.active and value(c.lower - c.body()) > tol:
            large_residuals_set.add(c)
        elif c.active and value(c.body() - c.upper) > tol:
            large_residuals_set.add(c)
    
    if LOUD:
        print("All equality constraints with residuals larger than",tol)
        for c in large_residuals_set:
            print(c)
            
    return large_residuals_set

def get_size_csv(store_):
    '''
    Get size 
    '''
    
    Z = np.unique(np.asarray(store_['position']))
    T = np.unique(np.asarray(store_['time']))
    
    Ngrid = len(Z)
    NFEt = len(T) - 1
    return Ngrid, NFEt
    

    
def makeplot_csv(store_):
    '''
    Make plot from results stored before
    To make plots from .csv file in fixed_bed_init.ipynb: first run code before solver, then run this function
    
    Arguments: 
        store_: a single pandas dataframe
        
    Return: None
    
    Other: plots
    '''
    # Extract variable values from dataframe
    den_N2 = np.asarray(store_['den_N2'])
    den_CO2 = np.asarray(store_['den_CO2'])
    den_He = np.asarray(store_['den_He'])
    vel_ = np.asarray(store_['vel'])
    nchem_eq = np.asarray(store_['nchem_eq'])
    nphys_eq = np.asarray(store_['nphys_eq'])
    nchem = np.asarray(store_['nchem'])
    nphys = np.asarray(store_['nphys'])

    Z = np.unique(np.asarray(store_['position']))
    T = np.unique(np.asarray(store_['time']))
    
    Ngrid, NFEt = get_size_csv(store_)

    # Reshape the array into [# of grids, # of time nodes]
    c_N2 = np.reshape(den_N2,(Ngrid,NFEt+1))
    c_CO2 = np.reshape(den_CO2,(Ngrid,NFEt+1))
    c_He = np.reshape(den_He,(Ngrid,NFEt+1))
    V = np.reshape(vel_,(Ngrid,NFEt+1))
    Nchem_e = np.reshape(nchem_eq,(Ngrid,NFEt+1))
    Nphys_e = np.reshape(nphys_eq,(Ngrid,NFEt+1))
    Nchem = np.reshape(nchem,(Ngrid,NFEt+1))
    Nphys = np.reshape(nphys,(Ngrid,NFEt+1))
 
    # Meshgrid
    x = np.array(range(NFEt+1))
    y = np.array(range(Ngrid))
    [X,Y] = np.meshgrid(x,y)
    
    z_N2 = c_N2
    z_CO2 = c_CO2
    z_He = c_He
    z_vel = V
    z_nce = Nchem_e
    z_npe = Nphys_e
    z_nc = Nchem
    z_nph = Nphys
    
    # Visualize pressure
    P = np.ones((NFEt,Ngrid))
    P = (z_N2 + z_CO2 + z_He) * RPV * temp
    h = plt.contourf(X,Y,P)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Pressure [MPa]')
    plt.show()
    
    
    # Visualize velocity
    h = plt.contourf(X,Y,z_vel)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Velocity [m/s]')
    plt.show()
    
    # Visualize N2 density
    h = plt.contourf(X,Y,z_N2)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('N2 Density [kmol/m$^3$]')
    plt.show()
    
    # Visualize CO2 density
    h = plt.contourf(X,Y,z_CO2)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('CO2 Density [kmol/m$^3$]')
    plt.show()
    
    # Visualize He density
    h = plt.contourf(X,Y,z_He)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('He Density [kmol/m$^3$]')
    plt.show()
    
    # Visualize Chemical isotherm
    h = plt.contourf(X,Y,z_nce)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Chemical Isotherm [mmol/g]')
    plt.show()
    
    # Visualize physical isotherm
    h = plt.contourf(X,Y,z_npe)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Physical Isotherm [mmol/g]')
    plt.show()
    
    # Visualize chemical loading
    h = plt.contourf(X,Y,z_nc)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Chemical Loading [kmol/kg]')
    plt.show()
    
    # Visualize physical loading
    h = plt.contourf(X,Y,z_nph)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Physical Loading [kmol/kg]')
    plt.show()
    
    
def extract2(m):
    ''' 
    Extract results from Pyomo model
    
    Arguments:
        m: the model
    
    Return: a single pandas dataframe storing all results
    '''
    nTime = len(m.t)
    nGrid = len(m.zgrid)
    n = nTime*nGrid
    print (nTime)
    print (n)
    
    ### Variables to be extracted
    
    # Two 3D variables, for COMPS(N2, CO2, He)
    
    C_N2_1, C_N2_2 = extract3d(m,m.C,'N2')
    C_CO2_1, C_CO2_2 = extract3d(m,m.C,'CO2')
    
    C_N21 = np.reshape(C_N2_1, n)
    C_CO21 = np.reshape(C_CO2_1, n)
    
    C_N22 = np.reshape(C_N2_2, n)
    C_CO22 = np.reshape(C_CO2_2, n)
    
    dcdt_N2_1, dcdt_N2_2 = extract3d(m,m.dCdt,'N2')
    dcdt_CO2_1, dcdt_CO2_2 = extract3d(m,m.dCdt,'CO2')
    
    dcdt_N21 = np.reshape(dcdt_N2_1,n)
    dcdt_CO21 = np.reshape(dcdt_CO2_1,n)
    
    dcdt_N22 = np.reshape(dcdt_N2_2,n)
    dcdt_CO22 = np.reshape(dcdt_CO2_2,n)
    
    # Three 2D variables
    vel_1, vel_2 = extract2d(m, m.v)
    
    vel1 = np.reshape(vel_1, n)
    vel2 = np.reshape(vel_2,n)
    
    P_1, P_2 = extract2d(m,m.P)
    P1 = np.reshape(P_1,n)
    P2 = np.reshape(P_2, n)
    
    total_den_1, total_den_2 = extract2d(m,m.total_den)
    total_den1 = np.reshape(total_den_1, n)
    total_den2 = np.reshape(total_den_2, n)
    
    if m.energy:
        T_1, T_2 = extract2d(m,m.temp)
        T1 = np.reshape(T_1, n)
        T2 = np.reshape(T_2, n)
        
        dTdt_1, dTdt_2 = extract2d(m,m.dTdt)
        dTdt1 = np.reshape(dTdt_1,n)
        dTdt2 = np.reshape(dTdt_2,n)
        
        nplin_1, nplin_2 = extract2d(m, m.nplin)
        nplin1 = np.reshape(nplin_1, n)
        nplin2 = np.reshape(nplin_2, n)
        
    # Seven 3D variables, for SCOMPS(CO2)
    if m.isotherm:
        # Seven 3D variables, for SCOMPS(CO2)
        spp_1, spp_2 = extract3d(m, m.spp, 'CO2') 
        
        spp1 = np.reshape(spp_1,n)
        spp2 = np.reshape(spp_2,n)
    if m.chemsorb:
        nchemstar_1, nchemstar_2 = extract3d(m, m.nchemstar,'CO2')
        
        nchemstar1 = np.reshape(nchemstar_1, n)
        nchemstar2 = np.reshape(nchemstar_2, n)
        
        dnchemdt_1, dnchemdt_2 = extract3d(m, m.dnchemdt,'CO2')
        dnchemdt1 = np.reshape(dnchemdt_1, n)
        dnchemdt2 = np.reshape(dnchemdt_2, n)
        
        nchem_1, nchem_2 = extract3d(m, m.nchem,'CO2')
        nchem1 = np.reshape(nchem_1, n)
        nchem2 = np.reshape(nchem_2, n)
        
    if m.physsorb:
        nphysstar_1, nphysstar_2 = extract3d(m, m.nphysstar,'CO2')
        nphysstar1 = np.reshape(nphysstar_1, n)
        nphysstar2 = np.reshape(nphysstar_2, n)
        
        dnphysdt_1, dnphysdt_2 = extract3d(m, m.dnphysdt,'CO2')
        dnphysdt1 = np.reshape(dnphysdt_1, n)
        dnphysdt2 = np.reshape(dnphysdt_2, n)
        
        nphys_1, nphys_2 = extract3d(m, m.nphys,'CO2')
        nphys1 = np.reshape(nphys_1, n)
        nphys2 = np.reshape(nphys_2, n)
        
        print(np.shape(nphys1))
        print(np.shape(nphys2))
    
    
    # Meshgrid
    x = []
    for t in m.t:
        x.append(value(t))
    
    y = []
    for z in m.zgrid:
        y.append(value(z))

    
    [X,Y] = np.meshgrid(x,y)
    time = np.reshape(X,n)
    space = np.reshape(Y,n)
   
    if m.chemsorb and m.physsorb:
        if m.energy:
            store = pd.DataFrame({'time': time,
                              'position':space,
                              'den_N2': C_N21,
                              'den_CO2': C_CO21,
                              'dcdt_N2':dcdt_N21,
                              'dcdt_CO2':dcdt_CO21,
                              'vel': vel1,
                              'pressure':P1,
                              'temp': T1,
                              'dTdt': dTdt1, 
                              'nplin': nplin1, 
                              'total_den':total_den1,
                              'solid_pres':spp1,
                              'nchem_eq': nchemstar1,
                              'nphys_eq': nphysstar1,
                              'dndt_chem': dnchemdt1,
                              'dndt_phys': dnphysdt1,
                              'nchem': nchem1,
                              'nphys':nphys1,
                              'den_N2_u': C_N22,
                              'den_CO2_u': C_CO22,
                              'dcdt_N2_u':dcdt_N22,
                              'dcdt_CO2_u':dcdt_CO22,
                              'vel_u': vel2,
                              'pressure_u':P2,
                              'temp_u': T2,
                              'total_den_u':total_den2,
                              'solid_pres_u':spp2,
                              'nchem_eq_u': nchemstar2,
                              'nphys_eq_u': nphysstar2,
                              'dndt_chem_u': dnchemdt2,
                              'dndt_phys_u': dnphysdt2,
                              'nchem_u': nchem2,
                              'nphys_u': nphys2
                             })
            
        else:
        # save the arrays into the data frame
            store = pd.DataFrame({'time': time,
                              'position':space,
                              'den_N2': C_N21,
                              'den_CO2': C_CO21,
                              'dcdt_N2':dcdt_N21,
                              'dcdt_CO2':dcdt_CO21,
                              'vel': vel1,
                              'pressure':P1,
                              'total_den':total_den1,
                              'solid_pres':spp1,
                              'nchem_eq': nchemstar1,
                              'nphys_eq': nphysstar1,
                              'dndt_chem': dnchemdt1,
                              'dndt_phys': dnphysdt1,
                              'nchem': nchem1,
                              'nphys':nphys1,
                              'den_N2_u': C_N22,
                              'den_CO2_u': C_CO22,
                              'dcdt_N2_u':dcdt_N22,
                              'dcdt_CO2_u':dcdt_CO22,
                              'vel_u': vel2,
                              'pressure_u':P2,
                              'total_den_u':total_den2,
                              'solid_pres_u':spp2,
                              'nchem_eq_u': nchemstar2,
                              'nphys_eq_u': nphysstar2,
                              'dndt_chem_u': dnchemdt2,
                              'dndt_phys_u': dnphysdt2,
                              'nchem_u': nchem2,
                              'nphys_u': nphys2
                             })
    
    elif m.chemsorb:
        store = pd.DataFrame({'time': time,
                              'position':space,
                              'den_N2': C_N2,
                              'den_CO2': C_CO2,
                              'dcdt_N2':dcdt_N2,
                              'dcdt_CO2':dcdt_CO2,
                              'vel': vel_,
                              'pressure':P,
                              'total_den':total_den,
                              'solid_pres':spp,
                              'nchem_eq': nchemstar,
                              'dndt_chem': dnchemdt,
                              'nchem': nchem})
        
    elif m.physsorb:
        store = pd.DataFrame({'time': time,
                              'position':space,
                              'den_N2': C_N2,
                              'den_CO2': C_CO2,
                              'dcdt_N2':dcdt_N2,
                              'dcdt_CO2':dcdt_CO2,
                              'vel': vel_,
                              'pressure':P,
                              'total_den':total_den,
                              'solid_pres':spp,
                              'nphys_eq': nphysstar,
                              'dndt_phys': dnphysdt,
                              'nphys':nphys})
    
    else:
        store = pd.DataFrame({'time': time,
                              'position':space,
                              'den_N2': C_N2,
                              'den_CO2': C_CO2,
                              'dcdt_N2':dcdt_N2,
                              'dcdt_CO2':dcdt_CO2,
                              'vel': vel_,
                              'pressure':P,
                              'total_den':total_den})
    
    return store

def extract3(m,result):
    ''' 
    Extract results from Pyomo model for brute force DoE problem. 
    Compared to extract2: less options kept, add the design variable values 
    
    Arguments:
        m: the model
    
    Return: a single pandas dataframe storing all results
    '''
    nTime = len(m.t)
    nGrid = len(m.zgrid)
    n = nTime*nGrid
    
    ### Variables to be extracted
    
    # Two 3D variables, for COMPS(N2, CO2, He)
    
    FCO2_1, FCO2_2, FCO2_3, FCO2_4 = extract2d(m, m.FCO2)
    
    FCO21 = np.reshape(FCO2_1, n)
    FCO22 = np.reshape(FCO2_2, n)
    FCO23 = np.reshape(FCO2_3, n)
    FCO24 = np.reshape(FCO2_4, n)
    
    C_N2_1, C_N2_2, C_N2_3, C_N2_4 = extract3d(m,m.C,'N2')
    C_CO2_1, C_CO2_2, C_CO2_3, C_CO2_4 = extract3d(m,m.C,'CO2')
    
    C_N21 = np.reshape(C_N2_1, n)
    C_CO21 = np.reshape(C_CO2_1, n)
    
    C_N22 = np.reshape(C_N2_2, n)
    C_CO22 = np.reshape(C_CO2_2, n)
    
    C_N23 = np.reshape(C_N2_3, n)
    C_CO23 = np.reshape(C_CO2_3, n)
    
    C_N24 = np.reshape(C_N2_4, n)
    C_CO24 = np.reshape(C_CO2_4, n)
    
    
    dcdt_N2_1, dcdt_N2_2, dcdt_N2_3, dcdt_N2_4 = extract3d(m,m.dCdt,'N2')
    dcdt_CO2_1, dcdt_CO2_2, dcdt_CO2_3, dcdt_CO2_4 = extract3d(m,m.dCdt,'CO2')
    
    dcdt_N21 = np.reshape(dcdt_N2_1,n)
    dcdt_CO21 = np.reshape(dcdt_CO2_1,n)
    
    dcdt_N22 = np.reshape(dcdt_N2_2,n)
    dcdt_CO22 = np.reshape(dcdt_CO2_2,n)
    
    dcdt_N23 = np.reshape(dcdt_N2_3,n)
    dcdt_CO23 = np.reshape(dcdt_CO2_3,n)
    
    dcdt_N24 = np.reshape(dcdt_N2_4,n)
    dcdt_CO24 = np.reshape(dcdt_CO2_4,n)
    
    # Three 2D variables
    vel_1, vel_2, vel_3, vel_4 = extract2d(m, m.v)
    
    vel1 = np.reshape(vel_1, n)
    vel2 = np.reshape(vel_2,n)
    vel3 = np.reshape(vel_3,n)
    vel4 = np.reshape(vel_4,n)
    
    P_1, P_2, P_3, P_4 = extract2d(m,m.P)
    P1 = np.reshape(P_1,n)
    P2 = np.reshape(P_2, n)
    P3 = np.reshape(P_3, n)
    P4 = np.reshape(P_4, n)
    
    total_den_1, total_den_2, total_den_3, total_den_4 = extract2d(m,m.total_den)
    total_den1 = np.reshape(total_den_1, n)
    total_den2 = np.reshape(total_den_2, n)
    total_den3 = np.reshape(total_den_3, n)
    total_den4 = np.reshape(total_den_4, n)
    
    if m.energy:
        T_1, T_2, T_3, T_4 = extract2d(m,m.temp)
        T1 = np.reshape(T_1, n)
        T2 = np.reshape(T_2, n)
        T3 = np.reshape(T_3, n)
        T4 = np.reshape(T_4, n)
        
        dTdt_1, dTdt_2, dTdt_3, dTdt_4 = extract2d(m,m.dTdt)
        dTdt1 = np.reshape(dTdt_1,n)
        dTdt2 = np.reshape(dTdt_2,n)
        dTdt3 = np.reshape(dTdt_3,n)
        dTdt4 = np.reshape(dTdt_4,n)
        
        nplin_1, nplin_2, nplin_3, nplin_4 = extract2d(m, m.nplin)
        nplin1 = np.reshape(nplin_1, n)
        nplin2 = np.reshape(nplin_2, n)
        nplin3 = np.reshape(nplin_3, n)
        nplin4 = np.reshape(nplin_4, n)
        
    
    # Seven 3D variables, for SCOMPS(CO2)
    spp_1, spp_2, spp_3, spp_4 = extract3d(m, m.spp, 'CO2') 

    spp1 = np.reshape(spp_1,n)
    spp2 = np.reshape(spp_2,n)
    spp3 = np.reshape(spp_3,n)
    spp4 = np.reshape(spp_4,n)

    nchemstar_1, nchemstar_2, nchemstar_3, nchemstar_4 = extract3d(m, m.nchemstar,'CO2')

    nchemstar1 = np.reshape(nchemstar_1, n)
    nchemstar2 = np.reshape(nchemstar_2, n)
    nchemstar3 = np.reshape(nchemstar_3, n)
    nchemstar4 = np.reshape(nchemstar_4, n)
    

    dnchemdt_1, dnchemdt_2, dnchemdt_3, dnchemdt_4 = extract3d(m, m.dnchemdt,'CO2')
    dnchemdt1 = np.reshape(dnchemdt_1, n)
    dnchemdt2 = np.reshape(dnchemdt_2, n)
    dnchemdt3 = np.reshape(dnchemdt_3, n)
    dnchemdt4 = np.reshape(dnchemdt_4, n)
    

    nchem_1, nchem_2, nchem_3, nchem_4 = extract3d(m, m.nchem,'CO2')
    nchem1 = np.reshape(nchem_1, n)
    nchem2 = np.reshape(nchem_2, n)
    nchem3 = np.reshape(nchem_3, n)
    nchem4 = np.reshape(nchem_4, n)


    nphysstar_1, nphysstar_2, nphysstar_3, nphysstar_4 = extract3d(m, m.nphysstar,'CO2')
    nphysstar1 = np.reshape(nphysstar_1, n)
    nphysstar2 = np.reshape(nphysstar_2, n)
    nphysstar3 = np.reshape(nphysstar_3, n)
    nphysstar4 = np.reshape(nphysstar_4, n)

    dnphysdt_1, dnphysdt_2, dnphysdt_3, dnphysdt_4 = extract3d(m, m.dnphysdt,'CO2')
    dnphysdt1 = np.reshape(dnphysdt_1, n)
    dnphysdt2 = np.reshape(dnphysdt_2, n)
    dnphysdt3 = np.reshape(dnphysdt_3, n)
    dnphysdt4 = np.reshape(dnphysdt_4, n)

    nphys_1, nphys_2, nphys_3, nphys_4 = extract3d(m, m.nphys,'CO2')
    nphys1 = np.reshape(nphys_1, n)
    nphys2 = np.reshape(nphys_2, n)
    nphys3 = np.reshape(nphys_3, n)
    nphys4 = np.reshape(nphys_4, n)
    
    
    if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
        status = 'converged'
    elif (result.solver.termination_condition==TerminationCondition.infeasible):
        status = 'infeasible solution'
    else: 
        status = 'other'
    

    
    # Meshgrid
    x = []
    for t in m.t:
        x.append(value(t))
    
    y = []
    for z in m.zgrid:
        y.append(value(z))

    
    [X,Y] = np.meshgrid(x,y)
    time = np.reshape(X,n)
    space = np.reshape(Y,n)
   
    if m.chemsorb and m.physsorb:
        if m.energy:
            store = pd.DataFrame({'time': time,
                              'position':space,
                              'T_inlet': m.temp_orig, 
                              'y_inlet': m.yfeed['CO2'],
                              status: 0,
                              'fco2': FCO21, 
                              'den_N2': C_N21,
                              'den_CO2': C_CO21,
                              'dcdt_N2':dcdt_N21,
                              'dcdt_CO2':dcdt_CO21,
                              'vel': vel1,
                              'pressure':P1,
                              'temp': T1,
                              'dTdt': dTdt1, 
                              'nplin': nplin1, 
                              'total_den':total_den1,
                              'solid_pres':spp1,
                              'nchem_eq': nchemstar1,
                              'nphys_eq': nphysstar1,
                              'dndt_chem': dnchemdt1,
                              'dndt_phys': dnphysdt1,
                              'nchem': nchem1,
                              'nphys':nphys1,
                              'fco2_k': FCO22, 
                              'den_N2_k': C_N22,
                              'den_CO2_k': C_CO22,
                              'dcdt_N2_k':dcdt_N22,
                              'dcdt_CO2_k':dcdt_CO22,
                              'vel_k': vel2,
                              'pressure_k':P2,
                              'temp_k': T2,
                              'total_den_k':total_den2,
                              'solid_pres_k':spp2,
                              'nchem_eq_k': nchemstar2,
                              'nphys_eq_k': nphysstar2,
                              'dndt_chem_k': dnchemdt2,
                              'dndt_phys_k': dnphysdt2,
                              'nchem_k': nchem2,
                              'nphys_k': nphys2,
                              'fco2_u': FCO23, 
                              'den_N2_u': C_N23,
                              'den_CO2_u': C_CO23,
                              'dcdt_N2_u':dcdt_N23,
                              'dcdt_CO2_u':dcdt_CO23,
                              'vel_u': vel3,
                              'pressure_u':P3,
                              'temp_u': T3,
                              'total_den_u':total_den3,
                              'solid_pres_u':spp3,
                              'nchem_eq_u': nchemstar3,
                              'nphys_eq_u': nphysstar3,
                              'dndt_chem_u': dnchemdt3,
                              'dndt_phys_u': dnphysdt3,
                              'nchem_u': nchem3,
                              'nphys_u': nphys3,
                              'fco2_f': FCO24, 
                              'den_N2_f': C_N24,
                              'den_CO2_f': C_CO24,
                              'dcdt_N2_f':dcdt_N24,
                              'dcdt_CO2_f':dcdt_CO24,
                              'vel_f': vel4,
                              'pressure_f':P4,
                              'temp_f': T4,
                              'total_den_f':total_den4,
                              'solid_pres_f':spp4,
                              'nchem_eq_f': nchemstar4,
                              'nphys_eq_f': nphysstar4,
                              'dndt_chem_f': dnchemdt4,
                              'dndt_phys_f': dnphysdt4,
                              'nchem_f': nchem4,
                              'nphys_f': nphys4
                             })
            
        else:
        # save the arrays into the data frame
        # TODO: add the third set 
            store = pd.DataFrame({'time': time,
                              'position':space,
                              'T_inlet': m.temp_orig, 
                              'y_inlet': m.yfeed['CO2'],
                              'fco2': FCO21, 
                              'den_N2': C_N21,
                              'den_CO2': C_CO21,
                              'dcdt_N2':dcdt_N21,
                              'dcdt_CO2':dcdt_CO21,
                              'vel': vel1,
                              'pressure':P1,
                              'total_den':total_den1,
                              'solid_pres':spp1,
                              'nchem_eq': nchemstar1,
                              'nphys_eq': nphysstar1,
                              'dndt_chem': dnchemdt1,
                              'dndt_phys': dnphysdt1,
                              'nchem': nchem1,
                              'nphys':nphys1,
                              'den_N2_u': C_N22,
                              'den_CO2_u': C_CO22,
                              'dcdt_N2_u':dcdt_N22,
                              'dcdt_CO2_u':dcdt_CO22,
                              'vel_u': vel2,
                              'pressure_u':P2,
                              'total_den_u':total_den2,
                              'solid_pres_u':spp2,
                              'nchem_eq_u': nchemstar2,
                              'nphys_eq_u': nphysstar2,
                              'dndt_chem_u': dnchemdt2,
                              'dndt_phys_u': dnphysdt2,
                              'nchem_u': nchem2,
                              'nphys_u': nphys2
                             })
    
    else:
        print('check the adsorption options!!! not a square problem')
    
    return store
                
def initial_bed_csv(m, store_):
    '''
    Initialize the bed with values for every time node and every grid.
    This version uses linear interporation

    Arguments:
        m: model
        store_ : the pandas dataframe storing solution
        note that this function will be used before running the model, so this store_ will be the previous solution. 

    Return: None

    '''
    
    # Extract the length of time nodes and grid nodes
    nTime = len(m.t)
    nGrid = len(m.zgrid)
    n = nTime*nGrid
    
    
    
    Z = np.unique(np.asarray(store_['position']))
    T = np.unique(np.asarray(store_['time']))
    
    print('Model # of time grid is', nTime, ', initial point # of time grid is', len(T))

    # Extract every variable
    den_N2 = np.asarray(store_['den_N2'])
    den_CO2 = np.asarray(store_['den_CO2'])

    dcdt_N2 = np.asarray(store_['dcdt_N2'])
    dcdt_CO2 = np.asarray(store_['dcdt_CO2'])
    
    P = np.asarray(store_['pressure'])
    vel_ = np.asarray(store_['vel'])
    total_den = np.asarray(store_['total_den'])
    
    if m.energy:
        temp_co = np.asarray(store_['temp'])
        dTdt_co = np.asarray(store_['dTdt'])
        nplin_co = np.asarray(store_['nplin'])
    
    if m.isotherm:
        s_pres = np.asarray(store_['solid_pres'])

    if m.physsorb:
        nphys_eq = np.asarray(store_['nphys_eq'])
        dnphysdt = np.asarray(store_['dndt_phys'])
        nphys = np.asarray(store_['nphys'])
        
    if m.chemsorb:
        nchem_eq = np.asarray(store_['nchem_eq'])
        dnchemdt = np.asarray(store_['dndt_chem'])
        nchem = np.asarray(store_['nchem'])
        
    c_N2 = interp2d(Z, T, den_N2, kind='cubic')
    c_CO2 = interp2d(Z, T, den_CO2, kind='cubic')
    
    dcdt_N2_ = interp2d(Z, T, dcdt_N2, kind='cubic')
    dcdt_CO2_ = interp2d(Z, T, dcdt_CO2, kind='cubic')
    
    V = interp2d(Z, T, vel_, kind='cubic')
    pres = interp2d(Z, T, P, kind='cubic')
    tot_den = interp2d(Z, T, total_den, kind='cubic')
    
    if m.energy:
        temp_e = interp2d(Z, T, temp_co, kind='cubic')
        dTdt_e = interp2d(Z, T, dTdt_co, kind='cubic')
        nplin_e = interp2d(Z, T, nplin_co, kind='cubic')
    
    if m.isotherm:
        solid_pres = interp2d(Z, T, s_pres, kind='cubic')
    if m.physsorb:
        Nphys_e = interp2d(Z, T, nphys_eq, kind='cubic')
        dndt_phys = interp2d(Z, T, dnphysdt, kind='cubic')
        Nphys = interp2d(Z, T, nphys, kind='cubic')
        
    if m.chemsorb:
        Nchem_e = interp2d(Z, T, nchem_eq, kind='cubic')
        dndt_chem = interp2d(Z, T, dnchemdt, kind='cubic')
        Nchem = interp2d(Z, T, nchem, kind='cubic')
        

    # Loop for every time and grid nodes
    for j in m.perturb:
        for z in m.zgrid:
            for i in m.t:
                z_ = value(z) /(Ngrid)
                t_ = value(i)
                m.C[j,'N2',z,i] = c_N2(z_, t_)[0]
                m.C[j,'CO2',z,i] = c_CO2(z_,t_)[0]

                m.dCdt[j,'N2',z,i] = dcdt_N2_(z_, t_)[0]
                m.dCdt[j,'CO2',z,i] = dcdt_CO2_(z_, t_)[0]

                m.v[j,z,i] = V(z_,t_)[0]    
                m.P[j,z,i] = pres(z_,t_)[0]
                m.total_den[j,z,i] = tot_den(z_,t_)[0]
                
                if m.energy:
                    m.temp[j,z,i] = temp_e(z_, t_)[0]
                    m.dTdt[j,z,i] = dTdt_e(z_, t_)[0]
                    m.nplin[j,z,i] = nplin_e(z_, t_)[0]
                    
                if m.isotherm:
                    m.spp[j,'CO2',z,i] = solid_pres(z_, t_)[0]
                    x = alpha_scale*(solid_pres(z_, t_)[0] - plin)

                    if alpha_option == 1:
                        alpha_ = 0.5*(1 - x / np.sqrt(eps_alpha + x*x))
                    elif alpha_option == 2:
                        alpha_ = 1 - 1 /(1 + exp(-x))
                    elif alpha_option == 3: 
                        alpha_ = 0.0
                    elif alpha_option == 4: 
                        alpha_ = 1.0

                    if alpha_variable:
                        m.alpha[j,'CO2',z,i] = alpha_

                if m.chemsorb:
                    m.nchemstar[j,'CO2',z,i] = Nchem_e(z_, t_)[0]
                    m.dnchemdt[j,'CO2',z,i] = dndt_chem(z_, t_)[0]
                    m.nchem[j,'CO2',z,i] = Nchem(z_, t_)[0]
                    if m.energy:
                        m.nchemstar_mod[j,'CO2',z,i] = alpha_*value(m.nplin[j,z,i])*solid_pres(z_, t_)[0]/plin + (1-alpha_)*Nchem_e(z_, t_)[0]
                    else:
                        m.nchemstar_mod[j,'CO2',z,i] = alpha_*m.nplin*solid_pres(z_, t_)[0]/plin + (1-alpha_)*Nchem_e(z_, t_)[0]

                if m.physsorb:
                    m.nphysstar[j,'CO2',z,i] = Nphys_e(z_, t_)[0]
                    #m.nphysstar_mod[j,'CO2',z,i] = alpha_*m.nplin*solid_pres(z_, t_)[0]/plin + (1-alpha_)*Nphys_e(z_, t_)[0]
                    m.dnphysdt[j,'CO2',z,i] = dndt_phys(z_, t_)[0]
                    m.nphys[j,'CO2',z,i] = Nphys(z_, t_)[0]   
            
            


def compute_Kp(mod,LOUD=True):
    ''' Compute Kp from the solution [m^2/s]
    
    '''
    Kp_ = {}
    for c in mod.SCOMPS:
        tfc = value(mod.fitted_transport_coefficient[c])

        Kp_[c] = radp*radp/15/ads_epsp/tfc
        
        if LOUD:
            print("fitted_transport_coefficient[",c,"] =",tfc,"s")
            print("Kp[",c,"] =",Kp_[c],"m^2/s")
        
    return Kp_


### DoE analysis function
def ext_pres_double(total, option1, option2, node):
    '''
    Extract the solution of a variable
    Arguments:
        total: a pandas dataframe of solutions
        option: name of the variable
        node: number of timesteps
    Note: the Ngrid is fixed to 20 
    Return: a list
    '''
    x_py1 = total[option1]
    x_pyomo1 = np.zeros((20,node))

    for i in range(0,20):
        for j in range(0,node):
            x_pyomo1[i,j] = x_py1.iloc[i*node+j]
            
    x_py2 = total[option2] 
    x_pyomo2 = np.zeros((20,node))

    for m in range(0,20):
        for k in range(0,node):
            x_pyomo2[m,k] = x_py2.iloc[m*node+k]        
        
    return x_pyomo1, x_pyomo2

def ext_pres(total, option1, node):
    '''
    Extract the solution of a variable
    Arguments:
        total: a pandas dataframe of solutions
        option: name of the variable
        node: number of timesteps
    Note: the Ngrid is fixed to 20 
    Return: a list
    '''
    x_py1 = total[option1]
    x_pyomo1 = np.zeros((20,node))

    for i in range(0,20):
        for j in range(0,node):
            x_pyomo1[i,j] = x_py1.iloc[i*node+j]      
        
    return x_pyomo1


def get_fco2(con_s, vel_s):
    '''
    Calculate the outlet CO2 molar flowrate with extracted solutions
    Argument:
        con_s: a list of CO2 density
        vel_s: a list of velocity
    Note: the Ngrid is fixed to 20 
    Return: a list of FCO2
    '''
    outlet = con_s[19, :]
    outlet_velo = vel_s[19, :]

    FCO2 = np.zeros((len(outlet)))    
    for i in range(0, len(outlet)):
        #LHS: mmol/min
        #RHS: mol/m3 * cm/s * m2 * 60*1000/100
        FCO2[i] = (outlet[i]*outlet_velo[i]*3.1415926*rbed*rbed*600)
    return FCO2

def average(lis):
    '''
    Get the average value of velocity/pressure etc. 
    '''
    li = np.asarray(lis)
    res = li.reshape([69*20,])
    return sum(res)/len(res)

def time_points(n, time_range):
    '''
    Split the timespan of the experiment into n nodes
    
    Argument: n: the number of time nodes.
    
    return: time_points, a list of values of timepoints in [s]
    '''
    time_points = []
    int_time = time_range/n
    
    # specify where the time begins. Usually 0
    time_ele = 0
    time_points.append(time_ele)
    for i in range(n):
        time_ele += int_time
        time_points.append(time_ele)

    return time_points

def Residual(m):
    '''
    Calculate the residual with a given model
    return: 
        res: the residual at every timepoint
        FCO2: the CO2 outlet flowrate at every timepoint
    '''
    
    # Extract time 
    T = []
    for j in m.t:
        T.append(value(j))
        t_final = T[-1]
        
    # Extract CO2 density 
    outlet_den = extract3d(m, m.C, 'CO2')
    outlet_ = []
    outlet_.append(outlet_den[Ngrid-1, :])
    outlet = np.reshape(outlet_, len(T))
    
    # Extract velocity 
    outlet_vel = extract2d(m, m.v)
    outlet_v = []
    outlet_v.append(outlet_vel[Ngrid-1, :])
    outlet_velo = np.reshape(outlet_v, len(T))

    # Extract values of FCO2
    FCO2_ = []
    for i in range(0, len(T)):
        FCO2_.append(value(m.FCO2[T[i]]))
    
    # Extract values of yCO2
    yCO2_ = []
    for i in range(0, len(T)):
        yCO2_.append(value(m.yCO2[T[i]]))
     
    # Get value of residual
    res = 0
    for i in range(0, len(outlet)):
        res += (outlet[i]*outlet_velo[i]*3.1415926*rbed*rbed*600 - FCO2_[i])**2
        
    FCO2 = np.zeros((len(outlet)))
    for i in range(0, len(outlet)):
        FCO2[i] = (outlet[i]*outlet_velo[i]*3.1415926*rbed*rbed*600)
    
    return res, FCO2

def breakthrough_modify(m,exp):
    '''
    Draw breakthrough curve, comparing experimental data with simulation data. 
    
    Arguments:
        m: moel
        exp: experimental data
    
    
    Return: None 
    
    Other: plot
    
    '''
    
    break_wvu = pd.read_csv('breakthrough_wvu.csv')
    
    # unit: mol/m3
    outlet_den = extract3d(m, m.C, 'CO2')
    
    # unit: cm/s
    outlet_vel = extract2d(m, m.v)
    
    outlet_n2 = extract3d(m, m.C, 'N2')
    #print(outlet_den)
    #print(outlet_n2)
    #print(outlet_vel)
    T = []
    for j in m.t:
        T.append(value(j))
        t_final = T[-1]
        
    #print(T)
    #print(exp['time']*60)
    #print(break_wvu['time']*60)
        
    data_c1 = exp['yCO2']
    data_c = np.asarray(data_c1)
    data_t1 = exp['time']  
    data_t = np.asarray(data_t1*60+10)
    new_data = np.interp(T, data_t, data_c)
    
    #print(break_wvu['FCO2'])
    #print(exp['time'])
    
    plt.plot(exp['time']*60, exp['yCO2'],'b.', color='r')
    #plt.plot(exp_nolin['time']*60, exp_nolin['yCO2'], label='WVU curve')
    plt.plot(break_wvu['time']*60, break_wvu['FCO2'], label='WVU curve')
    plt.plot(T, outlet_den[-1,:]*outlet_vel[-1,:]/(outlet_den[-1, -1]*outlet_vel[-1,-1]), label='Pyomo curve')
    
    # Use for 1200 - 1500 seconds:
    
    #plt.plot(exp['time'][35:42]*60, exp['yCO2'][35:42],'b.', color='r')
    #plt.plot(break_wvu['time'][1700:2000]*60, break_wvu['FCO2'][1700:2000], label='WVU curve')
    #plt.plot(T[1:], outlet_den[-1,1:]*outlet_vel[-1,1:]/6.7, label='Pyomo curve')
    
    #plt.plot(T, outlet_n2[-1,:]*outlet_vel[-1,:]/(outlet_n2[-1, -1]*outlet_vel[-1,-1]),label='N2')
    plt.xlabel('time [s]')
    plt.ylabel('Normalized outlet gas density of CO2')
    plt.title('breakthrough curve')
    plt.legend()
    plt.show()

    

def energy_balance_old(m, j, z, t):
    '''
    Energy balance assuming Tambient = Twall, Tg=Ts
    Equation from [Dowling, 2012]
    
    Arguments: 
        m: model
        j: model.perturb 
        i: model.SCOMPS or model.COMPS
        z: model.zgrid
        t: model.t

    Return: ODE for calculating component species density
    
    '''
    # calculate sum(ci*cpg)
    # mol/m3 * J/mol/K = J/m3/K
    sum_c = sum(m.C[j,i,z,t]*m.cpg[j,i,z,t] for i in m.COMPS)
    
    # calculate the coefficient of dT/dt
    # J/m3/K + kg/m3 * J/kg/K = J/m3/K
    dividant = ads_epsb*sum_c + den_bed*cps 
    
    # calculate the sum of Hi*dni/dt
    # kJ/mol * mol/kg/s = kJ/kg/s
    sum_hdn = sum(-m.H_ads[a]*(m.dnchemdt[j,a,z,t]+m.dnphysdt[j,a,z,t]) for a in m.SCOMPS)
    
    # sum of heat flow, [J/m3]
    # RHS: mol/m3 * J/mol = J/m3 
    h_sum = sum(m.C[j,b,z,t]*m.h[j,b,z,t] for b in m.COMPS)
    
    # heat flow feed, [J/m3]
    h_sum_feed = sum(m.den_f[p]*m.h_feed[p] for p in m.COMPS)
    
    # LHS: W/m3 = J/s/m3 
    # RHS: cm/s * 1m/100cm * J/m3 * 1/m = J/s/m3 
    if z==0: 
        duhdz = (m.v[j,z,t]*0.01*h_sum - vel_f*0.01*h_sum_feed)/dz
    else:
        # heat flow of the previous grid, [J/m3]
        h_sum_back = sum(m.C[j,b,z-1,t]*m.h[j,b,z-1,t] for b in m.COMPS)
        duhdz = (m.v[j,z,t]*0.01*h_sum - m.v[j,z-1,t]*0.01*h_sum_back)/dz
    
    # LHS: K/s
    # RHS: (kg/m3 * kJ/kg/s * 1000J/1kJ - J/m3/s - J/s/m3/K *K)/ (J/m3/K) = K/s 
    #if m.temp_option==2:
    #    return m.dTdt[j,z,t] == (den_bed*sum_hdn*1000 - duhdz - exp(m.ua[j])*(m.temp[j,z,t]-m.temp_bath))/dividant
    #elif m.temp_option==1:
    if m.est_ua:
        return m.dTdt[j,z,t] == (den_bed*sum_hdn*1000 - duhdz - exp(m.ua)*(m.temp[j,z,t]-m.temp_bath))/dividant
    else:
        return m.dTdt[j,z,t] == (den_bed*sum_hdn*1000 - duhdz - exp(m.ua[j])*(m.temp[j,z,t]-m.temp_bath))/dividant
    
    
def breakthrough_modify2(m, file=None, source="computer"):
    '''
    Draw breakthrough curve of the Pyomo model
    
    Arguments:
        m: Pyomo model
        file: when source = 'computer', this is where the computer experimental data is stored. Otherwise, its default is None
        source: if computer, it is plotting the computer experiment data. If lab, its comparing the experiments from the lab
    
    Return: None 
    
    Other: plot
    
    '''
    
    
    # unit: mol/m3
    outlet_den, _, _, _ = extract3d(m, m.C, 'CO2')
    
    # unit: cm/s
    outlet_vel, _, _, _ = extract2d(m, m.v)
    
    outlet_n2, _, _, _ = extract3d(m, m.C, 'N2')
    #print(outlet_den)
    #print(outlet_n2)
    #print(outlet_vel)
    
    model_temp, _,_,_ = extract2d(m,m.temp)
    
    
    T = []
    for j in m.t:
        T.append(value(j))
        t_final = T[-1]
        
    #print(T)
    #print(exp['time']*60)
    #print(break_wvu['time']*60)
    
    if source == "lab":
        break_wvu = pd.read_csv('breakthrough_wvu.csv')
    
        exp = pd.read_csv('co2_breakthrough.csv')
        
        data_c1 = exp['yCO2']
        data_c = np.asarray(data_c1)
        data_t1 = exp['time']  
        data_t = np.asarray(data_t1*60+10)
        new_data = np.interp(T, data_t, data_c)
    
    #print(break_wvu['FCO2'])
    #print(exp['time'])
    
        plt.plot(exp['time']*60, exp['yCO2'],'b.', color='r', label='Experimental data')
        #plt.plot(exp_nolin['time']*60, exp_nolin['yCO2'], label='WVU curve')
        #plt.plot(break_wvu['time']*60, break_wvu['FCO2'], label='WVU curve')
        plt.plot(T, outlet_den[-1,:]*outlet_vel[-1,:]/(outlet_den[-1, -1]*outlet_vel[-1,-1]), label='Model prediction')

        # Use for 1200 - 1500 seconds:

        #plt.plot(exp['time'][35:42]*60, exp['yCO2'][35:42],'b.', color='r')
        #plt.plot(break_wvu['time'][1700:2000]*60, break_wvu['FCO2'][1700:2000], label='WVU curve')
        #plt.plot(T[1:], outlet_den[-1,1:]*outlet_vel[-1,1:]/6.7, label='Pyomo curve')

        #plt.plot(T, outlet_n2[-1,:]*outlet_vel[-1,:]/(outlet_n2[-1, -1]*outlet_vel[-1,-1]),label='N2')
        plt.xlabel('time [s]')
        plt.ylabel('Normalized outlet gas density of CO\N{SUBSCRIPT TWO} ')
        plt.title('Breakthrough curve')
        #plt.savefig('break_tr%.fua%.f.png'%(tr,ua))
        plt.legend()
        plt.show()
        
    elif source == "computer": 
        
        sol = pd.read_csv(file)
    
        #for i in range(len(time_exp)):    
        #    yco2_exp[time_exp[i]] = sol['FCO2'][i]
        #    temp_mid_exp[time_exp[i]] = sol['temp_mid'][i]
        #    temp_end_exp[time_exp[i]] = sol['temp_end'][i]
        
        exp_fco2 = np.asarray(sol['FCO2'].values.tolist())
        exp_temp_mid = np.asarray(sol['temp_mid'].values.tolist())
        exp_temp_end = np.asarray(sol['temp_end'].values.tolist())
        
        plt.plot(T, exp_fco2/exp_fco2[-1], label = 'Experimental data')
        plt.plot(T, outlet_den[-1,:]*outlet_vel[-1,:]/(outlet_den[-1, -1]*outlet_vel[-1,-1]), label='Model prediction')
        plt.xlabel('time [s]')
        plt.ylabel('Normalized outlet gas density of CO\N{SUBSCRIPT TWO} ')
        plt.title('Breakthrough curve')
        #plt.savefig('break_tr%.fua%.f.png'%(tr,ua))
        plt.legend()
        plt.show()
        
        
        plt.plot(T, exp_temp_mid, label='Experimental data of middle T')
        plt.plot(T, model_temp[10,:], label='Model prediction of middle T')
        plt.plot(T, exp_temp_end, label='Experimental data of end T')
        plt.plot(T, model_temp[19,:], label='Model prediction of end T')
        plt.xlabel('time [s]')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature model prediction and experimental data')
        #plt.savefig('break_tr%.fua%.f.png'%(tr,ua))
        plt.legend()
        plt.show()
    
        
        
def makeplot_csv2(store_, temp_show = False):
    '''
    Make plot from results stored before
    To make plots from .csv file in fixed_bed_init.ipynb: first run code before solver, then run this function
    
    Arguments: 
        store_: a single pandas dataframe
        
    Return: None
    
    Other: plots
    '''
    # Extract variable values from dataframe
    den_N2 = np.asarray(store_['den_N2'])
    den_CO2 = np.asarray(store_['den_CO2'])
    vel_ = np.asarray(store_['vel'])
    nchem_eq = np.asarray(store_['nchem_eq'])
    nphys_eq = np.asarray(store_['nphys_eq'])
    nchem = np.asarray(store_['nchem'])
    nphys = np.asarray(store_['nphys'])
    if temp_show:
        temp_ = np.asarray(store_['temp'])
    pres = np.asarray(store_['pressure'])

    Z = np.unique(np.asarray(store_['position']))
    T = np.unique(np.asarray(store_['time']))
    
    Ngrid, NFEt = get_size_csv(store_)

    # Reshape the array into [# of grids, # of time nodes]
    P = np.reshape(pres, (Ngrid, NFEt+1))
    c_N2 = np.reshape(den_N2,(Ngrid,NFEt+1))
    c_CO2 = np.reshape(den_CO2,(Ngrid,NFEt+1))
    V = np.reshape(vel_,(Ngrid,NFEt+1))
    Nchem_e = np.reshape(nchem_eq,(Ngrid,NFEt+1))
    Nphys_e = np.reshape(nphys_eq,(Ngrid,NFEt+1))
    Nchem = np.reshape(nchem,(Ngrid,NFEt+1))
    Nphys = np.reshape(nphys,(Ngrid,NFEt+1))
    if temp_show:
        temp = np.reshape(temp_, (Ngrid, NFEt+1))
 
    # Meshgrid
    x = np.array(range(NFEt+1))*47.0588235294118
    y = np.array(range(Ngrid))/Ngrid
    [X,Y] = np.meshgrid(x,y)
    
    z_N2 = c_N2
    z_CO2 = c_CO2
    z_vel = V
    z_nce = Nchem_e
    z_npe = Nphys_e
    z_nc = Nchem
    z_nph = Nphys
    if temp_show:
        z_T = temp
    
    # Visualize pressure
    h = plt.contourf(X,Y,P)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Pressure [bar]')
    plt.show()
    
    
    # Visualize velocity
    h = plt.contourf(X,Y,z_vel)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Velocity [cm/s]')
    plt.show()
    
    # Visualize N2 density
    h = plt.contourf(X,Y,z_N2)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('N2 Density [mol/m$^3$]')
    plt.show()
    
    # Visualize CO2 density
    h = plt.contourf(X,Y,z_CO2)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('CO2 Density [mol/m$^3$]')
    plt.show()

    
    # Visualize Chemical isotherm
    h = plt.contourf(X,Y,z_nce)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Chemical Isotherm [mmol/g]')
    plt.show()
    
    # Visualize physical isotherm
    h = plt.contourf(X,Y,z_npe)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Physical Isotherm [mmol/g]')
    plt.show()
    
    # Visualize chemical loading
    h = plt.contourf(X,Y,z_nc)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Chemical Loading [mmol/g]')
    plt.show()
    
    # Visualize physical loading
    h = plt.contourf(X,Y,z_nph)
    plt.xlabel('Time [sec]')
    plt.ylabel('Bed Position [scaled]')
    plt.colorbar(h)
    plt.title('Physical Loading [mmol/g]')
    plt.show()
    
    if temp_show:
        # Visualize physical loading
        h = plt.contourf(X,Y,z_T)
        plt.xlabel('Time [sec]')
        plt.ylabel('Bed Position [scaled]')
        plt.colorbar(h)
        plt.title('Temperature[K]')
        plt.show()
        
        
        
def ext_pres(total, option1, node):
    '''
    Extract the solution of a variable
    Arguments:
        total: a pandas dataframe of solutions
        option: name of the variable
        node: number of timesteps
    Note: the Ngrid is fixed to 20 
    Return: a list
    '''
    x_py1 = total[option1]
    x_pyomo1 = np.zeros((20,node))

    for i in range(0,20):
        for j in range(0,node):
            x_pyomo1[i,j] = x_py1.iloc[i*node+j]      
        
    return x_pyomo1


def ext_time(total, node):
    '''
    Extract the time nodes
    Argument:
        total: a pandas dataframe of solutions
        node: the number of timesteps
    return: a list
    '''
    t_py = total['pressure']
    time_p = t_py.index.to_list()[:node]
    return time_p

def get_fco2(con_s, vel_s):
    '''
    Calculate the outlet CO2 molar flowrate with extracted solutions
    Argument:
        con_s: a list of CO2 density
        vel_s: a list of velocity
    Note: the Ngrid is fixed to 20 
    Return: a list of FCO2
    '''
    outlet = con_s[19, :]
    outlet_velo = vel_s[19, :]

    FCO2 = np.zeros((len(outlet)))    
    for i in range(0, len(outlet)):
        #LHS: mmol/min
        #RHS: mol/m3 * cm/s * m2 * 60*1000/100
        FCO2[i] = (outlet[i]*outlet_velo[i]*3.1415926*rbed*rbed*600)
    return FCO2