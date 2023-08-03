# ===========
# Description
# ===========
# A simple aircraft sizing problem, formulated as a Geometric Program
# From:  Hoburg and Abbeel
#        Geometric Programming for Aircraft Design Optimization
#        AIAA Journal
#        2014

# =================
# Import Statements
# =================
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import units
from pyomo.contrib.edi import Formulation
from pyomo.contrib.edi import BlackBoxFunctionModel

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
D   = f.Variable(name = "D",   guess = 300, units = "N",     description = "total drag force")
A   = f.Variable(name = "A",   guess = 10.0, units = "-",    description = "aspect ratio")
S   = f.Variable(name = "S",   guess = 10.0, units = "m^2",  description = "total wing area")
V   = f.Variable(name = "V",   guess = 30.0, units = "m/s",  description = "cruising speed")
W   = f.Variable(name = "W",   guess = 10000.0, units = "N", description = "total aircraft weight")
Re  = f.Variable(name = "Re",  guess = 3e6, units = "-",     description = "Reynold's number")
C_D = f.Variable(name = "C_D", guess = .025, units = "-",    description = "Drag coefficient of wing")
C_L = f.Variable(name = "C_L", guess = .5, units = "-",      description = "Lift coefficent of wing")
C_f = f.Variable(name = "C_f", guess = .003, units = "-",    description = "skin friction coefficient")
W_w = f.Variable(name = "W_w", guess = 2500, units = "N",    description = "wing weight")

# =================
# Declare Constants
# =================
k           = f.Constant(name = "k",      value = 1.2,     units ="-",      description = "form factor")
e           = f.Constant(name = "e",      value = 0.96,    units ="-",      description = "Oswald efficiency factor")
mu          = f.Constant(name = "mu",     value = 1.78e-5, units ="kg/m/s", description = "viscosity of air")
rho         = f.Constant(name = "rho",    value = 1.23,    units ="kg/m^3", description = "density of air")
tau         = f.Constant(name = "tau",    value = 0.12,    units ="-",      description = "airfoil thickness to chord ratio")
N_ult       = f.Constant(name = "N_ult",  value = 2.5,     units ="-",      description = "ultimate load factor")
V_min       = f.Constant(name = "V_min",  value = 22,      units ="m/s",    description = "takeoff speed")
C_Lmax      = f.Constant(name = "C_Lmax", value = 2.0,     units ="-",      description = "max CL with flaps down")
S_wetratio  = f.Constant(name = "Srat",   value = 2.05,    units ="-",      description = "wetted area ratio")
W_W_coeff1  = f.Constant(name = "W_c1",   value = 8.71e-5, units ="1/m",    description = "Wing Weight Coefficent 1")
W_W_coeff2  = f.Constant(name = "W_c2",   value = 45.24,   units ="Pa",     description = "Wing Weight Coefficent 2")
CDA0        = f.Constant(name = "CDA0",   value = 0.0306,  units ="m^2",    description = "fuselage drag area")
W_0         = f.Constant(name = "W_0",    value = 4940.0,  units ="N",      description = "aircraft weight excluding wing")

# =====================
# Declare the Objective
# =====================
f.Objective( D )

# ===================================
# Declare some intermediate variables
# ===================================
pi = np.pi
C_D_fuse = CDA0/S
C_D_wpar = k*C_f*S_wetratio
C_D_ind = C_L**2/(pi*A*e)
W_w_strc = W_W_coeff1*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
W_w_surf = W_W_coeff2 * S

# =======================
# Declare the Constraints
# =======================
f.ConstraintList(
    [
        C_D >= C_D_fuse + C_D_wpar + C_D_ind,
        W_w >= W_w_surf + W_w_strc,
        D >= 0.5*rho*S*C_D*V**2,
        Re == (rho/mu)*V*(S/A)**0.5,
        C_f == 0.074/Re**0.2,
        W == 0.5*rho*S*C_L*V**2,
        W == 0.5*rho*S*C_Lmax*V_min**2,
        W >= W_0 + W_w,
    ]   
)
