#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
from pyomo.contrib.edi import Formulation

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================

A = f.Variable(name="A", guess=10.0, units="-", description="aspect ratio")
C_D = f.Variable(
    name="C_D", guess=0.025, units="-", description="Drag coefficient of wing"
)
C_f = f.Variable(
    name="C_f", guess=0.003, units="-", description="skin friction coefficient"
)
C_L = f.Variable(
    name="C_L", guess=0.5, units="-", description="Lift coefficient of wing"
)
D = f.Variable(name="D", guess=300, units="N", description="total drag force")
Re = f.Variable(name="Re", guess=3e6, units="-", description="Reynold's number")
S = f.Variable(name="S", guess=10.0, units="m^2", description="total wing area")
V = f.Variable(name="V", guess=30.0, units="m/s", description="cruising speed")
W = f.Variable(name="W", guess=10000.0, units="N", description="total aircraft weight")
W_w = f.Variable(name="W_w", guess=2500, units="N", description="wing weight")

# =================
# Declare Constants
# =================
C_Lmax = f.Constant(
    name="C_Lmax", value=2.0, units="-", description="max CL with flaps down"
)
CDA0 = f.Constant(
    name="CDA0", value=0.0306, units="m^2", description="fuselage drag area"
)
e = f.Constant(name="e", value=0.96, units="-", description="Oswald efficiency factor")
k = f.Constant(name="k", value=1.2, units="-", description="form factor")
mu = f.Constant(
    name="mu", value=1.78e-5, units="kg/m/s", description="viscosity of air"
)
N_ult = f.Constant(
    name="N_ult", value=2.5, units="-", description="ultimate load factor"
)
rho = f.Constant(name="rho", value=1.23, units="kg/m^3", description="density of air")
S_wetratio = f.Constant(
    name="Srat", value=2.05, units="-", description="wetted area ratio"
)
tau = f.Constant(
    name="tau", value=0.12, units="-", description="airfoil thickness to chord ratio"
)
V_min = f.Constant(name="V_min", value=22, units="m/s", description="takeoff speed")
W_0 = f.Constant(
    name="W_0", value=4940.0, units="N", description="aircraft weight excluding wing"
)
W_W_coeff1 = f.Constant(
    name="W_c1", value=8.71e-5, units="1/m", description="Wing Weight Coefficient 1"
)
W_W_coeff2 = f.Constant(
    name="W_c2", value=45.24, units="Pa", description="Wing Weight Coefficient 2"
)

# =====================
# Declare the Objective
# =====================
f.Objective(D)

# ===================================
# Declare some intermediate variables
# ===================================
pi = np.pi
C_D_fuse = CDA0 / S
C_D_wpar = k * C_f * S_wetratio
C_D_ind = C_L**2 / (pi * A * e)
W_w_strc = W_W_coeff1 * (N_ult * A**1.5 * (W_0 * W * S) ** 0.5) / tau
W_w_surf = W_W_coeff2 * S

# =======================
# Declare the Constraints
# =======================
f.ConstraintList(
    [
        C_D >= C_D_fuse + C_D_wpar + C_D_ind,
        W_w >= W_w_surf + W_w_strc,
        D >= 0.5 * rho * S * C_D * V**2,
        Re == (rho / mu) * V * (S / A) ** 0.5,
        C_f == 0.074 / Re**0.2,
        W == 0.5 * rho * S * C_L * V**2,
        W == 0.5 * rho * S * C_Lmax * V_min**2,
        W >= W_0 + W_w,
    ]
)
