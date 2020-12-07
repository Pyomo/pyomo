#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Sample Problem 3: Inequality State Path Constraint
# (Ex 4 from Dynopt Guide)
#
#   min x3(tf)
#   s.t.    X1_dot = X2                     X1(0) =  0
#           X2_dot = -X2+u                  X2(0) = -1
#           X3_dot = X1^2+x2^2+0.005*u^2    X3(0) =  0
#           X2-8*(t-0.5)^2+0.5 <= 0
#           tf = 1
#

# Modified for no optimization, and PETSc solver

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()
m.tf = Param(initialize=1)
m.t = Var() #ContinuousSet(bounds=(0,m.tf))

m.u = Var(initialize=0) #Var(m.t, initialize=0)
m.x1 = Var() #Var(m.t)
m.x2 = Var() #Var(m.t)
m.x3 = Var() #Var(m.t)

m.dx1 = Var() #DerivativeVar(m.x1, wrt=m.t)
m.dx2 = Var() #DerivativeVar(m.x2, wrt=m.t)
m.dx3 = Var() #DerivativeVar(m.x3, wrt=m.t)

#m.obj = Objective(expr=m.x3[m.tf])

#def _x1dot(m, t):
#    if t == 0:
#        return Constraint.Skip
#    return m.dx1[t] == m.x2[t]
m.x1dotcon =  Constraint(expr=m.dx1 == m.x2) #Constraint(m.t, rule=_x1dot)

#def _x2dot(m, t):
#    if t == 0:
#        return Constraint.Skip
#    return m.dx2[t] ==  -m.x2[t]+m.u[t]
m.x2dotcon =  Constraint(expr=m.dx2 ==  -m.x2+m.u)#Constraint(m.t, rule=_x2dot)

#def _x3dot(m, t):
#    if t == 0:
#        return Constraint.Skip
#    return m.dx3[t] == m.x1[t]**2+m.x2[t]**2+0.005*m.u[t]**2
#m.x3dotcon = Constraint(m.t, rule=_x3dot)
m.x3dotcon = Constraint(expr=m.dx3 == m.x1**2+m.x2**2+0.005*m.u**2)

# Found a hole in the DAE solver, x3 isn't in the constraints so its not in the
# nl file.  Added this constraint to force it in
m.xdummy = Var()
m.dummy = Constraint(expr=m.x3 == m.xdummy)

#def _con(m, t):
#    return m.x2[t]-8*(t-0.5)**2+0.5 <= 0
#m.con = Constraint(m.t, rule=_con)

#def _init(m):
#    yield m.x1[0] == 0
#    yield m.x2[0] == -1
#    yield m.x3[0] == 0
#m.init_conditions = ConstraintList(rule=_init)

m.x1.value = 0
m.x2.value = -1
m.x3.value = 0
m.u.fix()

#Set suffixes to show the structure of the problem
# dae_suffix holds variable types 0=algebraic 1=differential 2=derivative
# 3=time. dae_link associates differential variables to their derivatives
m.dae_suffix = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
m.dae_link = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
m.dae_suffix[m.t] = 3 # this labels t as the time variables (not really needed here)
m.dae_suffix[m.x1] = 1
m.dae_suffix[m.x2] = 1
m.dae_suffix[m.x3] = 1
m.dae_suffix[m.dx1] = 2
m.dae_suffix[m.dx2] = 2
m.dae_suffix[m.dx3] = 2
# Link the differential vars to the derivatives
m.dae_link[m.x1] = 1
m.dae_link[m.x2] = 2
m.dae_link[m.x3] = 3
m.dae_link[m.dx1] = 1
m.dae_link[m.dx2] = 2
m.dae_link[m.dx3] = 3

# Usually would solve initial condition first, but don't need to here
opt = SolverFactory('petsc')
res = opt.solve(m, tee=True,
    options={
        "-dae_solve":"",             #tell solver to expect dae problem
        "-ts_monitor":"",            #show progess of TS solver
        "-ts_max_snes_failures":40,  #max nonlin solve fails before give up
        "-ts_max_reject":20,         #max steps to reject
        "-ts_type":"alpha",          #ts_solver
        "-snes_monitor":"",          #show progress on nonlinear solves
        "-pc_type":"lu",             #direct solve MUMPS default LU fact
        "-ksp_type":"preonly",       #no ksp used direct solve preconditioner
        "-scale_vars":0,             #variable scaling method
        "-scale_eqs":1,              #equation scaling method
        "-snes_type":"newtonls",     # newton line search for nonliner solver
        "-ts_adapt_type":"basic",
        "-ts_max_time":2,            # final time
        "-ts_save_trajectory":1,
        "-ts_trajectory_type":"visualization",
        #"-ts_exact_final_time":"stepover",
        #"-ts_exact_final_time":"matchstep",
        "-ts_exact_final_time":"interpolate",
        #"-ts_view":""
        })
