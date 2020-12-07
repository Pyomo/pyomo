# Ampl Car Example
#
# Modified pyomo.dae example to remove optimization, and set up PETSc solve
#
# min tf
# dxdt = 0
# dvdt = a-R*v^2
# x(0)=0; x(tf)=L
# v(0)=0; v(tf)=0
# -3<=a<=1

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.R = Param(initialize=0.001) #  Friction factor
m.L = Param(initialize=100.0) #  Final position

m.tau = Var() #ContinuousSet(bounds=(0,1)) # Unscaled time
m.time = Var() # Scaled time
m.x = Var(bounds=(0,m.L+50)) #Var(m.ta,bounds=(0,m.L+50))
m.v = Var(bounds=(0,None)) #Var(m.tau,bounds=(0,None))
m.a = Var(bounds=(-3.0,1.0),initialize=1) #Var(m.tau, bounds=(-3.0,1.0),initialize=0)
m.tf = Var(initialize=1.0)
m.a.fix()
m.tf.fix()

m.dtime = Var() #DerivativeVar(m.time)
m.dx = Var() #DerivativeVar(m.x)
m.dv = Var() #DerivativeVar(m.v)

#m.obj = Objective(expr=m.tf)

#def _ode1(m,i):
#    if i == 0 :
#        return Constraint.Skip
#    return m.dx[i] == m.tf * m.v[i]
#m.ode1 = Constraint(m.tau, rule=_ode1)
m.ode1 = Constraint(expr=m.dx == m.tf * m.v)

#def _ode2(m,i):
#    if i == 0 :
#        return Constraint.Skip
#    return m.dv[i] == m.tf*(m.a[i] - m.R*m.v[i]**2)
#m.ode2 = Constraint(m.tau, rule=_ode2)
m.ode2 = Constraint(expr=m.dv == m.tf*(m.a - m.R*m.v**2))

#def _ode3(m,i):
#    if i == 0:
#        return Constraint.Skip
#    return m.dtime[i] == m.tf
#m.ode3 = Constraint(m.tau, rule=_ode3)
m.ode3 = Constraint(expr=m.dtime == m.tf)

# Found a hole in the DAE solver, x isn't in the constraints so its not in the
# nl file.  Added this constraint to force it in
m.xdummy = Var()
m.timedummy = Var()
m.dummy1 = Constraint(expr=m.x == m.xdummy)
m.dummy2 = Constraint(expr=m.time == m.timedummy)

#def _init(m):
#    yield m.x[0] == 0
#    yield m.x[1] == m.L
#    yield m.v[0] == 0
#    yield m.v[1] == 0
#    yield m.time[0] == 0
#m.initcon = ConstraintList(rule=_init)

m.x.value = 0
m.v.value = 0
m.time = 0

#Set suffixes to show the structure of the problem
# dae_suffix holds variable types 0=algebraic 1=differential 2=derivative
# 3=time. dae_link associates differential variables to their derivatives
m.dae_suffix = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
m.dae_link = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
m.dae_suffix[m.tau] = 3 # this labels t as the time variables (not really needed here)
m.dae_suffix[m.time] = 1
m.dae_suffix[m.x] = 1
m.dae_suffix[m.v] = 1
m.dae_suffix[m.dtime] = 2
m.dae_suffix[m.dx] = 2
m.dae_suffix[m.dv] = 2
# Link the differential vars to the derivatives
m.dae_link[m.time] = 1
m.dae_link[m.x] = 2
m.dae_link[m.v] = 3
m.dae_link[m.dtime] = 1
m.dae_link[m.dx] = 2
m.dae_link[m.dv] = 3


#discretizer = TransformationFactory('dae.finite_difference')
#discretizer.apply_to(m,nfe=15,scheme='BACKWARD')

#solver = SolverFactory('ipopt')
#solver.solve(m,tee=True)
solver = SolverFactory('petsc')
res = solver.solve(m, tee=True,
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
        "-scale_eqs":0,              #equation scaling method
        "-snes_type":"newtonls",     # newton line search for nonliner solver
        "-ts_adapt_type":"basic",
        "-ts_adapt_dt_min":1e-1,
        "-ts_max_time":100,            # final time
        "-ts_save_trajectory":1,
        "-ts_trajectory_type":"visualization",
        #"-ts_exact_final_time":"stepover",
        #"-ts_exact_final_time":"matchstep",
        "-ts_exact_final_time":"interpolate",
        #"-ts_view":""
        })

m.display()
