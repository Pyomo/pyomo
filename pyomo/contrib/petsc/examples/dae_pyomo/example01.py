"""
A test problem from https://archimede.dm.uniba.it/~testset/report/chemakzo.pdf
"""
from __future__ import division  # No integer division
from __future__ import print_function  # Python 3 style print

from pyomo.environ import *
from pyomo.opt import SolverFactory
import os

if __name__ == "__main__":
    opt = SolverFactory('petsc')
    model = ConcreteModel(name="example01")

    # Set problem parameter values
    model.k = Param([1,2,3,4], initialize={
        1:18.7,
        2:0.58,
        3:0.09,
        4:0.42})
    model.Ke = Param(initialize=34.4)
    model.klA = Param(initialize=3.3)
    model.Ks = Param(initialize=115.83)
    model.pCO2 = Param(initialize=0.9)
    model.H = Param(initialize=737)

    # Problem variables ydot = dy/dt,
    #    (dy6/dt is not explicitly in the equations, so only 5 ydots)
    model.t = Var(initialize=0) #time, but not used anywhere could delete
                                #using time explicilty in equations is possible
    model.y = Var([1,2,3,4,5,6])  #
    model.ydot = Var([1,2,3,4,5]) # dy/dt
    model.r = Var([1,2,3,4,5])
    model.Fin = Var()

    # Equations
    model.eq_ydot1 = Constraint(expr=model.ydot[1] == -2.0*model.r[1] +
        model.r[2] - model.r[3] - model.r[4])
    model.eq_ydot2 = Constraint(expr=model.ydot[2] == -0.5*model.r[1] -
        model.r[4] - 0.5*model.r[5] + model.Fin)
    model.eq_ydot3 = Constraint(expr=model.ydot[3] == model.r[1] -
        model.r[2] + model.r[3])
    model.eq_ydot4 = Constraint(expr=model.ydot[4] == -model.r[2] + model.r[3] -
        -2.0*model.r[4])
    model.eq_ydot5 = Constraint(expr=model.ydot[5] == model.r[2] - model.r[3] +
        model.r[5])
    model.eq_y6 = Constraint(expr=0 == model.Ks*model.y[1]*model.y[4] -
        model.y[6])

    model.eq_r1 = Constraint(
        expr=model.r[1] == model.k[1]*model.y[1]**4*model.y[2]**0.5)
    model.eq_r2 = Constraint(
        expr=model.r[2] == model.k[2]*model.y[3]*model.y[4])
    model.eq_r3 = Constraint(
        expr=model.r[3] == model.k[2]/model.Ke*model.y[1]*model.y[5])
    model.eq_r4 = Constraint(
        expr=model.r[4] == model.k[3]*model.y[1]*model.y[4]**2)
    model.eq_r5 = Constraint(
        expr=model.r[5] == model.k[4]*model.y[6]**2*model.y[2]**0.5)
    model.eq_Fin = Constraint(
        expr=model.Fin == model.klA*(model.pCO2/model.H - model.y[2]))

    # Set initial condtions and solve initial from the values of differntial
    # variables (r and y6 well and the derivative vars too).
    y0 = {1:0.444, 2:0.00123, 3:0.0, 4:0.007, 5:0.0} #initial differntial vars
    for i in [1,2,3,4,5]: model.y[i].fix(y0[i])
    print("Solving initial:")
    res = opt.solve(model, tee=True, options={"-snes_monitor":""})
    for i in [1,2,3,4,5]: model.y[i].unfix()
    model.display() # show the initial state

    #Set suffixes to show the structure of the problem
    # dae_suffix holds variable types 0=algebraic 1=differential 2=derivative
    # 3=time. dae_link associates differential variables to their derivatives
    model.dae_suffix = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    model.dae_link = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    # Label the vars.  Seems 0 is default if I don't attach a suffix so don't
    # need to explicitly label algebraic vars
    model.dae_suffix[model.t] = 3 # this labels t as the time variables
    for i in [1,2,3,4,5]:
        model.dae_suffix[model.y[i]] = 1 #differential vars
        model.dae_suffix[model.ydot[i]] = 2 #derivative vars
        #link the differential vars to the derivative var
        model.dae_link[model.y[i]] = i
        model.dae_link[model.ydot[i]] = i

    print ("Solving DAE:")
    # Solve threw in a lot of example options, but the important ones are
    # the final time and the one to specify that it's a dae.  Probably also
    # would want to specify a time step, or an adaptive time stepping method
    res = opt.solve(model, tee=True,
        options={
            "-dae_solve":"",              #tell solver to expect dae problem
            "-ts_monitor":"",             #show progess of TS solver
            "-ts_max_snes_failures":10,   #max nonlin solve fails before give up
            #"-ts_type":"sundials",       #ts solver
            "-ts_type":"alpha",           #ts_solver
            "-snes_monitor":"",           #show progress on nonlinear solves
            "-pc_type":"lu",              #direct solve MUMPS default LU fact
            "-ksp_type":"preonly",        #no ksp used direct solve preconditioner
            "-scale_vars":"0",            #currently not compatable dae
            "-scale_eqs":"0",             #should work, but prob don't need
            "-snes_type":"newtonls",      # newton line search for nonliner solver
            "-ts_adapt_type":"basic",       # there are a lot of adaptive step
            "-ts_max_time":180,           # final time
            "-ts_save_trajectory":1,
            "-ts_trajectory_type":"visualization",
            #"-ts_exact_final_time":"stepover",
            #"-ts_exact_final_time":"matchstep",
            "-ts_exact_final_time":"interpolate",
            #"-ts_view":""
            })
    model.display() # display final state of the model (at -ts_max_time)
