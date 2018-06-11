# ___________________________________________________________________________
#    
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and 
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ___________________________________________________________________________

# Author: Erin Acquesta, 2017-12-06
# 
# This pyomo script will load the 'HIV_Transmission.py' model, translate a selection
# of the parameters the to variables (with corresponding constraints) then run the
# 'ipopt_sense' solver to get the perturbation solution of the model.  
# 


import pyomo.environ
from pyomo.core import *
from pyomo.core.base import _ConstraintData, _ObjectiveData
from pyomo.core.base.expr import clone_expression
from pyomo.opt import SolverFactory

from HIV_Transmission import m

#------------------------------------------#
#	    Model Translations
#------------------------------------------#


#add variable components for identified parameters
#Parameters must be mutable
m.add_component(m.aa.local_name+'_var',Var())
m.add_component(m.eps.local_name+'_var',Var())
#m.add_component(m.qq.local_name+'_var',Var(m.qq._index))

#Param substitution map
paramsublist = [m.aa, m.eps]
paramsubmap = {id(m.aa):m.aa_var, id(m.eps):m.eps_var}

#Loop through components to build substiution map
variable_sub_map = {}
for parameter in paramsublist:
    #Loop over each ParamData in the paramter (will this work on sparse params?)
    for kk in parameter:
        p = parameter[kk]
        print(kk, p)
        variable_sub_map[id(p)] = paramsubmap[id(parameter)][kk]

#substitute the Objectives/Constraints/Expressions
for component in m.component_objects(ctype=(Constraint,Objective), descend_into=True):
    for kk in component:
        c =component[kk]
        print(c.name)
        if isinstance(c, _ConstraintData):
            #Have to be really careful with upper and lower bounds of a constraint
            c._body = clone_expression(c._body, substitute=variable_sub_map)
        elif isinstance(c, _ObjectiveData):
            c.expr = clone_expression(c.expr, substitute=variable_sub_map)
       # elif isinstance(c, _ExpressionData):
        #    c.expr = clone_expression(c.expr, substitute=variable_sub_map)



#add constraints, based on parameter values for sIPOPT
m.add_component(m.aa.local_name+'_con',Constraint(expr=m.aa_var==m.aa))
m.add_component(m.eps.local_name+'_con',Constraint(expr=m.eps_var==m.eps))
#def _qq_vals(m,*args):
#    return m.qq_var[args]==m.qq[args]

#m.add_component(m.qq.local_name+'_con',Constraint(m.qq._index,rule=_qq_vals))

m.pprint()

#----------------------------------------#
#         	sIPOPT
#----------------------------------------#

#Create the ipopt_sens (aka sIPOPT) solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = False	#True prints solver output to screen
keepfiles = False	#True prints intermediate file names (.nl, .sol, ....)
opt = SolverFactory(solver, solver_io=solver_io)

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for 'ipopt_sens' ")
    print("")
    exit(1)

#Declare Suffixes
m.sens_state_0 = Suffix(direction=Suffix.EXPORT)
m.sens_state_1 = Suffix(direction=Suffix.EXPORT)
m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT)
m.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT)
m.sens_init_constr = Suffix(direction=Suffix.EXPORT)

#set sIPOPT data
opt.options['run_sens'] = 'yes'
m.sens_state_0[m.aa_var] = 1
m.sens_state_0[m.eps_var] = 2
m.sens_state_1[m.aa_var] = 1
m.sens_state_1[m.eps_var] = 2
m.sens_state_value_1[m.aa_var] = 0.00015
m.sens_state_value_1[m.eps_var] = 0.75
m.sens_init_constr[m.aa_con] = 1
m.sens_init_constr[m.eps_con] = 2

#Send the model to the ipopt_sens and collect the solution
results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)

#Print solution
#print("Nominal and perturbed solution:")
#for vv in [m.L[20], m.aa]:
#    print("%5s %14g" % (vv, value(vv), model.sens_sol_state_1[vv]))




