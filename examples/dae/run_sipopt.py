# ___________________________________________________________________________
#    
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and 
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ___________________________________________________________________________

# Author: Erin Acquesta, 2018-02-14
# 
# This pyomo function is designed to accept the following inputs
#   m : pyomo model object
#   p : list of parameters
#   eps : list of values to perturbed p by. 
#           Note: length(eps) must equal length(p)  
# 


from pyomo.environ import *
from pyomo.core import *
from pyomo.core.base import _ConstraintData, _ObjectiveData, _ExpressionData
from pyomo.core.base.expr import clone_expression, identify_variables
from pyomo.opt import SolverFactory

from HIV_Transmission_discrete import m


#------------------------------------------#
#       Model Translations
#------------------------------------------#

m.b=Block()

#add variable components for identified parameters
#Parameters must be mutable
m.b.add_component(m.aa.local_name+'_var',Var())
m.b.add_component(m.eps.local_name+'_var',Var())
m.b.add_component(m.qq.local_name+'_var',Var(m.qq._data.keys()))


#----------------------------------------------------------#
# We should consider cloning the whole model. 
#	- current code deactivates user's Objective
#	 and Constraints. 
#	- substitutions with Expressions are problematic
#	because it would require another type of 
#	substitution.
#	- If we clone the whole model we can muck up 
#	whatever we want to and leave the user's model in
#	original form.
#----------------------------------------------------------#


#Param substitution map
varSubList = [m.b.aa_var, m.b.eps_var, m.b.qq_var]
paramSubList = [m.aa,m.eps,m.qq]
#varSubList = [m.b.aa_var]
#paramSubList = [m.aa]
paramCompMap = ComponentMap(zip(paramSubList, varSubList))
expParamList = ComponentMap(zip(paramSubList, paramSubList))

#Loop through components to build substiution map
variableSubMap = {}
paramDict = {}
#import pdb;pdb.set_trace()
for parameter in paramSubList:
    #Loop over each ParamData in the paramter (will this work on sparse params?)
    for kk in parameter:
        variableSubMap[id(parameter[kk])]=paramCompMap[parameter][kk]
        #variableSubMap[id(pp)] = paramCompMap[pp][kk]
        paramDict[id(parameter[kk])]=expParamList[parameter][kk]

#----------------------------------------------------------#
#Objective, Expression, and Constraints will ALL be cloned
#	regardless if needed.
#	Need to consider an efficient way to handle cloning
#	only when needed.
#----------------------------------------------------------#


#substitute the Objectives
for cc in list(m.component_data_objects(Objective,
                                        active=True,
                                        descend_into=True)):
    m.b.add_component(cc.local_name,
                      Objective(expr=clone_expression(cc.expr,
                                          substitute=variableSubMap)))
    cc.deactivate()

#
#subtitute the Expressions
for cc in list(m.component_data_objects(Expression,
                                   active=True,
                                   descend_into=True)):	
    m.b.add_component(cc.local_name,
                      Expression(expr=
                                clone_expression(cc.expr,
                                                 substitute=variableSubMap)))

    #-------------------------------------------------------------------#
    #Can NOT Deactive Expressions. Need to consider how to handle model 
    #	Expression calls. We need a substitution mechanism or work
    #   around to address the existance fo the original Expression
    #   as well as the cloned Expression. 
    #-------------------------------------------------------------------#

#substitue the Constraints while using a constraint list
#m.b.conlist=ConstraintList()
    #---------------------------------------#
    #  Currenlty not using the constraint
    #  list for constraint substitutions
    #  *****should ask about this*****
    #---------------------------------------#
for cc in list(m.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
    if cc.equality:
        m.b.add_component(cc.local_name,
                          Constraint(expr=
                                     clone_expression(cc.expr,
                                                   substitute=variableSubMap)))
    else:
        if cc.lower==None or cc.upper==None:
            m.b.add_component(cc.local_name,
                              Constraint(expr=
                                         clone_expression(cc.expr,
                                                   substitute=variableSubMap)))
        else:
            m.b.add_component(cc.local_name,
                              Constraint(expr=
                                         clone_expression(cc.lower,
                                                   substitute=variableSubMap)
                                         <=clone_expression(cc.body,
                                                   substitute=variableSubMap)
                                         <=clone_expression(cc.upper,
                                                   substitute_variableSubMap)))
        
    cc.deactivate()


#-----------------------------------------------------#
#
#  Need to figure out how to loop through a range
#  index and still call the appropriate components
#  from the varSubList and paramSubList.
#
#-----------------------------------------------------#

#m.b.paramConst = Constraint(range(len(variableSubMap.keys())))
#for ii,jj in variableSubMap.items():
#    import pdb;pdb.set_trace()
#    m.b.paramConst[ii]=paramDict[ii].value==jj
#    print(kk)
#    print(ii)
#    print(jj)
#    print('--------')
#    kk += 1
#    #m.b.paramConst=ii==jj
     #
     #

for ii,jj in variableSubMap.items():
    m.b.add_component(jj.local_name+'_con',
                      Constraint(expr=paramDict[ii].value==jj))

#add #constraints, based on parameter values for sIPOPT
##m.a#dd_component(m.aa.local_name+'_con',Constraint(expr=m.aa_var==m.aa))
##m.a#dd_component(m.eps.local_name+'_con',Constraint(expr=m.eps_var==m.eps))
##   #
##def# _qq_vals(m,*args):
##   # return m.qq_var[args]==m.qq[args]
##   #
##m.a#dd_component(m.qq.local_name+'_con',Constraint(m.qq._index,rule=_qq_vals))
#    #
#m.pp#rint()
#    #
##---#-------------------------------------#
##   #        sIPOPT
##---#-------------------------------------#
#    #
##Cre#ate the ipopt_sens (aka sIPOPT) solver plugin using the ASL interface
#solv#er = 'ipopt_sens'
#solv#er_io = 'nl'
#stre#am_solver = True    #True prints solver output to screen
#keep#files = False   #True prints intermediate file names (.nl, .sol, ....)
#opt #= SolverFactory(solver, solver_io=solver_io)
#    #
#if o#pt is None:
#    #print("")
#    #print("ERROR: Unable to create solver plugin for 'ipopt_sens' ")
#    #print("")
#    #exit(1)
#    #
##Dec#lare Suffixes
#m.se#ns_state_0 = Suffix(direction=Suffix.EXPORT)
#m.se#ns_state_1 = Suffix(direction=Suffix.EXPORT)
#m.se#ns_state_value_1 = Suffix(direction=Suffix.EXPORT)
#m.se#ns_sol_state_1 = Suffix(direction=Suffix.IMPORT)
#m.se#ns_init_constr = Suffix(direction=Suffix.EXPORT)
#    #
##set# sIPOPT data
#opt.#options['run_sens'] = 'yes'
#m.se#ns_state_0[m.qq_var[(0,0)]] = 1
##m.s#ens_state_0[m.eps_var] = 2
#m.sens_state_1[m.qq_var[(0,0)]] = 1
##m.sens_state_1[m.eps_var] = 2
#m.sens_state_value_1[m.qq_var[(0,0)]] = 0.9999
##m.sens_state_value_1[m.eps_var] = 0.75
#m.sens_init_constr[m.qq_con[(0,0)]] = 1
##m.sens_init_constr[m.eps_con] = 2
#
##Send the model to the ipopt_sens and collect the solution
#results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)
#
##Print solution
#print("Nominal and perturbed solution:")
#for vv in [m.qq_var[(0,0)],m.L[m.tf]]:
#    print("%5s %14g %14g" % (vv, value(vv), m.sens_sol_state_1[vv]))
#
#
#
#
