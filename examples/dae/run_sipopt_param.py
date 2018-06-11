from pyomo.environ import *
from pyomo.core import *
from pyomo.core.base import _ConstraintData
from pyomo.core.base.expr import clone_expression
from pyomo.opt import SolverFactory

from param import m

#-------------------------------------------#
#		Model Translation
#-------------------------------------------#

#add variable components for identified parameters
#Parameters must be mutable
m.add_component(m.eta1.local_name+'_var',Var())
m.add_component(m.eta2.local_name+'_var',Var())

#Param substitution map
paramsublist = [m.eta1, m.eta2]
paramsubmap = {id(m.eta1):m.eta1_var, id(m.eta2):m.eta2_var}

#Loop through components to build substitution map
variable_sub_map = {}
for parameter in paramsublist:
    #Loop over each ParamData in the parameter (will this work on sparse params?)
    for kk in parameter:
        p = parameter[kk]
        print(kk,p)
        variable_sub_map[id(p)] = paramsubmap[id(parameter)][kk]

#substitusion the Constraint
for component in m.component_objects(ctype=(Constraint), descend_into=True):
    for kk in component:
        c=component[kk]
        print(c.name)
        if isinstance(c, _ConstraintData):
            #Have to be really careful with upper and lower bounds of a constraint
            c._body = clone_expression(c._body, substitute=variable_sub_map)

#add constraint, based on parameter values orginially passed with model
m.add_component(m.eta1.local_name+'_con',Constraint(expr=m.eta1_var==m.eta1))
m.add_component(m.eta2.local_name+'_con',Constraint(expr=m.eta2_var==m.eta2))

m.pprint()

#------------------------------------------#
#		sIPOPT
#------------------------------------------#

#Create the ipopt_sense (aka sIPOPT) solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = False  	#True prints solver output to screen
keepfiles = False 	#True prints intermediate file names (.nl, .sol, ....)
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
m.sens_state_0[m.eta1_var] = 1
m.sens_state_1[m.eta1_var] = 1
m.sens_state_value_1[m.eta1_var] = 4.0
m.sens_state_0[m.eta2_var] = 2
m.sens_state_1[m.eta2_var] = 2
m.sens_state_value_1[m.eta2_var] = 1.0
m.sens_init_constr[m.eta1_con] = 1
m.sens_init_constr[m.eta2_con] = 2

#Send the modle to ipopt_sens and collect the solution
results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)

#print solution
print("nominal and perturbed solution:")
for vv in [m.x1, m.x2, m.x3, m.eta1_var, m.eta2_var]:
    print("%5s %14g %14g" % (vv, value(vv), m.sens_sol_state_1[vv]))




