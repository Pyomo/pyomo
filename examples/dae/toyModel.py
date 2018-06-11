from pyomo.environ import *
from pyomo.core.base.expr import clone_expression, identify_variables

m = ConcreteModel()
m.x = Var()
m.y = Var()
m.z = Var()
m.p = Param(initialize=5,mutable=True)
m.p_var=Var()
m.q = Param(initialize=6)
m.C1 = Constraint(expr=m.x+m.y<=m.p*m.q)
m.C2 = Constraint(expr=5*m.x==m.q)
m.C3 = Constraint(expr=5<=m.p*m.y<=10)
                             
varSubList=[m.p_var]         
paramSubList=[m.p]           
                             
                             
paramCompMap = ComponentMap(zip(paramSubList,varSubList))
                             
variableSubMap={}            
for parameter in paramSubList:
    for kk in parameter:     
        variableSubMap[id(parameter[kk])]=paramCompMap[parameter][kk]
#variableSubMap = {}         
#variableSubMap[id(m.p)]=paramCompMap[m.p][m.z]
                             
m.add_component(m.C1.local_name+'_test',
		Constraint(expr=clone_expression(m.C1.expr,
                                                 substitute=variableSubMap)))
m.add_component(m.C3.local_name+'_test',
                Constraint(expr=clone_expression(m.C3.lower,
                                                 substitute=variableSubMap)
                              <=clone_expression(m.C3.body,
                                                 substitute=variableSubMap)
                              <=clone_expression(m.C3.upper,
                                                 substitute=variableSubMap)))
                             
                             
                             
m.pprint()                   
                             
                            
                            
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
