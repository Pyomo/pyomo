# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
from pyomo.environ import *

from pyomo.core.base import _ConstraintData, _ObjectiveData, _ExpressionData
from pyomo.core.base.misc import sorted_robust
from pyomo.core.expr.current import (clone_expression, identify_variables, 
                                     ExpressionReplacementVisitor)

from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory



def sipopt(instance,paramSubList,perturbList,cloneModel=True,
					     streamSoln=False,
					     keepfiles=False):
    """
    This function accepts a Pyomo ConcreteModel, a list of parameters, along
    with their corresponding perterbation list. The model is then converted
    into the design structure required to call sipopt to get an approximation
    perturbed solution with updated bounds on the decision variable. 
    
    Arguments:
        instance     : ConcreteModel: Expectation No Exceptions
            pyomo model object

        paramSubList : Param         
            list of mutable parameters
            Exception : "paramSubList argument is expecting a List of Params"	    

        perturbList  : Param	    
            list of perturbed parameter values
            Exception : "perturbList argument is expecting a List of Params"

            length(paramSubList) must equal length(perturbList)
            Exception : "paramSubList will not map to perturbList"  


        cloneModel   : boolean      : default=True	    
            indicator to clone the model
                -if set to False, the original model will be altered

        streamSoln   : boolean      : default=False	    
            indicator to stream IPOPT solution

        keepfiles    : boolean	    : default=False 
            indicator to print intermediate file names
    
    Returns:
        m		  : ConcreteModel
            converted model for sipopt

        m.sol_state_1     : Suffix	  
            approximated results at perturbation

        m.sol_state_1_z_L : Suffix        
            updated lower bound

        m.sol_state_1_z_U : Suffix        
            updated upper bound
    """

    #Verify User Inputs    
    if len(paramSubList)!=len(perturbList):
        raise ValueError("Length of paramSubList argument does not equal "
                        "length of perturbList")

    for pp in paramSubList:
        if pp.type() is not Param:
            raise ValueError("paramSubList argument is expecting a list of Params")

    for pp in paramSubList:
        if not pp._mutable:
            raise ValueError("parameters within paramSubList must be mutable")  

        
    for pp in perturbList:
        if pp.type() is not Param:
            raise ValueError("perturbList argument is expecting a list of Params")
    #Add model block to compartmentalize all sipopt data
    b=Block()
    block_name = unique_component_name(instance, '_sipopt_data')
    instance.add_component(block_name, b)

    #Based on user input clone model or use orignal model for anlaysis
    if cloneModel:
        b.tmp_lists = (paramSubList, perturbList)
        m = instance.clone()
        instance.del_component(block_name)
        b = getattr(m, block_name)
        paramSubList, perturbList = b.tmp_lists
        del b.tmp_lists
    else:
        m = instance
    
    #Generate component maps for associating Variables to perturbations
    varSubList = []
    for parameter in paramSubList:
        tempName = unique_component_name(b,parameter.local_name)
        b.add_component(tempName,Var(parameter.index_set()))
        myVar = b.component(tempName)
        varSubList.append(myVar)
 
    #Note: substitutions are not currently compatible with 
    #      ComponentMap [ECSA 2018/11/23], this relates to Issue #755
    paramCompMap = ComponentMap(zip(paramSubList, varSubList))
    variableSubMap = {}
    #variableSubMap = ComponentMap()
    paramPerturbMap = ComponentMap(zip(paramSubList,perturbList))
    perturbSubMap = {}
    #perturbSubMap = ComponentMap()
   
    paramDataList = [] 
    for parameter in paramSubList:
        # Loop over each ParamData in the Param Component
        #
        # Note: Sets are unordered in Pyomo.  For this to be
        # deterministic, we need to sort the index (otherwise, the
        # ordering of things in the paramDataList may change).  We use
        # sorted_robust to guard against mixed-type Sets in Python 3.x
        for kk in sorted_robust(parameter):
            variableSubMap[id(parameter[kk])]=paramCompMap[parameter][kk]
            perturbSubMap[id(parameter[kk])]=paramPerturbMap[parameter][kk]
            paramDataList.append(parameter[kk])

    #clone Objective, add to Block, and update any Expressions
    for cc in list(m.component_data_objects(Objective,
                                            active=True,
                                            descend_into=True)):
        tempName=unique_component_name(m,cc.local_name)    
        b.add_component(tempName,
                  Objective(expr=ExpressionReplacementVisitor(
                  substitute=variableSubMap,
                  remove_named_expressions=True).dfs_postorder_stack(cc.expr)))
        cc.deactivate()
    
    #clone Constraints, add to Block, and update any Expressions
    b.constList = ConstraintList()
    for cc in list(m.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
        if cc.equality:
            b.constList.add(expr= ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
        else:
            try:
                b.constList.add(expr=ExpresssionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
            except:
                # Params in either the upper or lower bounds of a ranged 
                # inequaltiy will result in an invalid expression (you cannot 
                # have variables in the bounds of a constraint).  If we hit that
                # problem, we will break up the ranged inequality into separate
                # constraints

                # Note that the test for lower / upper == None is probably not
                # necessary, as the only way we should get here (especially if
                # we are more careful about the exception that we catch) is if
                # this is a ranged inequality and we are attempting to insert a
                # variable into either the lower or upper bound.
                if cc.lower is not None:
                    b.constList.add(expr=ExpressionReplacementVisitor(
                        substitute=variableSubMap,
                        remove_named_expressions=True).dfs_postorder_stack(
                            cc.lower) <= ExpressionReplacementVisitor(
                                substitute=variableSubMap,
                                remove_named_expressions=
                                  True).dfs_postorder_stack(cc.body)
                                )
                #if cc.lower is not None:
                #    b.constList.add(expr=0<=ExpressionReplacementVisitor(
                #        substitute=variableSubMap,
                #        remove_named_expressions=True).dfs_postorder_stack(
                #            cc.lower) - ExpressionReplacementVisitor(
                #                substitute=variableSubMap,
                #                remove_named_expressions=
                #                  True).dfs_postorder_stack(cc.body)
                #                )
                if cc.upper is not None:
                    b.constList.add(expr=ExpressionReplacementVisitor(
                        substitute=variableSubMap,
                        remove_named_expressions=True).dfs_postorder_stack(
                            cc.upper) >= ExpressionReplacementVisitor(
                                substitute=variableSubMap,
                                remove_named_expressions=
                                  True).dfs_postorder_stack(cc.body)
                                )
        cc.deactivate()

    #paramData to varData constraint list
    b.paramConst = ConstraintList()
    for ii in paramDataList:
        jj=variableSubMap[id(ii)]
        b.paramConst.add(ii==jj)

    
    #Create the ipopt_sens (aka sIPOPT) solver plugin using the ASL interface
    opt = SolverFactory('ipopt_sens', solver_io='nl')

    if not opt.available(False):
        raise ImportError('ipopt_sens is not available')
    
    #Declare Suffixes
    m.sens_state_0 = Suffix(direction=Suffix.EXPORT)
    m.sens_state_1 = Suffix(direction=Suffix.EXPORT)
    m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT)
    m.sens_init_constr = Suffix(direction=Suffix.EXPORT)
    
    m.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT)
    m.sens_sol_state_1_z_L = Suffix(direction=Suffix.IMPORT)
    m.sens_sol_state_1_z_U = Suffix(direction=Suffix.IMPORT)

    #set sIPOPT data
    opt.options['run_sens'] = 'yes'
    
    # for reasons that are not entirely clear, 
    #     ipopt_sens requires the indices to start at 1
    kk=1
    for ii in paramDataList:
        m.sens_state_0[variableSubMap[id(ii)]] = kk
        m.sens_state_1[variableSubMap[id(ii)]] = kk
        m.sens_state_value_1[variableSubMap[id(ii)]] = \
                                                   value(perturbSubMap[id(ii)])
        m.sens_init_constr[b.paramConst[kk]] = kk
        kk += 1    
    
    
    #Send the model to the ipopt_sens and collect the solution
    results = opt.solve(m, keepfiles=keepfiles, tee=streamSoln)

    return m
