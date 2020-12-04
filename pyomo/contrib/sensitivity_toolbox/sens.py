# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
from pyomo.environ import Param, Var, Block, ComponentMap, Objective, Constraint, ConstraintList, Suffix, value

from pyomo.core.base.misc import sorted_robust
from pyomo.core.expr.current import ExpressionReplacementVisitor

from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.opt import SolverFactory
import logging

logger = logging.getLogger('pyomo.contrib.sensitivity_toolbox')

@deprecated("The sipopt function has been deprecated. Use the sensitivity_calculation() "
            "function with method='sipopt' to access this functionality.",
            logger='pyomo.contrib.sensitivity_toolbox',
            version='TBD')
def sipopt(instance, paramSubList, perturbList,
           cloneModel=True, tee=False, keepfiles=False):    
    m = sensitivity_calculation('sipopt', instance, paramSubList, perturbList,
         cloneModel, tee, keepfiles, solver_options=None)

    return m

@deprecated("The kaug function has been deprecated. Use the sensitivity_calculation() "
            "function with method='kaug' to access this functionality.", 
            logger='pyomo.contrib.sensitivity_toolbox',
            version='TBD')
def kaug(instance, paramSubList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    m = sensitivity_calculation('kaug', instance, paramSubList, perturbList,
         cloneModel, tee, keepfiles, solver_options)

    return m

def sensitivity_calculation(method, instance, paramSubList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    """This function accepts a Pyomo ConcreteModel, a list of 
    parameters, along with their corresponding perturbation list. The model
    is then converted into the design structure required to call sipopt or
    kaug dsdp mode to get an approximate perturbed solution with updated 
    bounds on the decision variable.
    
    Parameters
    ----------
    method: string
        string of method either 'sipopt' or 'kaug'
        
    instance: ConcreteModel
        pyomo model object

    paramSubList: list
        list of mutable parameters

    perturbList: list
        list of perturbed parameter values

    cloneModel: bool, optional
        indicator to clone the model. If set to False, the original
        model will be altered

    tee: bool, optional
        indicator to stream IPOPT solution

    keepfiles: bool, optional
        preserve solver interface files
            
    solver_options : dictionary, optional
        ipopt solver options dictionary object for method = 'kaug' (default=None)
    
    Returns
    -------
    model: ConcreteModel
        if method is 'sipopt',
        The model is modified for use with sipopt.  The returned model has
            three :class:`Suffix` members defined:

        - ``model.sol_state_1``: the approximated results at the
          perturbation point
        - ``model.sol_state_1_z_L``: the updated lower bound
        - ``model.sol_state_1_z_U``: the updated upper bound
        
        if method is 'kaug',
        The model is modified for use with kaug.  
        The model contains the approximated results at the perturbation point
        
    Raises
    ------
    ValueError
        method argument should be either 'sipopt' or 'kaug'
    ValueError
        perturbList argument is expecting a List of Params
    ValueError
        length(paramSubList) must equal length(perturbList)
    ValueError
        paramSubList will not map to perturbList
    RuntimeError
        ipopt binary must be available
    RuntimeError
        k_aug binary must be available
    RuntimeError
        dotsens binary must be available
    Exception
        kaug does not support inequality constraints
    """
    # Verify User Inputs    
    if method == 'sipopt':
        ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
        if not ipopt_sens.available(False):
            raise RuntimeError('ipopt_sens is not available')
        # set sIPOPT data
        ipopt_sens.options['run_sens'] = 'yes'
    elif method == 'kaug':
        ipopt = SolverFactory('ipopt',solver_io='nl')
        if solver_options is not None:
            ipopt.options = solver_options
        kaug = SolverFactory('k_aug',solver_io='nl')
        dotsens = SolverFactory('dot_sens',solver_io='nl')
        if not ipopt.available(False):
            raise RuntimeError('ipopt is not available')
        if not kaug.available(False):
            raise RuntimeError('k_aug is not available')
        if not dotsens.available(False):
            raise RuntimeError('dotsens is not available')
        
        for cc in list(instance.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
            if not cc.equality:
                raise Exception('kaug does not support inequality constraints.')   
    else:
        raise ValueError("method should be 'sipopt' or 'kaug'")
        
    if len(paramSubList)!=len(perturbList):
        raise ValueError("Length of paramSubList argument does not equal "
                        "length of perturbList")

    for pp in paramSubList:
        if pp.ctype is not Param:
            raise ValueError("paramSubList argument is expecting a list of Params")

    for pp in paramSubList:
        if not pp._mutable:
            raise ValueError("parameters within paramSubList must be mutable")  
        
    for pp in perturbList:
        if pp.ctype is not Param:
            raise ValueError("perturbList argument is expecting a list of Params")
          
    # Add model block to compartmentalize all sipopt data
    b=Block()
    block_name = unique_component_name(instance, '_'+method+'_data')
    instance.add_component(block_name, b)

    # Based on user input clone model or use orignal model for analysis
    if cloneModel:
        b.tmp_lists = (paramSubList, perturbList)
        m = instance.clone()
        instance.del_component(block_name)
        b = getattr(m, block_name)
        paramSubList, perturbList = b.tmp_lists
        del b.tmp_lists
    else:
        m = instance
    
    # Generate component maps for associating Variables to perturbations
    varSubList = []
    for parameter in paramSubList:
        tempName = unique_component_name(b,parameter.local_name)
        # initialize variable with the nominal value
        if parameter.is_indexed():
            d = {k: value(parameter[k]) for k in parameter.index_set()}
        else:
            d = value(parameter)
        b.add_component(tempName,Var(parameter.index_set(),initialize=d)) 
        myVar = b.component(tempName)
        varSubList.append(myVar)
 
    # Note: substitutions are not currently compatible with 
    #      ComponentMap [ECSA 2018/11/23], this relates to Issue #755
    paramCompMap = ComponentMap(zip(paramSubList, varSubList))
    variableSubMap = {}
    # variableSubMap = ComponentMap()
    paramPerturbMap = ComponentMap(zip(paramSubList,perturbList))
    perturbSubMap = {}
    # perturbSubMap = ComponentMap()
   
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

    # clone Objective, add to Block, and update any Expressions
    for cc in list(m.component_data_objects(Objective,
                                            active=True,
                                            descend_into=True)):
        tempName=unique_component_name(m,cc.local_name)    
        b.add_component(tempName,
                  Objective(expr=ExpressionReplacementVisitor(
                  substitute=variableSubMap,
                  remove_named_expressions=True).dfs_postorder_stack(cc.expr)))
        cc.deactivate()
    
    # clone Constraints, add to Block, and update any Expressions
    b.constList = ConstraintList()
    for cc in list(m.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
        if cc.equality:
            b.constList.add(expr= ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
        else:
            if cc.lower is None or cc.upper is None:
                b.constList.add(expr=ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
            else:
                # Constraint must be a ranged inequality, break into separate constraints

                # Add constraint for lower bound
                b.constList.add(expr=ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(
                        cc.lower) <= ExpressionReplacementVisitor(
                            substitute=variableSubMap,
                            remove_named_expressions=
                            True).dfs_postorder_stack(cc.body)
                    )
                # Add constraint for upper bound
                b.constList.add(expr=ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(
                        cc.upper) >= ExpressionReplacementVisitor(
                            substitute=variableSubMap,
                            remove_named_expressions=
                            True).dfs_postorder_stack(cc.body)
                    )
        cc.deactivate()

    # paramData to varData constraint list
    b.paramConst = ConstraintList()
    for ii in paramDataList:
        jj=variableSubMap[id(ii)]
        b.paramConst.add(ii==jj)
        
    # Declare Suffixes
    m.sens_state_0 = Suffix(direction=Suffix.EXPORT)
    m.sens_state_1 = Suffix(direction=Suffix.EXPORT)
    m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT)
    m.sens_init_constr = Suffix(direction=Suffix.EXPORT)
    
    m.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT)
    m.sens_sol_state_1_z_L = Suffix(direction=Suffix.IMPORT)
    m.sens_sol_state_1_z_U = Suffix(direction=Suffix.IMPORT)
    
    
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
    
    if method == 'sipopt':
        # Send the model to the ipopt_sens and collect the solution
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)
    elif method == 'kaug':
        kaug_suffix = {'ipopt_zL_out':'IMPORT','ipopt_zU_out':'IMPORT', 
                       'ipopt_zL_in':'EXPORT','ipopt_zU_in':'EXPORT',
                        'dcdp':'EXPORT','DeltaP':'EXPORT'}
        for _suffix in kaug_suffix.keys():
            _add_kaug_suffix(m, _suffix, kaug_suffix[_suffix])
            
        kk = 1
        for ii in paramDataList:
            m.dcdp[b.paramConst[kk]] = kk
            m.DeltaP[b.paramConst[kk]] = value(ii)-value(perturbSubMap[id(ii)])
            kk += 1
        
                    
        logger.info("ipopt starts")
        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)  #: important!
        m.ipopt_zU_in.update(m.ipopt_zU_out)  #: important!    
        logger.debug("ipopt completed")
        #: k_aug
        logger.info("k_aug starts")
        kaug.options['dsdp_mode'] = ""  #: sensitivity mode!
        kaug.solve(m, tee=tee)
        logger.debug("k_aug completed")
        dotsens.options["dsdp_mode"] = ""
        logger.info("dotsens starts")
        dotsens.solve(m, tee=tee) 
        logger.debug("dotsens completed")
                
    return m        

def _add_kaug_suffix(model, suffix_name, _direction):
    # _add_kaug_suffix checks the model to see if suffix_name already exists.
    # It adds suffix_name to the model for a given direction '_direction'.
    suffix_checker = model.component(suffix_name)
    if _direction == 'IMPORT':
        if suffix_checker is None:
            setattr(model, suffix_name, Suffix(direction=Suffix.IMPORT))
        else:
            model.del_component(suffix_name)
            setattr(model, suffix_name, Suffix(direction=Suffix.IMPORT))
    elif _direction == 'EXPORT':
        if suffix_checker is None:
            setattr(model, suffix_name, Suffix(direction=Suffix.EXPORT))
        else:
            model.del_component(suffix_name)
            setattr(model, suffix_name, Suffix(direction=Suffix.EXPORT))
    else:
        raise ValueError("_direction argument should be 'IMPORT' or 'EXPORT'")