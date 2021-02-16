# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
from pyomo.environ import (
        Param,
        Var,
        Block,
        ComponentMap,
        Objective,
        Constraint,
        ConstraintList,
        Suffix,
        value,
        ComponentUID,
        )

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

_SIPOPT_SUFFIXES = {
        'sens_state_0': Suffix.EXPORT,
        # ^ Not sure what this suffix does -RBP
        'sens_state_1': Suffix.EXPORT,
        'sens_state_value_1': Suffix.EXPORT,
        'sens_init_constr': Suffix.EXPORT,

        'sens_sol_state_1': Suffix.IMPORT,
        'sens_sol_state_1_z_L': Suffix.IMPORT,
        'sens_sol_state_1_z_U': Suffix.IMPORT,
        }

_K_AUG_SUFFIXES = {
        'ipopt_zL_out': Suffix.IMPORT,
        'ipopt_zU_out': Suffix.IMPORT,
        'ipopt_zL_in': Suffix.EXPORT,
        'ipopt_zU_in': Suffix.EXPORT,
        'dcdp': Suffix.EXPORT,
        'DeltaP': Suffix.EXPORT,
        }

def _add_sensitivity_suffixes(block):
    suffix_dict = {}
    suffix_dict.update(_SIPOPT_SUFFIXES)
    suffix_dict.update(_K_AUG_SUFFIXES)
    for name, direction in suffix_dict.items():
        if block.component(name) is None:
            # Only add suffix if it doesn't already exist.
            # If something of this name does already exist, just
            # assume it is the proper suffix and move on.
            block.add_component(name, Suffix(direction=direction))

class _NotAnIndex(object):
    pass

def _generate_data_objects(components):
    if type(components) not in {list, tuple}:
        components = (components,)
    for comp in components:
        if comp.is_indexed():
            for idx in sorted_robust(comp):
                yield comp[idx]
        else:
            yield comp

def _generate_component_items(components):
    if type(components) not in {list, tuple}:
        components = (components,)
    for comp in components:
        if comp.is_indexed():
            for idx in sorted_robust(comp):
                yield idx, comp[idx]
        else:
            yield _NotAnIndex, comp

def _sensitivity_calculation(method, instance, paramList, perturbList,
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
    ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
    ipopt_sens.options['run_sens'] = 'yes'
    kaug = SolverFactory('k_aug', solver_io='nl')
    dotsens = SolverFactory('dot_sens', solver_io='nl')
    ipopt = SolverFactory('ipopt', solver_io='nl')

    m = setup_sensitivity(method, instance, paramList, perturbList,
         cloneModel=cloneModel, tee=tee, keepfiles=keepfiles,
         solver_options=solver_options)

    if method == 'kaug':
        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)  #: important!
        m.ipopt_zU_in.update(m.ipopt_zU_out)  #: important!    

        kaug.options['dsdp_mode'] = ""  #: sensitivity mode!
        kaug.solve(m, tee=tee)

    perturb_parameters(method, m, paramList, perturbList)
    # ^ These are the user's paramList and perturbList...
    # Will they work on the cloned model?
    # I think so because I only use the names and values in this function.

    if method == 'sipopt':
        # Send the model to the ipopt_sens and collect the solution
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)

    elif method == 'kaug':
        dotsens.options["dsdp_mode"] = ""
        dotsens.solve(m, tee=tee) 

    return m

def setup_sensitivity(method, instance, paramList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    # Verify User Inputs    
    err_msg = ("Specified \"parmeters\" must be mutable parameters"
              "or fixed variables.")
    for param in paramList:
        if param.ctype is Param:
            if not param.mutable:
                raise ValueError(err_msg)
        elif param.ctype is Var:
            for data in _generate_data_objects(param):
                if not data.fixed:
                    # No _real_ reason we need the vars to be fixed...
                    raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)

    # Add model block to compartmentalize data needed by the sensitivity solver
    block = Block()
    #block_name = unique_component_name(instance, '_SENSITIVITY_TOOLBOX_DATA')
    block_name = unique_component_name(instance, '_' + method + '_data')
    # This will add a new block every time it is called on a model,
    # which is not what we want...
    instance.add_component(block_name, block)

    # Based on user input clone model or use orignal model for analysis
    if cloneModel:
        block.tmp_list = (paramList, perturbList)
        m = instance.clone()
        instance.del_component(block_name)
        block = m.component(block_name)
        paramList, perturbList = block.tmp_list
        del block.tmp_list
    else:
        m = instance

    # TODO: generate var, param list earlier
    userParams = list(param for param in paramList if param.ctype is Param)
    userVars = list(var for var in paramList if var.ctype is Var)

    # Iterate over user data, create var, param list.
    # The only real reason I don't do this here is that I need to preserve
    # the "user-provided" objects and order so I know how to use the
    # perturbations...

    # For every user-provided var we add a param, and for every user-
    # provided param we add a var.
    addedParams = list()
    for var in userVars:
        name = '_'.join((var.local_name, 'param'))
        if var.is_indexed():
            d = {k: value(var[k]) for k in var.index_set()}
            myParam = Param(var.index_set(), initialize=d)
        else:
            d = value(var)
            myParam = Param(initialize=d)
        block.add_component(name, myParam)
        addedParams.append(myParam)

    addedVars = list()
    for param in userParams:
        #name = '_'.join((param.local_name, 'var'))
        name = unique_component_name(block, param.local_name)
        if param.is_indexed():
            d = {k: value(param[k]) for k in param.index_set()}
            myVar = Var(param.index_set(), initialize=d)
        else:
            d = value(param)
            myVar = Var(initialize=d)
        block.add_component(name, myVar)
        addedVars.append(myVar)

    userParamData = list(_generate_data_objects(userParams))
    userVarData = list(_generate_data_objects(userVars))
    addedVarData = list(_generate_data_objects(addedVars))
    addedParamData = list(_generate_data_objects(addedParams))

    perturbedParamData = list(_generate_data_objects(perturbList))

    paramDataList = userParamData

    # Presumably, the user provided fixed vars as parameters.
    # We make sure these are unfixed.
    for var in userVarData:
        var.unfix()

    # Populate the dictionaries necessary for replacement
    paramDataIds = list(id(param) for param in userParamData)
    variableSubMap = dict(zip(paramDataIds, addedVarData))
    perturbSubMap = dict(zip(paramDataIds, perturbedParamData))
 
    # Visitor that we will use to replace user-provided parameters
    # in the objective and the constraints.
    param_replacer = ExpressionReplacementVisitor(
            substitute=variableSubMap,
            remove_named_expressions=True,
            )

    # clone Objective, add to Block, and update any Expressions
    for obj in list(m.component_data_objects(Objective,
                                            active=True,
                                            descend_into=True)):
        #tempName = unique_component_name(m, obj.local_name)
        # Should not need unique_component_name as we add this component
        # to our private block
        tempName = obj.local_name
        new_expr = param_replacer.dfs_postorder_stack(obj.expr)
        block.add_component(tempName, Objective(expr=new_expr))
        obj.deactivate()

    # clone Constraints, add to Block, and update any Expressions
    #
    # Unfortunate that this deactivates and replaces constraints
    # even if they don't contain the parameters.
    # In fact it will do this even if the user only specified fixed
    # variables.
    # 
    block.constList = ConstraintList()
    for con in list(m.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
        if con.equality:
            new_expr = param_replacer.dfs_postorder_stack(con.expr)
            block.constList.add(expr=new_expr)
        else:
            if con.lower is None or con.upper is None:
                new_expr = param_replacer.dfs_postorder_stack(con.expr)
                block.constList.add(expr=new_expr)
            else:
                # Constraint must be a ranged inequality, break into separate constraints
                new_body = param_replacer.dfs_postorder_stack(con.body)
                new_lower = param_replacer.dfs_postorder_stack(con.lower)
                new_upper = param_replacer.dfs_postorder_stack(con.upper)

                # Add constraint for lower bound
                block.constList.add(expr=(new_lower <= new_upper))

                # Add constraint for upper bound
                block.constList.add(expr=(new_upper >= new_body))
        con.deactivate()

    # paramData to varData constraint list
    block.paramConst = ConstraintList()
    sens_vardata = userVarData + addedVarData
    sens_paramdata = addedParamData + userParamData

    # Implementation as a ConstraintList destroys the structure of these
    # parameters, and I don't see any reason why it is necessary.
    for var, param in zip(sens_vardata, sens_paramdata):
        block.paramConst.add(var - param == 0)

    # Declare Suffixes
    _add_sensitivity_suffixes(m)
    
    # for reasons that are not entirely clear, 
    #     ipopt_sens requires the indices to start at 1
    # ^ presumably the same reason ConstraintList's indices
    # start at 1...
    for i, var in enumerate(sens_vardata):
        idx = i + 1
        con = block.paramConst[idx]
        param = sens_paramdata[i]

        # sipopt
        m.sens_state_0[var] = idx
        m.sens_state_1[var] = idx
        #m.sens_state_value_1[var] = value(perturbedParamData[i])
        m.sens_init_constr[con] = idx

        # k_aug
        m.dcdp[con] = idx
        #m.DeltaP[con] = value(param - perturbedParamData[i])
    
    return m

def perturb_parameters(method, instance, paramList, perturbList):
    """
    """
    #block_name = unique_component_name(instance, '_SENSITIVITY_TOOLBOX_DATA')
    block_name = '_' + method + '_data'
    block = instance.component(block_name)

    # NOTE that this doesn't work if we previously cloned the user's model.

    # TODO:
    # A better implementation would be to iterate over a list
    # that contains the information we need to locate the "parameter"
    # in the user-provided list.
    # We would just iterate over this list and put the proper value
    # in the corresponding value of the perturbation suffixes.
#    for var, param, con, list_idx, comp_idx in sens_data_list:
#        if comp_idx is _NotAnIndex:
#            ptb = value(perturbList[list_idx])
#        else:
#            ptb = value(perturbList[list_idx][comp_idx])
#
#        # sipopt
#        instance.sens_state_value_1[var] = ptb
#
#        # k_aug
#        #instance.DeltaP[con] = value(ptb - var)
#        instance.DeltaP[con] = value(var - ptb)
#        # FIXME: ^ This is incorrect. DeltaP should be (ptb - current).
#        # But at least one test doesn't pass unless I use (current - ptb).
    
    # Need to put perturbList in the right order. The right order depends
    # on which of the user-provided "parameters" are vars and which are params.
    varPerturbValues = list()
    paramPerturbValues = list()
    sens_vardata = list()
    assert len(paramList) == len(perturbList)
    for comp, ptb in zip(paramList, perturbList):
        if comp.ctype is Var: 
            if comp.is_indexed():
                try:
                    for idx in sorted_robust(comp):
                        # Assuming ptb is a parameter
                        varPerturbValues.append(value(ptb[idx]))
                        sens_vardata.append(comp[idx])
                except TypeError:
                    # ptb is a number
                    for idx in sorted_robust(comp):
                        varPerturbValues.append(value(ptb))
                        sens_vardata.append(comp[idx])
            else:
                # My intended use case: comp is a data object
                # and ptb is a number
                varPerturbValues.append(value(ptb))
                sens_vardata.append(comp)
                # ^ We just appended the component in the user's list.
                # This is presumably in the original model, not the cloned
                # model, which is what we need for the suffix to make sense.

        elif comp.ctype is Param:
            # This is the var we created for this param in the
            # previous function.
            var = block.component(comp.local_name)
            assert var is not None
            if comp.is_indexed():
                try:
                    for idx in sorted_robust(comp):
                        # Assuming ptb is a parameter
                        paramPerturbValues.append(value(ptb[idx]))
                        sens_vardata.append(var[idx])
                except TypeError:
                    # ptb is a number
                    for idx in sorted_robust(comp):
                        paramPerturbValues.append(value(ptb))
                        sens_vardata.append(var[idx])
            else:
                # My intended use case: comp is a data object
                # and ptb is a number
                paramPerturbValues.append(value(ptb))
                sens_vardata.append(var)

    # Because this is the order of our sens_vardata and paramConst
    perturbValues = varPerturbValues + paramPerturbValues

    for i, ptb in enumerate(perturbValues):
        var = sens_vardata[i]
        con = block.paramConst[i+1]

        # sipopt
        instance.sens_state_value_1[var] = ptb

        # k_aug
        #instance.DeltaP[con] = value(ptb - var)
        instance.DeltaP[con] = value(var - ptb)
        # FIXME: ^ This is incorrect. DeltaP should be (ptb - current).

def sensitivity_calculation_class(method, instance, paramList, perturbList,
         cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    sens = SensitivityInterface(method, instance, cloneModel=cloneModel)
    sens.setup_sensitivity(paramList)

    m = sens.model_instance

    if method == 'kaug':
        kaug = SolverFactory('k_aug', solver_io='nl')
        dotsens = SolverFactory('dot_sens', solver_io='nl')
        ipopt = SolverFactory('ipopt', solver_io='nl')

        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)  #: important!
        m.ipopt_zU_in.update(m.ipopt_zU_out)  #: important!    

        kaug.options['dsdp_mode'] = ""  #: sensitivity mode!
        kaug.solve(m, tee=tee)

    sens.perturb_parameters(perturbList)

    if method == 'sipopt':
        ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
        ipopt_sens.options['run_sens'] = 'yes'

        # Send the model to the ipopt_sens and collect the solution
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)

    elif method == 'kaug':
        dotsens.options["dsdp_mode"] = ""
        dotsens.solve(m, tee=tee) 

    return m

class SensitivityInterface(object):

    def __init__(self, method, instance, cloneModel=True):
        """
        """
        self.method = method
        self._original_model = instance
        # We need a pointer to the original model to properly handle the case
        # where instance is not a ConcreteModel but instead a general
        # block-like object.
        # The user will later provide a list of parameters, presumably
        # in the same block that they passed in. But if we have cloned the
        # model, these parameters will not be meaningful. We get around this
        # by finding components on the cloned model with the same CUID.
        # However, this only works if we can `find_component_on` the cloned
        # block, which means we need to construct the CUID in the `context`
        # of the block that we cloned.

        if cloneModel:
            # Note that we are not "cloning" the user's parameters
            # or perturbations.
            self.model_instance = instance.clone()
        else:
            self.model_instance = instance

    def get_default_block_name(self):
        # return '_SENSITIVITY_TOOLBOX_DATA'
        return '_'.join(('', self.method, 'data'))

    def get_default_var_name(self, name):
        #return '_'.join(('sens_var', name))
        return name

    def get_default_param_name(self, name):
        #return '_'.join(('sens_param', name))
        return name

    def setup_sensitivity(self, paramList):
        """
        """
        # We need to translate the components in paramList into
        # components in our possibly cloned model.
        orig = self._original_model
        instance = self.model_instance
        if orig is not instance:
            paramList = list(
                ComponentUID(param, context=orig).find_component_on(instance)
                for param in paramList
                )

        # If a sensitivity block already exists, and we have not done
        # any expression replacement, we delete the old block, re-fix the
        # sensitivity variables, and start again.
        existing_block = instance.component(self.get_default_block_name())
        if existing_block is not None:
            if (hasattr(existing_block, 'has_replaced_expressions') and
                    not existing_block.has_replaced_expressions):
                for var, _, _, _ in existing_block._sens_data_list:
                    # Re-fix variables that the previous block was
                    # treating as parameters.
                    var.fix()
                instance.del_component(existing_block)
            else:
                msg = ("Re-using sensitivity interface is not supported "
                        "when calculating sensitivity for mutable parameters. "
                        "Used fixed vars instead if you want to do this."
                        )
                raise RuntimeError(msg)

        block = Block()
        instance.add_component(self.get_default_block_name(), block)
        self.block = block
        block._has_replaced_expressions = False
        block._sens_data_list = []
        block._paramList = paramList

        sens_data_list = block._sens_data_list
        # This is a list of (vardata, paramdata, list_idx, comp_idx) tuples.
        # Its purpose is to match corresponding vars and params and
        # to map these to a component or value in the user-provided
        # lists.
        for i, comp in enumerate(paramList):
            if comp.ctype is Param:
                if not comp.mutable:
                    raise ValueError(
                            "Specified parameters must be mutable. "
                            "Got %s, which is not mutable." % comp.name
                            )
                # Add a param:
                if comp.is_indexed():
                    d = {k: value(comp[k]) for k in comp.index_set()}
                    var = Var(comp.index_set(), initialize=d)
                else:
                    d = value(comp)
                    var = Var(initialize=d)
                name = self.get_default_var_name(comp.local_name)
                name = unique_component_name(block, name)
                block.add_component(name, var)

                if comp.is_indexed():
                    sens_data_list.extend(
                            (var[idx], param, i, idx)
                            for idx, param in _generate_component_items(comp)
                            )
                else:
                    sens_data_list.append((var, comp, i, _NotAnIndex))

            elif comp.ctype is Var:
                for _, data in _generate_component_items(comp):
                    if not data.fixed:
                        raise ValueError(
                                "Specified \"parameter\" variables must be "
                                "fixed. Got %s, which is not fixed."
                                % comp.name
                                )
                # Add a var:
                if comp.is_indexed():
                    d = {k: value(comp[k]) for k in comp.index_set()}
                    param = Param(comp.index_set(), mutable=True, initialize=d)
                else:
                    d = value(comp)
                    param = Param(mutable=True, initialize=d)
                name = self.get_default_param_name(comp.local_name)
                name = unique_component_name(block, name)
                block.add_component(name, param)

                if comp.is_indexed():
                    sens_data_list.extend(
                            (var, param[idx], i, idx)
                            for idx, var in _generate_component_items(comp)
                            )
                else:
                    sens_data_list.append((comp, param, i, _NotAnIndex))

        for var, _, _, _ in sens_data_list:
            # This unfixes all variables, not just those the user added.
            var.unfix()

        # Map used to replace user-provided parameters.
        variableSubMap = dict((id(param), var)
                for var, param, list_idx, _ in sens_data_list
                if paramList[list_idx].ctype is Param)

        if variableSubMap:
            # We now replace the provided parameters in the user's
            # expressions. Only do this if we have to, i.e. the
            # user provided some parameters rather than all vars.

            # Visitor that we will use to replace user-provided parameters
            # in the objective and the constraints.
            param_replacer = ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True,
                    )
            # TODO: Flag to ExpressionReplacementVisitor to only replace
            # named expressions if a node has been replaced within that
            # expression.

            # clone Objective, add to Block, and update any Expressions
            for obj in list(instance.component_data_objects(Objective,
                                                    active=True,
                                                    descend_into=True)):
                tempName = unique_component_name(block, obj.local_name)
                new_expr = param_replacer.dfs_postorder_stack(obj.expr)
                block.add_component(tempName, Objective(expr=new_expr))
                obj.deactivate()

            # clone Constraints, add to Block, and update any Expressions
            #
            # Unfortunate that this deactivates and replaces constraints
            # even if they don't contain the parameters.
            # In fact it will do this even if the user only specified fixed
            # variables.
            # 
            block.constList = ConstraintList()
            for con in list(instance.component_data_objects(Constraint, 
                                           active=True,
                                           descend_into=True)):
                if con.equality:
                    new_expr = param_replacer.dfs_postorder_stack(con.expr)
                    block.constList.add(expr=new_expr)
                else:
                    if con.lower is None or con.upper is None:
                        new_expr = param_replacer.dfs_postorder_stack(con.expr)
                        block.constList.add(expr=new_expr)
                    else:
                        # Constraint must be a ranged inequality, break into
                        # separate constraints
                        new_body = param_replacer.dfs_postorder_stack(con.body)
                        new_lower = param_replacer.dfs_postorder_stack(con.lower)
                        new_upper = param_replacer.dfs_postorder_stack(con.upper)

                        # Add constraint for lower bound
                        block.constList.add(expr=(new_lower <= new_upper))

                        # Add constraint for upper bound
                        block.constList.add(expr=(new_upper >= new_body))
                con.deactivate()

            # Assume that we just replaced some params
            # TODO: underscore
            block._has_replaced_expressions = True

        block.paramConst = ConstraintList()
        for var, param, _, _ in sens_data_list:
            block.paramConst.add(var - param == 0)

        # Declare Suffixes
        _add_sensitivity_suffixes(instance)
        
        for i, (var, _, _, _) in enumerate(sens_data_list):
            idx = i + 1
            con = block.paramConst[idx]

            # sipopt
            instance.sens_state_0[var] = idx
            instance.sens_state_1[var] = idx
            instance.sens_init_constr[con] = idx

            # k_aug
            instance.dcdp[con] = idx


    def perturb_parameters(self, perturbList): 
        """
        """
        # Note that entries of perturbList need not be components
        # of the cloned model. All we need are the values.
        instance = self.model_instance
        sens_data_list = self.block._sens_data_list
        paramConst = self.block.paramConst

        for i, (var, param, list_idx, comp_idx) in enumerate(sens_data_list):
            con = paramConst[i+1]
            if comp_idx is _NotAnIndex:
                ptb = value(perturbList[list_idx])
            else:
                try:
                    ptb = value(perturbList[list_idx][comp_idx])
                except TypeError:
                    # If the user provided a scalar value to perturb
                    # an indexed component.
                    ptb = value(perturbList[list_idx])

            # sipopt
            instance.sens_state_value_1[var] = ptb

            # k_aug
            #instance.DeltaP[con] = value(ptb - var)
            instance.DeltaP[con] = value(var - ptb)
            # FIXME: ^ This is incorrect. DeltaP should be (ptb - current).
            # But at least one test doesn't pass unless I use (current - ptb).

sensitivity_calculation = sensitivity_calculation_class
