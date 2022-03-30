#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import logging

import pyomo.environ as pyo
from pyomo.core.expr.current import (replace_expressions,
                                     identify_mutable_parameters)
from pyomo.environ import ComponentUID

logger = logging.getLogger(__name__)

def convert_params_to_vars(model, param_names=None, fix_vars=False):
    """
    Convert select Params to Vars
    
    Parameters
    ----------
    model : Pyomo concrete model
        Original model
    param_names : list of strings
        List of parameter names to convert, if None then all Params are converted
    fix_vars : bool
        Fix the new variables, default is False
        
    Returns
    -------
    model : Pyomo concrete model
        Model with select Params converted to Vars
    """
    
    model = model.clone()
    
    #print('--- Original Model ---') 
    #model.pprint()
    if param_names is None:
        param_names = [param.name for param in model.component_data_objects(pyo.Param)]
        
    # Convert Params to Vars, unfix Vars, and create a substitution map
    substitution_map = {}
    for i, theta in enumerate(param_names):
        # Leverage the parser in ComponentUID to locate the component.  
        theta_cuid = ComponentUID(theta)
        theta_object = theta_cuid.find_component_on(model)
        
        # Extract theta objects when indexed, there must be a better way to do this
        theta_objects = []
        if theta_object.is_indexed():
            for theta_obj in theta_object:
                theta_cuid = ComponentUID(theta + '[' + str(theta_obj) + ']')
                theta_objects.append(theta_cuid.find_component_on(model))
        else:
            theta_objects = [theta_object]
            
        for theta_object in theta_objects:
            if theta_object is None:
                logger.warning(
                    "theta_name[%s] (%s) was not found on the model",
                    (i, theta))
            
            elif theta_object.is_parameter_type():
                val = theta_object.value
                model.del_component(theta_object)        
                model.add_component(theta, pyo.Var(initialize = val))
                theta_var_cuid = ComponentUID(theta)
                theta_var_object = theta_var_cuid.find_component_on(model)
                substitution_map[id(theta_object)] = theta_var_object
                if fix_vars:
                    theta_var_object.fix()
                else:
                    theta_var_object.unfix()
                    
            elif theta_object.is_variable_type():
                if fix_vars:
                    theta_object.fix()
                else:
                    theta_object.unfix()
                
    if len(substitution_map) == 0:
        return model
    
    # Convert Params to Vars in Constraint expressions
    model.constraints = pyo.ConstraintList()
    for c in model.component_data_objects(pyo.Constraint):
        if c.active and any(v.name in param_names for v in identify_mutable_parameters(c.expr)):
            if c.equality:
                model.constraints.add(
                    replace_expressions(expr=c.lower, substitution_map=substitution_map) ==
                    replace_expressions(expr=c.body, substitution_map=substitution_map))
            elif c.lower is not None:
                model.constraints.add(
                    replace_expressions(expr=c.lower, substitution_map=substitution_map) <=
                    replace_expressions(expr=c.body, substitution_map=substitution_map))
            elif c.upper is not None:
                model.constraints.add(
                    replace_expressions(expr=c.upper, substitution_map=substitution_map) >=
                    replace_expressions(expr=c.body, substitution_map=substitution_map))
            else:
                raise ValueError("Unable to parse constraint to convert params to vars.")
            c.deactivate() 
    
    # Convert Params to Vars in Objective expressions
    for obj in model.component_data_objects(pyo.Objective):
        if obj.active and any(v.name in param_names for v in identify_mutable_parameters(obj)):
            expr = replace_expressions(expr=obj.expr, substitution_map=substitution_map)
            model.del_component(obj)        
            model.add_component(obj.name, pyo.Objective(rule=expr, sense=obj.sense))
     
    #print('--- Updated Model ---') 
    #model.pprint()
    #solver = pyo.SolverFactory('ipopt')
    #solver.solve(model)
    
    return model
