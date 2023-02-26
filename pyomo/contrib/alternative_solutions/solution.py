import pyomo.environ as pe
from pyomo.common.collections import ComponentMap, ComponentSet

def get_model_solution(model, variable_list, ignore_fixed_vars=False, 
                       round_discrete_vars=True):
    solution = {}
    variables = ComponentMap()
    fixed_vars = ComponentSet()
    for var in variable_list:
        if ignore_fixed_vars and var.is_fixed():
            continue
        if var.is_continuous() or not round_discrete_vars:
            variables[var] = pe.value(var)
        else:
            variables[var] = round(pe.value(var))
        if var.is_fixed():
            fixed_vars.add(var)
            
    solution["variables"] = variables
    solution["fixed_variables"] = fixed_vars

    objectives = ComponentMap()
    for obj in model.component_data_objects(pe.Objective, active=True):
        objectives[obj] = pe.value(obj)
    solution["objectives"] = objectives

    return solution

