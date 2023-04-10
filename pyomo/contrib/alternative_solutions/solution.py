#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pe
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.alternative_solutions import aos_utils, var_utils

class Solution:
    """
    A class to store solutions from a Pyomo model.
    
    Attributes
    ----------
    variables : ComponentMap
        A map between Pyomo variables and their values for a solution.
    fixed_vars : ComponentSet
        The set of Pyomo variables that are fixed in a solution.
    objectives : ComponentMap
        A map between Pyomo objectives and their values for a solution.
    
    Methods
    -------
    pprint():
        Prints a solution.
    get_variable_name_values(self, ignore_fixed_vars=False):
        Get a dictionary of variable name-variable value pairs.
    get_fixed_variable_names(self):
        Get a list of fixed-variable names.
    def get_objective_name_values(self):
        Get a dictionary of objective name-objective value pairs.
    """
    
    def __init__(self, model, variable_list, ignore_fixed_vars=False, 
                 round_discrete_vars=True):
        """
        Constructs a Pyomo Solution object.

        Parameters
        ----------
            model : ConcreteModel
                A concrete Pyomo model.
            variables: A collection of Pyomo _GenereralVarData variables
                The variables for which the solution will be stored.
            ignore_fixed_vars : boolean
                Boolean indicating that fixed variables should not be added to 
                the solution.
            round_discrete_vars : boolean
                Boolean indicating that discrete values should be rounded to 
                the nearest integer in the solutions results.
        """

        aos_utils._is_concrete_model(model)
        assert isinstance(ignore_fixed_vars, bool), \
            'ignore_fixed_vars must be a Boolean'
        assert isinstance(round_discrete_vars, bool), \
            'round_discrete_vars must be a Boolean'
            
        self.variables = ComponentMap()
        self.fixed_vars = ComponentSet()
        for var in variable_list:
            if ignore_fixed_vars and var.is_fixed():
                continue
            if var.is_continuous() or not round_discrete_vars:
                self.variables[var] = pe.value(var)
            else:
                self.variables[var] = round(pe.value(var))
            if var.is_fixed():
                self.fixed_vars.add(var)
                
        self.objectives = ComponentMap()
        # TODO: Should inactive objectives be included?
        for obj in model.component_data_objects(pe.Objective, active=True):
            self.objectives[obj] = pe.value(obj)
            
    def pprint(self):
        '''Print the solution variable and objective values.'''
        fixed_string = "Yes"
        print("Variable, Value, Fixed?")
        for variable, value in self.variables.items():
            if variable in self.fixed_vars:
                print("{}, {}, {}".format(variable.name, value, fixed_string))
            else:
                print("{}, {}".format(variable.name, value))
        print()
        print("Objective, Value")
        for objective, value in self.objectives.items():
            print("{}, {}".format(objective.name, value))
            
    def get_variable_name_values(self, ignore_fixed_vars=False):
        '''Get a dictionary of variable name-variable value pairs.'''
        return {var.name: value for var, value in self.variables.items() if
                not (ignore_fixed_vars and var in self.fixed_vars)}
    
    def get_fixed_variable_names(self):
        '''Get a list of fixed-variable names.'''
        return [var.name for var in self.fixed_vars]
    
    def get_objective_name_values(self):
        '''Get a dictionary of objective name-objective value pairs.'''
        return {obj.name: value for obj, value in self.objectives.items()} 