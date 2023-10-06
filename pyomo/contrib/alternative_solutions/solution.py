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
from pyomo.contrib.alternative_solutions import aos_utils

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
    
    def __init__(self, model, variable_list, include_fixed=True, 
                 objective=None):
        """
        Constructs a Pyomo Solution object.

        Parameters
        ----------
            model : ConcreteModel
                A concrete Pyomo model.
            variable_list: A collection of Pyomo _GenereralVarData variables
                The variables for which the solution will be stored.
            include_fixed : boolean
                Boolean indicating that fixed variables should be added to the 
                solution.
            objective: None or Objective
                The objective functions for which the value will be saved. None
                indicates that the active objective should be used, but a
                different objective can be stored as well.
        """
        
        self.variables = ComponentMap()
        self.fixed_vars = ComponentSet()
        for var in variable_list:
            is_fixed = var.is_fixed()
            if is_fixed:
                self.fixed_vars.add(var)
            if include_fixed or not is_fixed:
                self.variables[var] = pe.value(var)
        
        if objective is None:
            objective = aos_utils._get_active_objective(model)
        self.objective = (objective, pe.value(objective))
    
    def _round_variable_value(self, variable, value, round_discrete=True):
        return value if not round_discrete or variable.is_continuous() \
            else round(value)
    
    def pprint(self, round_discrete=True):
        '''Print the solution variables and objective values.'''
        fixed_string = " (Fixed)"
        print()
        print("Variable\tValue")
        for variable, value in self.variables.items():
            fxd = fixed_string if variable in self.fixed_vars else ""
            val = self._round_variable_value(variable, value, round_discrete)
            print("{}\t\t\t{}{}".format(variable.name, val, fxd))
        print()
        print("Objective value for {} = {}".format(*self.objective))
            
    def get_variable_name_values(self, include_fixed=True, 
                                 round_discrete=True):
        '''Get a dictionary of variable name-variable value pairs.'''
        return {var.name: self._round_variable_value(var, val, round_discrete) 
                for var, val in self.variables.items() \
                    if include_fixed or not var in self.fixed_vars}
    
    def get_fixed_variable_names(self):
        '''Get a list of fixed-variable names.'''
        return [var.name for var in self.fixed_vars]