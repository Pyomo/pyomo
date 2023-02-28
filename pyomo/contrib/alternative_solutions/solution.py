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

class Solution:
    """
    A class to store solutions from a Pyomo model.
    
    Attributes
    ----------
    variables : ComponentMap
        A map between Pyomo variable objects and their values for a solution.
    fixed_vars : ComponentSet
        The set of Pyomo variables that are fixed in a solution.
    objectives : ComponentMap
        A map between Pyomo objective objects and their values for a solution.
    
    Methods
    -------
    pprint():
        Prints the solution.
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
        for obj in model.component_data_objects(pe.Objective, active=True):
            self.objectives[obj] = pe.value(obj)
            
    def pprint(self):
        '''Print the solution variable and objective values.'''
        fixed_string = "(fixed)"
        print("Variable: Value")
        for variable, value in self.variables.items():
            if variable in self.fixed_vars:
                print("{}: {} {}".format(variable.name, value, fixed_string))
            else:
                print("{}: {}".format(variable.name, value))
        print()
        print("Objective: Value")
        for objective, value in self.objectives.items():
            print("{}: {}".format(objective.name, value))