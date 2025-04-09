#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import json
import pyomo.environ as pyo
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
    objective : ComponentMap
        A map between Pyomo objectives and their values for a solution.

    Methods
    -------
    pprint():
        Prints a solution.
    get_variable_name_values(self, ignore_fixed_vars=False):
        Get a dictionary of variable name-variable value pairs.
    get_fixed_variable_names(self):
        Get a list of fixed-variable names.
    get_objective_name_values(self):
        Get a dictionary of objective name-objective value pairs.
    """

    def __init__(self, model, variable_list, include_fixed=True, objective=None):
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
                self.variables[var] = pyo.value(var)

        if objective is None:
            objective = aos_utils.get_active_objective(model)
        self.objective = (objective, pyo.value(objective))

    @property
    def objective_value(self):
        """
        Returns
        -------
            The value of the objective.
        """
        return self.objective[1]

    def pprint(self, round_discrete=True, sort_keys=True, indent=4):
        """
        Print the solution variables and objective values.

        Parameters
        ----------
            rounded_discrete : boolean
                If True, then round discrete variable values before printing.
        """
        print(
            self.to_string(
                round_discrete=round_discrete, sort_keys=sort_keys, indent=indent
            )
        )  # pragma: no cover

    def to_string(self, round_discrete=True, sort_keys=True, indent=4):
        return json.dumps(
            self.to_dict(round_discrete=round_discrete),
            sort_keys=sort_keys,
            indent=indent,
        )

    def to_dict(self, round_discrete=True):
        ans = {}
        ans["objective"] = str(self.objective[0])
        ans["objective_value"] = self.objective[1]
        soln = {}
        for variable, value in self.variables.items():
            val = self._round_variable_value(variable, value, round_discrete)
            soln[variable.name] = val
        ans["solution"] = soln
        ans["fixed_variables"] = [str(v) for v in self.fixed_vars]
        return ans

    def __str__(self):
        return self.to_string()

    __repn__ = __str__

    def get_variable_name_values(self, include_fixed=True, round_discrete=True):
        """
        Get a dictionary of variable name-variable value pairs.

        Parameters
        ----------
            include_fixed : boolean
                If True, then include fixed variables in the dictionary.
            round_discrete : boolean
                If True, then round discrete variable values in the dictionary.

        Returns
        -------
            Dictionary mapping variable names to variable values.
        """
        return {
            var.name: self._round_variable_value(var, val, round_discrete)
            for var, val in self.variables.items()
            if include_fixed or not var in self.fixed_vars
        }

    def get_fixed_variable_names(self):
        """
        Get a list of fixed-variable names.

        Returns
        -------
            A list of the variable names that are fixed.
        """
        return [var.name for var in self.fixed_vars]

    def _round_variable_value(self, variable, value, round_discrete=True):
        """
        Returns a rounded value unless the variable is discrete or rounded_discrete is False.
        """
        return value if not round_discrete or variable.is_continuous() else round(value)
